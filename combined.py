import os
import torch
import numpy
import pykeops
from torch import nn
import os.path as osp
import torch_geometric
from argparse import Namespace
import pytorch_lightning as pl
from pykeops.torch import LazyTensor
from argparse import ArgumentParser
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch.nn import Module, ModuleList, Sequential
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATConv
import torch_geometric.nn as gnn

# Define the pairwise Euclidean distance function

def pairwise_euclidean_distances(x, dim=-1):
    return torch.cdist(x,x)**2, x        
    
class MLP(nn.Module):
    def __init__(self, layers_size, final_activation=False, dropout=0):
        super(MLP, self).__init__()
        layers = [nn.Dropout(dropout) if dropout > 0 else None] + [
            nn.Linear(layers_size[li - 1], layers_size[li])
            for li in range(1, len(layers_size))
        ] + [nn.LeakyReLU(0.1) if li != len(layers_size) - 1 or final_activation else None for li in range(1, len(layers_size))]
        self.MLP = nn.Sequential(*[layer for layer in layers if layer is not None])
    def forward(self, x, e=None):
        x = self.MLP(x)
        return x
class Identity(nn.Module):
    def __init__(self, retparam=None):
        super(Identity, self).__init__()
        self.retparam = retparam
    def forward(self, *params):
        return params[self.retparam] if self.retparam is not None else params
# Define the GCNRW module
class GCNRW(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=6000, lambda_=10):
        super(GCNRW, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.lambda_ = lambda_
        self.W = nn.Parameter(torch.Tensor(n_features, n_hidden).uniform_(-1, 1))

    def forward(self, A, X, Y):
        A_hat = A + torch.eye(A.size(0)).to(A.device)
        A_hat2 = torch.mm(A_hat, A_hat)
        H = torch.sigmoid(-torch.mm(torch.mm(A_hat2, X), self.W))

        self.b = None

        if Y.shape[1] >= self.n_hidden:
            self.b = torch.inverse(H.t() @ H + self.lambda_*torch.eye(self.n_hidden)) @ H.t() @ Y
        else:
            self.b = H.t() @ torch.inverse(H @ H.t() + self.lambda_*torch.eye(H.shape[0])) @ Y

        Z = torch.mm(H, self.b)
        return Z

    def fit(self, A, X, Y):
        A = A.to(X.device)
        X = X.to(A.device)
        Y = Y.to(A.device)
        self.b = None  # Initialize output weights
        Z = self.forward(A, X,Y)  

        return Z

    def predict(self, A, X):
        A = A.to(X.device)
        X = X.to(A.device)
        A_hat = A + torch.eye(A.size(0)).to(A.device)
        A_hat2 = torch.mm(A_hat, A_hat)
        H = torch.sigmoid(-torch.mm(torch.mm(A_hat2, X), self.W))
        Z = torch.mm(H, self.b)
        return Z

# Define the DGM_d module
class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance=pairwise_euclidean_distances, sparse=True):
        super(DGM_d, self).__init__()
        self.sparse = sparse
        self.temperature = nn.Parameter(torch.tensor(1. if distance == "hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.k = k
        self.debug = False
        self.distance = pairwise_euclidean_distances

    def forward(self, x, A, not_used=None, fixedges=None):
        x = self.embed_f(x, A)
        if self.training:
            if fixedges is not None:
                return x, fixedges, torch.zeros(fixedges.shape[0], fixedges.shape[-1] // self.k, self.k, dtype=torch.float, device=x.device)
            D, _x = self.distance(x)
            edges_hat, logprobs = self.sample_without_replacement(D)
        else:
            with torch.no_grad():
                if fixedges is not None:
                    return x, fixedges, torch.zeros(fixedges.shape[0], fixedges.shape[-1] // self.k, self.k, dtype=torch.float, device=x.device)
                D, _x = self.distance(x)
                # Sampling here
                edges_hat, logprobs = self.sample_without_replacement(D)
        if self.debug:
            self.D = D
            self.edges_hat = edges_hat
            self.logprobs = logprobs
            self.x = x
        return x, edges_hat, logprobs

    def sample_without_replacement(self, logits):
        b, n, _ = logits.shape
        logits = logits * torch.exp(torch.clamp(self.temperature, -5, 5))
        q = torch.rand_like(logits) + 1e-8
        lq = (logits - torch.log(-torch.log(q)))
        logprobs, indices = torch.topk(-lq, self.k)
        rows = torch.arange(n).view(1, n, 1).to(logits.device).repeat(b, 1, self.k)
        edges = torch.stack((indices.view(b, -1), rows.view(b, -1)), -2)
        if self.sparse:
            return (edges + (torch.arange(b).to(logits.device) * n)[:, None, None]).transpose(0, 1).reshape(2, -1), logprobs
        return edges, logprobs

# Define the DGM_Model
class DGM_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(DGM_Model, self).__init__()
        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        conv_layers = hparams.conv_layers
        fc_layers = hparams.fc_layers
        dgm_layers = hparams.dgm_layers
        k = hparams.k
        self.graph_f = ModuleList()
        self.node_g = ModuleList()
        for i, (dgm_l, conv_l) in enumerate(zip(dgm_layers, conv_layers)):
            if len(dgm_l) > 0:
                self.graph_f.append(DGM_d(GCNConv(dgm_l[0], dgm_l[-1]), k=hparams.k, distance=hparams.distance))
            else:
                self.graph_f.append(Identity())
            self.node_g.append(GCNConv(conv_l[0], conv_l[1]))
        # Replace MLP with GCNRW
        self.fc = GCNRW(fc_layers[-2], fc_layers[-1])
        if hparams.pre_fc is not None and len(hparams.pre_fc) > 0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.avg_accuracy = None
        self.automatic_optimization = False
        self.debug = False

    def forward(self, x, edges=None):
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc) > 0:
            x = self.pre_fc(x)
        graph_x = x.detach()
        lprobslist = []
        for f, g in zip(self.graph_f, self.node_g):
            graph_x, edges, lprobs = f(graph_x, edges, None)
            b, n, d = x.shape
            self.edges = edges
            x = torch.nn.functional.relu(g(torch.dropout(x.view(-1, d), self.hparams.dropout, train=self.training), edges)).view(b, n, -1)
            graph_x = torch.cat([graph_x, x.detach()], -1)
            if lprobs is not None:
                lprobslist.append(lprobs)
        return self.fc(x, edges[0]), torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        X, y, mask, edges = train_batch
        edges = edges[0]
        assert X.shape[0] == 1  # only works in transductive setting
        mask = mask[0]
        pred, logprobs = self(X, edges)
        train_pred = pred[:, mask.to(torch.bool), :]
        train_lab = y[:, mask.to(torch.bool), :]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(train_pred, train_lab)
        loss.backward()
        correct_t = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().mean().item()
        if logprobs is not None:
            corr_pred = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().detach()
            wron_pred = (1 - corr_pred)
            if self.avg_accuracy is None:
                self.avg_accuracy = torch.ones_like(corr_pred) * 0.5
            point_w = (self.avg_accuracy - corr_pred)  # *(1*corr_pred + self.k*(1-corr_pred))
            graph_loss = point_w * logprobs[:, mask.to(torch.bool), :].exp().mean([-1, -2])
            graph_loss = graph_loss.mean()  # + self.graph_f[0].Pr.abs().sum()*1e-3
            graph_loss.backward()
            self.log('train_graph_loss', graph_loss.detach().cpu())
            if self.debug:
                self.point_w = point_w.detach().cpu()
            self.avg_accuracy = self.avg_accuracy.to(corr_pred.device) * 0.95 + 0.05 * corr_pred
        optimizer.step()
        self.log('train_acc', correct_t)
        self.log('train_loss', loss.detach().cpu())

    def test_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        edges = edges[0]
        assert X.shape[0] == 1  # only works in transductive setting
        mask = mask[0]
        pred, logprobs = self(X, edges)
        pred = pred.softmax(-1)
        for i in range(1, self.hparams.test_eval):
            pred_, logprobs = self(X, edges)
            pred += pred_.softmax(-1)
        test_pred = pred[:, mask.to(torch.bool), :]
        test_lab = y[:, mask.to(torch.bool), :]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred, test_lab)
        self.log('test_loss', loss.detach().cpu())
        self.log('test_acc', 100 * correct_t)

    def validation_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        edges = edges[0]
        assert X.shape[0] == 1  # only works in transductive setting
        mask = mask[0]
        pred, logprobs = self(X, edges)
        pred = pred.softmax(-1)
        for i in range(1, self.hparams.test_eval):
            pred_, logprobs = self(X, edges)
            pred += pred_.softmax(-1)
        test_pred = pred[:, mask.to(torch.bool), :]
        test_lab = y[:, mask.to(torch.bool), :]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred, test_lab)
        self.log('val_loss', loss.detach())
        self.log('val_acc', 100 * correct_t)

# Define the get_planetoid_dataset function
def get_planetoid_dataset(name, normalize_features=True, transform=None, split="complete"):
    path = osp.join('.', 'data', name)
    if split == 'complete':
        dataset = Planetoid(path, name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1
    else:
        dataset = Planetoid(path, name, split=split)
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset

# Define the one_hot_embedding function
def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]

# Define the PlanetoidDataset class
class PlanetoidDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', samples_per_epoch=100, name='Cora', device='cpu'):
        dataset = get_planetoid_dataset(name)
        self.X = dataset[0].x.float().to(device)
        self.y = one_hot_embedding(dataset[0].y, dataset.num_classes).float().to(device)
        self.edge_index = dataset[0].edge_index.to(device)
        self.n_features = dataset[0].num_node_features
        self.num_classes = dataset.num_classes
        if split == 'train':
            self.mask = dataset[0].train_mask.to(device)
        if split == 'val':
            self.mask = dataset[0].val_mask.to(device)
        if split == 'test':
            self.mask = dataset[0].test_mask.to(device)
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X, self.y, self.mask, self.edge_index

# Set the GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define the run_training_process function
def run_training_process(run_params):
    train_data = None
    test_data = None
    if run_params.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        train_data = PlanetoidDataset(split='train', name=run_params.dataset, device='cuda')
        val_data = PlanetoidDataset(split='val', name=run_params.dataset, samples_per_epoch=1)
        test_data = PlanetoidDataset(split='test', name=run_params.dataset, samples_per_epoch=1)
    train_loader = DataLoader(train_data, batch_size=1, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)

    class MyDataModule(pl.LightningDataModule):
        def setup(self, stage=None):
            pass

        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val_loader

        def test_dataloader(self):
            return test_loader

    if run_params.pre_fc is None or len(run_params.pre_fc) == 0:
        if len(run_params.dgm_layers[0]) > 0:
            run_params.dgm_layers[0][0] = train_data.n_features
        run_params.conv_layers[0][0] = train_data.n_features
    else:
        run_params.pre_fc[0] = train_data.n_features
    run_params.fc_layers[-1] = train_data.num_classes

    model = DGM_Model(run_params)
    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=1, verbose=True, monitor='val_loss', mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=20, verbose=False, mode='min')
    callbacks = [checkpoint_callback, early_stop_callback]
    if val_data == test_data:
        callbacks = None
    logger = TensorBoardLogger("logs/")
    trainer = pl.Trainer.from_argparse_args(run_params, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=MyDataModule())
    trainer.test()

# Run the training process
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args(['--gpus', '1', '--log_every_n_steps', '100', '--max_epochs', '100', '--progress_bar_refresh_rate', '10', '--check_val_every_n_epoch', '1'])
    parser.add_argument("--num_gpus", default=10, type=int)
    parser.add_argument("--dataset", default='Cora')
    parser.add_argument("--conv_layers", default=[[32, 32], [32, 16], [16, 8]], type=lambda x: eval(x))
    parser.add_argument("--dgm_layers", default=[[32, 16, 4], [], []], type=lambda x: eval(x))
    parser.add_argument("--fc_layers", default=[8, 8, 3], type=lambda x: eval(x))
    parser.add_argument("--pre_fc", default=[-1, 32], type=lambda x: eval(x))
    parser.add_argument("--gfun", default='gcn')
    parser.add_argument("--ffun", default='gcn')
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--pooling", default='add')
    parser.add_argument("--distance", default='euclidean')
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--test_eval", default=10, type=int)
    parser.set_defaults(default_root_path='./log')
    params = parser.parse_args(namespace=params)
    run_training_process(params)
