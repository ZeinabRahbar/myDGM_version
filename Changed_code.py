import pykeops
from pykeops.torch import LazyTensor
import os
import torch
import numpy
from torch import nn
import os.path as osp
import torch_geometric
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch.nn import Module, ModuleList, Sequential
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATConv
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
class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance="euclidean", sparse=True):
        super(DGM_d, self).__init__()

        self.sparse=sparse

        self.temperature = nn.Parameter(torch.tensor(1. if distance=="hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.k = k
        self.distance = distance

        self.debug=False

    def forward(self, x, A, not_used=None, fixedges=None):
        if x.shape[0]==1:
            x = x[0]
        x = self.embed_f(x,A)
        if x.dim()==2:
            x = x[None,...]

        if self.training:
            if fixedges is not None:
                return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
            #sampling here
            edges_hat, logprobs = self.sample_without_replacement(x)

        else:
            with torch.no_grad():
                if fixedges is not None:
                    return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
                #sampling here
                edges_hat, logprobs = self.sample_without_replacement(x)

        if self.debug:
            if self.distance=="euclidean":
                D, _x = pairwise_euclidean_distances(x)
            if self.distance=="hyperbolic":
                D, _x = pairwise_poincare_distances(x)

            self.D = (D * torch.exp(torch.clamp(self.temperature,-5,5))).detach().cpu()
            self.edges_hat=edges_hat.detach().cpu()
            self.logprobs=logprobs.detach().cpu()

        return x, edges_hat, logprobs


    def sample_without_replacement(self, x):

        b,n,_ = x.shape

        if self.distance=="euclidean":
            G_i = LazyTensor(x[:, :, None, :])    # (M**2, 1, 2)
            X_j = LazyTensor(x[:, None, :, :])    # (1, N, 2)

            mD = ((G_i - X_j) ** 2).sum(-1)

            #argKmin already add gumbel noise
            lq = mD * torch.exp(torch.clamp(self.temperature,-5,5))
            indices = lq.argKmin(self.k, dim=1)

            x1 = torch.gather(x, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x.shape[-1]))
            x2 = x[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x.shape[-1])
            logprobs = (-(x1-x2).pow(2).sum(-1) * torch.exp(torch.clamp(self.temperature,-5,5))).reshape(x.shape[0],-1,self.k)

        if self.distance=="hyperbolic":
            pass
            x_norm = (x**2).sum(-1,keepdim=True)
            x_norm = (x_norm.sqrt()-1).relu() + 1
            x = x/(x_norm*(1+1e-2)) #safe distance to the margin
            x_norm = (x**2).sum(-1,keepdim=True)

            G_i = LazyTensor(x[:, :, None, :])    # (M**2, 1, 2)
            X_j = LazyTensor(x[:, None, :, :])    # (1, N, 2)

            G_i2 = LazyTensor(1-x_norm[:, :, None, :])    # (M**2, 1, 2)
            X_j2 = LazyTensor(1-x_norm[:, None, :, :])    # (1, N, 2)

            pq = ((G_i - X_j) ** 2).sum(-1)
            N = (G_i2*X_j2)
            XX = (1e-6+1+2*pq/N)
            mD = (XX+(XX**2-1).sqrt()).log()**2

            lq = mD * torch.exp(torch.clamp(self.temperature,-5,5))
            indices = lq.argKmin(self.k, dim=1)

            x1 = torch.gather(x, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x.shape[-1]))
            x2 = x[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x.shape[-1])

            x1_n = torch.gather(x_norm, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x_norm.shape[-1]))
            x2_n = x_norm[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x_norm.shape[-1])

            pq = (x1-x2).pow(2).sum(-1)
            pqn = ((1-x1_n)*(1-x2_n)).sum(-1)
            XX = 1e-6+1+2*pq/pqn
            dist = torch.log(XX+(XX**2-1).sqrt())**2
            logprobs = (-dist * torch.exp(torch.clamp(self.temperature,-5,5))).reshape(x.shape[0],-1,self.k)

            if self.debug:
                self._x=x.detach().cpu()+0


        rows = torch.arange(n).view(1,n,1).to(x.device).repeat(b,1,self.k)
        edges = torch.stack((indices.view(b,-1),rows.view(b,-1)),-2)

        if self.sparse:
            return (edges+(torch.arange(b).to(x.device)*n)[:,None,None]).transpose(0,1).reshape(2,-1), logprobs
        return edges, logprobs

class DGM_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(DGM_Model,self).__init__()

        self.save_hyperparameters(hparams)
        conv_layers = hparams["conv_layers"]
        fc_layers = hparams["fc_layers"]
        dgm_layers = hparams["dgm_layers"]
        k = hparams["k"]
        self.graph_f = ModuleList()
        self.node_g = ModuleList()
        for i,(dgm_l,conv_l) in enumerate(zip(dgm_layers,conv_layers)):
            if len(dgm_l)>0:
                if 'ffun' not in hparams or hparams["ffun"] == 'gcn':
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0],dgm_l[-1]),k=hparams["k"],distance=hparams["distance"]))
                if hparams["ffun"] == 'gat':
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0],dgm_l[-1]),k=hparams["k"],distance=hparams["distance"]))
                if hparams["ffun"] == 'mlp':
                    self.graph_f.append(DGM_d(MLP(dgm_l),k=hparams["k"],distance=hparams["distance"]))
                if hparams["ffun"] == 'knn':
                    self.graph_f.append(DGM_d(Identity(retparam=0),k=hparams["k"],distance=hparams["distance"]))
            else:
                self.graph_f.append(Identity())
            if hparams["gfun"] == 'edgeconv':
                conv_l=conv_l.copy()
                conv_l[0]=conv_l[0]*2
                self.node_g.append(EdgeConv(MLP(conv_l), hparams["pooling"]))
            if hparams["gfun"] == 'gcn':
                self.node_g.append(GCNConv(conv_l[0],conv_l[1]))
            if hparams["gfun"] == 'gat':
                self.node_g.append(GATConv(conv_l[0],conv_l[1]))
        self.fc = MLP(fc_layers, final_activation=False)
        if hparams["pre_fc"] is not None and len(hparams["pre_fc"])>0:
            self.pre_fc = MLP(hparams["pre_fc"], final_activation=True)
        self.avg_accuracy = None
        self.automatic_optimization = False
        self.debug=False

    def forward(self,x, edges=None):
        if self.hparams["pre_fc"] is not None and len(self.hparams["pre_fc"])>0:
            x = self.pre_fc(x)
        graph_x = x.detach()
        lprobslist = []
        for f,g in zip(self.graph_f, self.node_g):
            graph_x,edges,lprobs = f(graph_x,edges,None)
            b,n,d = x.shape
            self.edges=edges
            x = torch.nn.functional.relu(g(torch.dropout(x.view(-1,d), self.hparams["dropout"], train=self.training), edges)).view(b,n,-1)
            graph_x = torch.cat([graph_x,x.detach()],-1)
            if lprobs is not None:
                lprobslist.append(lprobs)
        return self.fc(x),torch.stack(lprobslist,-1) if len(lprobslist)>0 else None
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        X, y, mask, edges = train_batch
        edges = edges[0]
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        pred,logprobs = self(X,edges)
        train_pred = pred[:,mask.to(torch.bool),:]
        train_lab = y[:,mask.to(torch.bool),:]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(train_pred,train_lab)
        loss.backward()
        correct_t = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().mean().item()
        if logprobs is not None:
            corr_pred = (train_pred.argmax(-1)==train_lab.argmax(-1)).float().detach()
            wron_pred = (1-corr_pred)
            if self.avg_accuracy is None:
                self.avg_accuracy = torch.ones_like(corr_pred)*0.5
            point_w = (self.avg_accuracy-corr_pred)#*(1*corr_pred + self.k*(1-corr_pred))
            graph_loss = point_w * logprobs[:,mask.to(torch.bool),:].exp().mean([-1,-2])
            graph_loss = graph_loss.mean()# + self.graph_f[0].Pr.abs().sum()*1e-3
            graph_loss.backward()
            self.log('train_graph_loss', graph_loss.detach().cpu())
            if(self.debug):
                self.point_w = point_w.detach().cpu()
            self.avg_accuracy = self.avg_accuracy.to(corr_pred.device)*0.95 +  0.05*corr_pred
        optimizer.step()
        self.log('train_acc', correct_t)
        self.log('train_loss', loss.detach().cpu())

    def test_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        edges = edges[0]
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        pred,logprobs = self(X,edges)
        pred = pred.softmax(-1)
        for i in range(1,self.hparams["test_eval"]):
            pred_,logprobs = self(X,edges)
            pred+=pred_.softmax(-1)
        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool),:]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        self.log('test_loss', loss.detach().cpu())
        self.log('test_acc', 100*correct_t)

    def validation_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        edges = edges[0]
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        pred,logprobs = self(X,edges)
        pred = pred.softmax(-1)
        for i in range(1,self.hparams["test_eval"]):
            pred_,logprobs = self(X,edges)
            pred+=pred_.softmax(-1)
        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool),:]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        self.log('val_loss', loss.detach())
        self.log('val_acc', 100*correct_t)
    def test_dataloader(self):
        """
        Implement this method to return the test DataLoader.
        """
        test_data = TadpoleDataset(train=False, samples_per_epoch=1, device=self.device)
        return DataLoader(test_data, batch_size=1)

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import mobilenet_v2
import time

timy = time.time()

class TadpoleDataset(torch.utils.data.Dataset):
    def __init__(self, fold=0, train=True, samples_per_epoch=10, device='cuda', full=False):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to (224, 224)
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if train:
            split = 'train'
        else:
            split = 'test'

        self.train_dataset = datasets.MNIST(root='./data', train=(split == 'train'), download=True, transform=transform)
        self.test_dataset = datasets.MNIST(root='./data', train=(split == 'test'), download=True, transform=transform)

        self.n_features = 1280
        self.num_classes = 10

        if split == 'train':
            self.mask = torch.zeros(70000, dtype=torch.float32)
            self.mask[:1000] = 1  # Set the first 60,000 elements to 1
        else:
            self.mask = torch.zeros(70000, dtype=torch.float32)
            self.mask[1000:10000] = 1  # Set the first 60,000 elements to 1

        self.model = mobilenet_v2(pretrained=True)  # Load MobileNet-V2 pretrained model
        self.model.to(device).eval()  # Set the model to evaluation mode

        self.X_train = self.vectorize_images(self.train_dataset.data, device)
        self.y_train = torch.eye(self.num_classes)[self.train_dataset.targets].float().to(device)

        self.X_test = self.vectorize_images(self.test_dataset.data, device)
        self.y_test = torch.eye(self.num_classes)[self.test_dataset.targets].float().to(device)

        self.X = torch.cat([self.X_train, self.X_test], dim=0)
        self.y = torch.cat([self.y_train, self.y_test], dim=0)

        self.samples_per_epoch = samples_per_epoch

    def vectorize_images(self, images, device):
        num_images = images.shape[0]
        features = []

        with torch.no_grad():  # Disable gradient tracking
            for i in range(num_images):
                image = images[i].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float().to(device)
                feature = self.model.features(image).squeeze()
                features.append(feature)

        return torch.stack(features, dim=0)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X, self.y, self.mask, [[]]

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class TadpoleDataset(Dataset):
    def __init__(self, fold=0, train=True, samples_per_epoch=10, device='cuda', full=False):
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize images to (28, 28) for MNIST
            transforms.ToTensor()
        ])
        
        # Load MNIST dataset
        self.dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        # Shuffle dataset
        indices = torch.randperm(len(self.dataset))
        self.dataset.data = self.dataset.data[indices]
        self.dataset.targets = self.dataset.targets[indices]

        # Create masks for training and testing
        self.mask = torch.zeros(len(self.dataset), dtype=torch.float32)
        
        if train:
            self.mask[:1000] = 1  # Set first 1000 elements to 1 for training
            self.data = self.dataset.data[:1000].view(-1, 784).float()  # Vectorize to shape (1000, 784)
            self.targets = self.dataset.targets[:1000]
        else:
            self.mask[1000:] = 1  # Set remaining elements to 1 for testing
            self.data = self.dataset.data[1000:].view(-1, 784).float()  # Vectorize to shape (N-1000, 784)
            self.targets = self.dataset.targets[1000:]

        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return min(self.samples_per_epoch, len(self.data))

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.mask[idx], []
        
os.environ["CUDA_VISIBLE_DEVICES"]="0";
def run_training_process(run_params):
    if run_params["dataset"] == 'tadpole':
        train_data = TadpoleDataset(fold=0,train=True, device='cuda')
        val_data = test_data = TadpoleDataset(fold = 0, train=False,samples_per_epoch=1)
    train_loader = DataLoader(train_data, batch_size=1,num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)
    class MyDataModule(pl.LightningDataModule):
        def setup(self,stage=None):
            pass
        def train_dataloader(self):
            return train_loader
        def val_dataloader(self):
            return val_loader
        def test_dataloader(self):
            return test_loader
    if run_params["pre_fc"] is None or len(run_params["pre_fc"])==0:
        if len(run_params["dgm_layers"][0])>0:
            run_params["dgm_layers"][0][0]=train_data.n_features
        run_params["conv_layers"][0][0]=train_data.n_features
    else:
        run_params["pre_fc"][0]=train_data.n_features
    run_params["fc_layers"][-1] = train_data.num_classes
    model = DGM_Model(run_params)
    checkpoint_callback = ModelCheckpoint(save_last=True,save_top_k=1,verbose=True,monitor='val_loss',mode='min')
    early_stop_callback = EarlyStopping( monitor='val_loss', min_delta=0.00,patience=20, verbose=False, mode='min')
    callbacks = [checkpoint_callback,early_stop_callback]
    if val_data==test_data:
        callbacks = None
    logger = TensorBoardLogger("logs/")
    trainer = pl.Trainer(max_epochs=run_params['max_epochs'], logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=MyDataModule())
    trainer.test()
run_params = {
    'gpus': 1,
    'log_every_n_steps': 100,
    'max_epochs': 100,
    'progress_bar_refresh_rate': 10,
    'check_val_every_n_epoch': 1,
    'num_gpus': 10,
    'dataset': 'tadpole',
    'conv_layers': [[32, 32], [32, 16], [16, 8]],
    'dgm_layers': [[32, 16, 4], [], []],
    'fc_layers': [8, 8, 3],
    'pre_fc': [-1, 32],
    'gfun': 'gat',
    'ffun': 'mlp',
    'k': 5,
    'pooling': 'add',
    'distance': 'euclidean',
    'dropout': 0.0,
    'lr': 1e-2,
    'test_eval': 10,
    'default_root_path': './log'
}
for counter in range(10):
    run_training_process(run_params)
