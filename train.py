import os
import torch
import numpy as np

import torch_geometric
from torch import nn
from torch.nn import Module, ModuleList, Sequential
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATConv
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from argparse import Namespace

import torch
import numpy
import torch

from torch.nn import Module, ModuleList, Sequential
from torch import nn
import pykeops
from pykeops.torch import LazyTensor
#Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x,x)**2
    return dist, x

# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x**2).sum(dim,keepdim=True)
    x_norm = (x_norm.sqrt()-1).relu() + 1 
    x = x/(x_norm*(1+1e-2))
    x_norm = (x**2).sum(dim,keepdim=True)
    
    pq = torch.cdist(x,x)**2
    dist = torch.arccosh(1e-6+1+2*pq/((1-x_norm)*(1-x_norm.transpose(-1,-2))))**2
    return dist, x

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
    
class DGM_c(nn.Module):
    input_dim = 4
    debug=False
    
    def __init__(self, embed_f, k=None, distance="euclidean"):
        super(DGM_c, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(1).float())
        self.threshold = nn.Parameter(torch.tensor(0.5).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.distance = distance
        
        self.scale = nn.Parameter(torch.tensor(-1).float(),requires_grad=False)
        self.centroid = nn.Parameter(torch.zeros((1,1,DGM_c.input_dim)).float(),requires_grad=False)
        
        
    def forward(self, x, A, not_used=None, fixedges=None):
        
        x = self.embed_f(x,A)  
        
        # estimate normalization parameters
        if self.scale <0:            
            self.centroid.data = x.mean(-2,keepdim=True).detach()
            self.scale.data = (0.9/(x-self.centroid).abs().max()).detach()
        
        if self.distance=="hyperbolic":
            D, _x = pairwise_poincare_distances((x-self.centroid)*self.scale)
        else:
            D, _x = pairwise_euclidean_distances((x-self.centroid)*self.scale)
            
        A = torch.sigmoid(self.temperature*(self.threshold.abs()-D))
        
        if DGM_c.debug:
            self.A = A.data.cpu()
            self._x = _x.data.cpu()

        return x, A, None
 
 
class MLP(nn.Module): 
    def __init__(self, layers_size,final_activation=False, dropout=0):
        super(MLP, self).__init__()
        layers = []
        for li in range(1,len(layers_size)):
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layers_size[li-1],layers_size[li]))
            if li==len(layers_size)-1 and not final_activation:
                continue
            layers.append(nn.LeakyReLU(0.1))
        self.MLP = nn.Sequential(*layers)
        
    def forward(self, x, e=None):
        x = self.MLP(x)
        return x
    
class Identity(nn.Module):
    def __init__(self,retparam=None):
        self.retparam=retparam
        super(Identity, self).__init__()
        
    def forward(self, *params):
        if self.retparam is not None:
            return params[self.retparam]
        return params

from torch.nn import Module, ModuleList, Sequential
from torch import nn

    
class DGM_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(DGM_Model,self).__init__()
        
        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)
        
        self.save_hyperparameters(hparams)
        conv_layers = hparams.conv_layers
        fc_layers = hparams.fc_layers
        dgm_layers = hparams.dgm_layers
        k = hparams.k

            
        self.graph_f = ModuleList() 
        self.node_g = ModuleList() 
        for i,(dgm_l,conv_l) in enumerate(zip(dgm_layers,conv_layers)):
            if len(dgm_l)>0:
                if 'ffun' not in hparams or hparams.ffun == 'gcn':
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'gat':
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'mlp':
                    self.graph_f.append(DGM_d(MLP(dgm_l),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'knn':
                    self.graph_f.append(DGM_d(Identity(retparam=0),k=hparams.k,distance=hparams.distance))
            else:
                self.graph_f.append(Identity())
            
            if hparams.gfun == 'edgeconv':
                conv_l=conv_l.copy()
                conv_l[0]=conv_l[0]*2
                self.node_g.append(EdgeConv(MLP(conv_l), hparams.pooling))
            if hparams.gfun == 'gcn':
                self.node_g.append(GCNConv(conv_l[0],conv_l[1]))
            if hparams.gfun == 'gat':
                self.node_g.append(GATConv(conv_l[0],conv_l[1]))
        
        self.fc = MLP(fc_layers, final_activation=False)
        if hparams.pre_fc is not None and len(hparams.pre_fc)>0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.avg_accuracy = None
        
        self.automatic_optimization = False
        self.debug=False
        
    def forward(self,x, edges=None):
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc)>0:
            x = self.pre_fc(x)
            
        graph_x = x.detach()
        lprobslist = []
        for f,g in zip(self.graph_f, self.node_g):
            graph_x,edges,lprobs = f(graph_x,edges,None)
            b,n,d = x.shape
            self.edges=edges
            x = torch.nn.functional.relu(g(torch.dropout(x.view(-1,d), self.hparams.dropout, train=self.training), edges)).view(b,n,-1)
            graph_x = torch.cat([graph_x,x.detach()],-1)
            if lprobs is not None:
                lprobslist.append(lprobs)
                
        return self.fc(x),torch.stack(lprobslist,-1) if len(lprobslist)>0 else None
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        
        X, y, mask, edges = train_batch
        print("e",edges.ahape)
        edges = edges[0]
        print("ee",edges.ahape)
        
        
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        
        pred,logprobs = self(X,edges)

        print("p.shape", pred.shape)
        print("logprobs.shape", logprobs.shape)

        
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
        for i in range(1,self.hparams.test_eval):
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
        for i in range(1,self.hparams.test_eval):
            pred_,logprobs = self(X,edges)
            pred+=pred_.softmax(-1)
        
        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool),:]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        
        self.log('val_loss', loss.detach())
        self.log('val_acc', 100*correct_t)

import sys
import torch
import pickle
import numpy as np
import os.path as osp
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


class UKBBAgeDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, fold=0, train=True, samples_per_epoch=100, device='cpu'):
        with open('data/UKBB.pickle', 'rb') as f:
            X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data

        self.X = torch.from_numpy(X_[:,:,fold]).float().to(device)
        self.y = torch.from_numpy(y_[:,:,fold]).float().to(device)
        self.weight = torch.from_numpy(np.squeeze(weight_[:1,fold])).float().to(device)
        if train:
            self.mask = torch.from_numpy(train_mask_[:,fold]).to(device)
        else:
            self.mask = torch.from_numpy(test_mask_[:,fold]).to(device)        
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask
        
class TadpoleDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, fold=0, train=True, samples_per_epoch=100, device='cpu',full=False):
        with open('data/tadpole_data.pickle', 'rb') as f:
            X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data
        
        if not full:
            X_ = X_[...,:30,:] # For DGM we use modality 1 (M1) for both node representation and graph learning.

        
        self.n_features = X_.shape[-2]
        self.num_classes = y_.shape[-2]
        
        self.X = torch.from_numpy(X_[:,:,fold]).float().to(device)
        self.y = torch.from_numpy(y_[:,:,fold]).float().to(device)
        self.weight = torch.from_numpy(np.squeeze(weight_[:1,fold])).float().to(device)
        if train:
            self.mask = torch.from_numpy(train_mask_[:,fold]).to(device)
        else:
            self.mask = torch.from_numpy(test_mask_[:,fold]).to(device)
            
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask, [[]]
 
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

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 

class PlanetoidDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', samples_per_epoch=100, name='Cora', device='cpu'):
        dataset = get_planetoid_dataset(name)
        self.X = dataset[0].x.float().to(device)
        self.y = one_hot_embedding(dataset[0].y,dataset.num_classes).float().to(device)
        self.edge_index = dataset[0].edge_index.to(device)
        self.n_features = dataset[0].num_node_features
        self.num_classes = dataset.num_classes
        
        if split=='train':
            self.mask = dataset[0].train_mask.to(device)
        if split=='val':
            self.mask = dataset[0].val_mask.to(device)
        if split=='test':
            self.mask = dataset[0].test_mask.to(device)
         
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask,self.edge_index

os.environ["CUDA_VISIBLE_DEVICES"]="0";

import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

def run_training_process(run_params):
    
    train_data = None
    test_data = None
    
    if run_params.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        train_data = PlanetoidDataset(split='train', name=run_params.dataset, device='cuda')
        val_data = PlanetoidDataset(split='val', name=run_params.dataset, samples_per_epoch=1)
        test_data = PlanetoidDataset(split='test', name=run_params.dataset, samples_per_epoch=1)
        
    if run_params.dataset == 'tadpole':
        train_data = TadpoleDataset(fold=run_params.fold,train=True, device='cuda')
        val_data = test_data = TadpoleDataset(fold=run_params.fold, train=False,samples_per_epoch=1)
                                   
    if train_data is None:
        raise Exception("Dataset %s not supported" % run_params.dataset)
        
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
    
    #configure input feature size
    if run_params.pre_fc is None or len(run_params.pre_fc)==0: 
        if len(run_params.dgm_layers[0])>0:
            run_params.dgm_layers[0][0]=train_data.n_features
        run_params.conv_layers[0][0]=train_data.n_features
    else:
        run_params.pre_fc[0]=train_data.n_features
    run_params.fc_layers[-1] = train_data.num_classes
    
    model = DGM_Model(run_params)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode='min')
    callbacks = [checkpoint_callback,early_stop_callback]
    
    if val_data==test_data:
        callbacks = None
        
    logger = TensorBoardLogger("logs/")
    trainer = pl.Trainer.from_argparse_args(run_params,logger=logger,
                                            callbacks=callbacks) 
    trainer.fit(model, datamodule=MyDataModule())
    trainer.test()
    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args(['--gpus','1',                         
                              '--log_every_n_steps','100',                          
                              '--max_epochs','100',
                              '--progress_bar_refresh_rate','10',                         
                              '--check_val_every_n_epoch','1'])
    parser.add_argument("--num_gpus", default=10, type=int)
    
    parser.add_argument("--dataset", default='Cora')
    parser.add_argument("--fold", default='0', type=int) #Used for k-fold cross validation in tadpole/ukbb
    
    parser.add_argument("--conv_layers", default=[[32,32],[32,16],[16,8]], type=lambda x :eval(x))
    parser.add_argument("--dgm_layers", default= [[32,16,4],[],[]], type=lambda x :eval(x))
    parser.add_argument("--fc_layers", default=[8,8,3], type=lambda x :eval(x))
    parser.add_argument("--pre_fc", default=[-1,32], type=lambda x :eval(x))

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
