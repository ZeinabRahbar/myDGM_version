import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, EdgeConv

class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance="euclidean", sparse=True):
        super(DGM_d, self).__init__()
        self.sparse = sparse
        self.temperature = nn.Parameter(torch.tensor(1. if distance=="hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.k = k
        self.distance = distance

    def forward(self, x, A, not_used=None, fixedges=None):
        x = self.embed_f(x, A)
        if x.dim() == 2:
            x = x[None, ...]

        if self.training:
            if fixedges is not None:
                return x, fixedges, torch.zeros(fixedges.shape[0], fixedges.shape[-1]//self.k, self.k, dtype=torch.float, device=x.device)
            edges_hat, logprobs = self.sample_without_replacement(x)
        else:
            with torch.no_grad():
                if fixedges is not None:
                    return x, fixedges, torch.zeros(fixedges.shape[0], fixedges.shape[-1]//self.k, self.k, dtype=torch.float, device=x.device)
                edges_hat, logprobs = self.sample_without_replacement(x)

        return x, edges_hat, logprobs

    def sample_without_replacement(self, x):
        b, n, _ = x.shape
        if self.distance == "euclidean":
            G_i = torch.unsqueeze(x, 2)  # (M**2, 1, 2)
            X_j = torch.unsqueeze(x, 1)  # (1, N, 2)

            mD = ((G_i - X_j) ** 2).sum(-1)

            lq = mD * torch.exp(torch.clamp(self.temperature, -5, 5))
            indices = torch.topk(lq, k=self.k, dim=1, largest=False).indices

            x1 = torch.gather(x, -2, indices.view(indices.shape[0], -1)[:, None].repeat(1, 1, x.shape[-1]))
            x2 = x[:, :, None, :].repeat(1, 1, self.k, 1).view(x.shape[0], -1, x.shape[-1])
            logprobs = (-(x1 - x2).pow(2).sum(-1) * torch.exp(torch.clamp(self.temperature, -5, 5))).reshape(x.shape[0], -1, self.k)

        edges = torch.stack((indices.view(b, -1), torch.arange(n).view(1, n, 1).repeat(b, 1, self.k).view(b, -1)), -1)

        if self.sparse:
            return (edges + (torch.arange(b).to(x.device) * n)[:, None, None]).transpose(0, 1).reshape(2, -1), logprobs
        return edges, logprobs

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

class DGM_Model(nn.Module):
    def __init__(self, hparams):
        super(DGM_Model, self).__init__()
        self.hparams = hparams
        self.graph_f = nn.ModuleList()
        self.node_g = nn.ModuleList()
        
        for i, (dgm_l, conv_l) in enumerate(zip(hparams["dgm_layers"], hparams["conv_layers"])):
            if len(dgm_l) > 0:
                if hparams["ffun"] == 'gcn':
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0], dgm_l[-1]), k=hparams["k"], distance=hparams["distance"]))
                elif hparams["ffun"] == 'gat':
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0], dgm_l[-1]), k=hparams["k"], distance=hparams["distance"]))
                elif hparams["ffun"] == 'mlp':
                    self.graph_f.append(DGM_d(MLP(dgm_l), k=hparams["k"], distance=hparams["distance"]))
                elif hparams["ffun"] == 'knn':
                    self.graph_f.append(DGM_d(nn.Identity(), k=hparams["k"], distance=hparams["distance"]))
            else:
                self.graph_f.append(nn.Identity())
            
            if hparams["gfun"] == 'edgeconv':
                conv_l = conv_l.copy()
                conv_l[0] = conv_l[0] * 2
                self.node_g.append(EdgeConv(MLP(conv_l)))
            elif hparams["gfun"] == 'gcn':
                self.node_g.append(GCNConv(conv_l[0], conv_l[1]))
            elif hparams["gfun"] == 'gat':
                self.node_g.append(GATConv(conv_l[0], conv_l[1]))
        
        self.fc = MLP(hparams["fc_layers"], final_activation=False)
        if hparams["pre_fc"] is not None and len(hparams["pre_fc"]) > 0:
            self.pre_fc = MLP(hparams["pre_fc"], final_activation=True)

    def forward(self, x, edges=None):
        if self.hparams["pre_fc"] is not None and len(self.hparams["pre_fc"]) > 0:
            x = self.pre_fc(x)
        
        graph_x = x.detach()
        lprobslist = []
        for f, g in zip(self.graph_f, self.node_g):
            graph_x, edges, lprobs = f(graph_x, edges)
            x = torch.relu(g(x, edges)).view(x.shape[0], x.shape[1], -1)
            graph_x = torch.cat([graph_x, x.detach()], -1)
            if lprobs is not None:
                lprobslist.append(lprobs)
        
        return self.fc(x), torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None


if __name__ == "__main__":
    hparams = {
        "dgm_layers": [[32, 16, 4], [], []],
        "conv_layers":  [[32, 32], [32, 16], [16, 8]],
        "fc_layers":  [8, 8, 3],
        "pre_fc": [-1, 32],
        "ffun": "gcn",
        "gfun": "gcn",
        "k": 5,
        "distance": "euclidean",
        "dropout": 0.0,
    }
    
    model = DGM_Model(hparams)
    # Assuming x is your input data and edges is your edge index
    x = torch.randn(1, 100, 64)  # Example input
    edges = torch.randint(0, 100, (2, 200))  # Example edge index
    
    output, logprobs = model(x, edges)
    print(output.shape, logprobs.shape)


# class GraphLearn(nn.Module):
#     def __init__(self, embed_f, k=5):
#         super(GraphLearn, self).__init__()
#         self.graph_f = nn.ModuleList()
#         self.node_g = nn.ModuleList()

#         self.conv_layers=[[16, 32], [32, 64]]
#         self.fc_layers=[16, 10]
#         self.dgm_layers=[[16, 16], []]

#         self.k=k
#         self.distance='euclidean'
#         self.ffun='gcn'
#         self.gfun='mlp'
#         self.pre_fc=[64,32]
#         self.dropout=0.5

#         for i, (dgm_l, conv_l) in enumerate(zip(self.dgm_layers, self.conv_layers)):
#             if len(dgm_l) > 0:
#                 if self.ffun == 'gcn':
#                     self.graph_f.append(DGM_d(embed_f, k=self.k, distance=self.distance))
#                 elif self.ffun == 'gat':
#                     self.graph_f.append(DGM_d(embed_f, k=self.k, distance=self.distance))
#                 elif self.ffun == 'mlp':
#                     self.graph_f.append(DGM_d(embed_f, k=self.k, distance=self.distance))
#                 elif self.ffun == 'knn':
#                     self.graph_f.append(DGM_d(embed_f, k=self.k, distance=self.distance))
#             else:
#                 self.graph_f.append(nn.Identity())
            
#             if self.gfun == 'edgeconv':
#                 conv_l = conv_l.copy()
#                 conv_l[0] = conv_l[0] * 2
#                 self.node_g.append(EdgeConv(MLP(conv_l)))
#             elif self.gfun == 'gcn':
#                 self.node_g.append(GCNConv(conv_l[0], conv_l[1]))
#             elif self.gfun == 'gat':
#                 self.node_g.append(GATConv(conv_l[0], conv_l[1]))
        
#         self.fc = MLP(self.fc_layers, final_activation=False)
        
#         if self.pre_fc is not None and len(self.pre_fc) > 0:
#             self.pre_fc = MLP(self.pre_fc, final_activation=True)

#     def forward(self, x, edges=None):
#         if self.self.pre_fc is not None and len(self.self.pre_fc) > 0:
#             x = self.pre_fc(x)
        
#         graph_x = x.detach()
#         lprobslist = []
#         for f, g in zip(self.graph_f, self.node_g):
#             graph_x, edges, lprobs = f(graph_x, edges)
#             x = torch.relu(g(x, edges)).view(x.shape[0], x.shape[1], -1)
#             graph_x = torch.cat([graph_x, x.detach()], -1)
#             if lprobs is not None:
#                 lprobslist.append(lprobs)
        
#         print("aaaaa", self.fc(x))
#         print(self.fc(x).shape)
#         print(graph_x)
#         print(graph_x.shape)

#         print(torch.stack(lprobslist, -1))
#         print(torch.stack(lprobslist, -1).shape)

#         return self.fc(x), torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None

