import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import mlp
from .conv import GeneralConv
class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, n_layers, dropout = 0.2, ff_hidden = 32, n_resid_layers = 1, n_mlp_layers = 1):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.node_ff  = nn.ModuleList()
        self.edge_ff = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        self.Print = False
        self.node_linear = nn.Linear(in_dim,n_hid)
        self.edge_linear = nn.Linear(20, n_hid)
        self.concat_linear = nn.Linear(64, n_hid)
        self.node_ff = mlp(in_dim, n_hid, hidden_dim=ff_hidden, num_layers = n_mlp_layers)
        self.edge_ff = mlp(20, n_hid, hidden_dim=ff_hidden, num_layers = n_mlp_layers)

        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(n_hid, n_hid, n_resid_layers = n_resid_layers, n_mlp_layers = n_mlp_layers))
        self.gcs.append(GeneralConv(n_hid, n_hid, n_resid_layers = n_resid_layers, n_mlp_layers = n_mlp_layers))

    def forward(self, node_feature, edge_index, edge_feature):

        node_res = self.node_ff(node_feature)
        node_skip = self.node_linear(node_feature)
        tot_node = node_skip + node_res
        meta_node = self.drop(tot_node)

        edge_res = self.edge_ff(edge_feature)
        edge_skip = self.edge_linear(edge_feature)
        tot_edge = edge_skip + edge_res
        meta_edge = self.drop(tot_edge)

        for gc in self.gcs:
            meta_node = gc(meta_node, edge_index, meta_edge)
            meta_node = node_res + meta_node
        return meta_node

    def pool(self,atom_features,idx):
        assert sum([len(index) for index in idx]) == atom_features.shape[0]
        agg_feature = [torch.mean(atom_features[index], dim = 0, keepdim = True) for index in idx]
        return torch.cat(agg_feature,dim = 0)

