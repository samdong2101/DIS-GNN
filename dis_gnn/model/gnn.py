import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import mlp
from .conv import GeneralConv
class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, n_layers, batch_size, dropout = 0.2, ff_hidden = 32, n_resid_layers = 1, n_mlp_layers = 1):
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
        self.state_linear = nn.Linear(batch_size, batch_size)
        self.node_ff = mlp(in_dim, n_hid, hidden_dim=ff_hidden, num_layers = n_mlp_layers)
        self.edge_ff = mlp(20, n_hid, hidden_dim=ff_hidden, num_layers = n_mlp_layers)
        self.state_ff = mlp(batch_size, batch_size, hidden_dim = ff_hidden, num_layers = n_mlp_layers) 
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(n_hid, n_hid, n_resid_layers = n_resid_layers, n_mlp_layers = n_mlp_layers))
        self.gcs.append(GeneralConv(n_hid, n_hid, n_resid_layers = n_resid_layers, n_mlp_layers = n_mlp_layers))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, node_feature, edge_index, edge_feature, global_state, group_size): 
        
        node_res = self.node_ff(node_feature)
        node_skip = self.node_linear(node_feature)
        tot_node = node_skip + node_res
        meta_node = self.drop(tot_node)
        
        edge_res = self.edge_ff(edge_feature) 
        edge_skip = self.edge_linear(edge_feature)
        tot_edge = edge_skip + edge_res
        meta_edge = self.drop(tot_edge)
        
        global_state_res = self.state_ff(global_state)
        global_state_skip = self.state_linear(global_state)
        tot_state = global_state_res + global_state_skip
        tot_state = torch.repeat_interleave(tot_state, group_size).to(self.device)
        meta_state = self.drop(tot_state) 
        for gc in self.gcs:
            meta_node = gc(meta_node, edge_index, meta_edge, meta_state)
            meta_node = node_res + meta_node
        return meta_node

    def pool(self,atom_features,idx):
        assert sum([len(index) for index in idx]) == atom_features.shape[0]
        agg_feature = [torch.mean(atom_features[index], dim = 0, keepdim = True) for index in idx]
        return torch.cat(agg_feature,dim = 0)
