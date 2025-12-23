import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import mlp
from .layers import GatedMLP
from .conv import GeneralConv

class GNN(nn.Module):
    def __init__(self, in_dim, line_in_dim, n_hid, n_layers, batch_size, dropout = 0.2, ff_hidden = 32, n_resid_layers = 1, n_mlp_layers = 1):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.line_gcs = nn.ModuleList()
        self.final_gcs = nn.ModuleList()
        self.in_dim    = in_dim
        self.line_in_dim = line_in_dim
        self.n_hid     = n_hid
        self.node_ff  = nn.ModuleList()
        self.edge_ff = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        
        self.node_linear = nn.Linear(in_dim,n_hid)
        self.line_node_linear = nn.Linear(line_in_dim,n_hid)
        self.edge_linear = nn.Linear(80, n_hid)
        self.line_edge_linear = nn.Linear(40, n_hid)
        self.state_linear = nn.Linear(batch_size, batch_size)
        self.cell_linear = nn.Linear(batch_size*9, batch_size)
        self.coord_linear = nn.Linear(3, n_hid)

        self.norm = nn.LayerNorm(n_hid)
        self.node_ff = mlp(in_dim, n_hid, hidden_dim=ff_hidden, num_layers = n_mlp_layers)
        self.line_node_ff = mlp(line_in_dim, n_hid, hidden_dim=ff_hidden, num_layers=n_mlp_layers)
        self.edge_ff = mlp(80, n_hid, hidden_dim=ff_hidden, num_layers = n_mlp_layers)
        self.line_edge_ff = mlp(40, n_hid, hidden_dim=ff_hidden, num_layers = n_mlp_layers) 
        self.state_ff = mlp(batch_size, batch_size, hidden_dim = ff_hidden, num_layers = n_mlp_layers) 
        self.cell_ff = mlp(batch_size*9, batch_size, hidden_dim = ff_hidden, num_layers = n_mlp_layers)
        self.coord_ff = mlp(3, n_hid, hidden_dim = ff_hidden, num_layers = n_mlp_layers)

        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(n_hid, n_hid, n_resid_layers = n_resid_layers, n_mlp_layers = n_mlp_layers))
            self.line_gcs.append(GeneralConv(n_hid, n_hid, n_resid_layers = n_resid_layers, n_mlp_layers = n_mlp_layers))
            self.final_gcs.append(GeneralConv(n_hid, n_hid, n_resid_layers = n_resid_layers, n_mlp_layers = n_mlp_layers))
        self.gcs.append(GeneralConv(n_hid, n_hid, n_resid_layers = n_resid_layers, n_mlp_layers = n_mlp_layers))
        self.line_gcs.append(GeneralConv(n_hid, n_hid, n_resid_layers=n_resid_layers, n_mlp_layers=n_mlp_layers))
        self.final_gcs.append(GeneralConv(n_hid,n_hid,n_resid_layers=n_resid_layers,n_mlp_layers=n_mlp_layers))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, node_feature, edge_index,  edge_feature, line_node_feature = None, line_edge_index = None, line_edge_feature = None, global_state = None, group_size = None, cell = None, coords = None): 
        
        node_res = self.node_ff(node_feature)
        meta_node = self.residual_connection(node_feature, self.node_linear, self.node_ff) 
        meta_edge = self.residual_connection(edge_feature, self.edge_linear, self.edge_ff)
         
        if None not in (line_node_feature, line_edge_index, line_node_feature):
            line_node_res = self.line_node_ff(line_node_feature)
            line_meta_node = self.residual_connection(line_node_feature, self.line_node_linear, self.line_node_ff)
            line_meta_edge = self.residual_connection(line_edge_feature, self.line_edge_linear, self.line_edge_ff)
            


        if global_state is not None:
            meta_state = self.residual_connection(global_state, self.state_linear, self.state_ff)
            meta_state = torch.repeat_interleave(meta_state, group_size).to(self.device) 
        
        if cell is not None:
            meta_cell = self.residual_connection(cell, self.cell_linear, self.cell_ff) 

        if coords is not None:
            meta_coords = self.residual_connection(coords, self.coord_linear, self.coord_ff) 
        
        for i in range(len(self.gcs)):
            gc = self.gcs[i]
            lgc = self.line_gcs[i]
            fgc = self.final_gcs[i]
            line_meta_node = lgc(line_meta_node, line_edge_index, line_meta_edge, None, None, None, 'line')
            meta_node = gc(meta_node, edge_index, meta_edge, meta_state, meta_cell, meta_coords, 'crystal')
            final_node = fgc(meta_node, edge_index, line_meta_node, meta_state, meta_cell, meta_coords,'crystal')
            final_node = node_res + final_node #meta_node
        return final_node

    def pool(self,atom_features,idx):
        assert sum([len(index) for index in idx]) == atom_features.shape[0]
        agg_feature = [torch.mean(atom_features[index], dim = 0, keepdim = True) for index in idx]
        return torch.cat(agg_feature,dim = 0)

    def residual_connection(self, feature, lin_layer, ff):
        
        feature_res = lin_layer(feature) 
        feature_skip = ff(feature)
        tot_feature = feature_res + feature_skip 
        dropped_feature = self.drop(tot_feature)
        
        return dropped_feature 

