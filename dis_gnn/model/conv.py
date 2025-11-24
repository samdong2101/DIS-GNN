import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math
import time
from torch_scatter import scatter_mean

class Convolution(MessagePassing):
    def __init__(self, in_dim, out_dim, dropout = 0.2, n_resid_layers = 1, **kwargs):
        super(HGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.total_rel     = num_types * num_relations * num_types
        self.resid_hidden  = 128
        self.n_resid_layers = n_resid_layers
        self.n_mlp_layeres = n_mlp_layers
        self.mlp_hidden = 64
        self.bond_linears = ResidualMessageMLP(n_hid*3, out_dim, hidden_dim = self.resid_hidden,n_layers = self.n_resid_layers))
        self.node_linears = ResidualMessageMLP(n_hid*2, out_dim,hidden_dim = self.resid_hidden, n_layers = self.n_resid_layers))
        self.drop = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_feature):
        concat_edge, concat_node = self.message(edge_index[0],edge_index[1], node_inp, node_inp, node_type, node_type, edge_type, edge_feature)
        return concat_node

    def average_edge_vector(self, edge_index, edge_attr):
        src = edge_index[0]  
        dest = edge_index[1] 
        num_nodes = max([int(src.max()) + 1, int(dest.max()) + 1])       
        out = scatter_mean(edge_attr, src, dim=0, dim_size=num_nodes)
        out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
        return out
    
    def message(self, edge_index_i, edge_index_j, node_inp_i, node_inp_j, node_type_i, node_type_j,edge_type, edge_feature):
        '''
            j: source, i: target; <j, i>
        '''
        edge_index = [edge_index_i, edge_index_j]
        node_inp_i_rollout = node_inp_i[edge_index_i]
        node_inp_j_rollout = node_inp_j[edge_index_j]
        concat_edge = torch.concat([node_inp_i_rollout, node_inp_j_rollout, edge_feature],dim = 1)
        concat_edge = self.bond_linears.forward(concat_edge)
        
        agg_edges = self.average_edge_vector(edge_index, concat_edge)
        concat_node = torch.concat([node_inp_i, agg_edges],dim = 1)
        concat_node = self.node_linears.forward(concat_node)
        
        return concat_edge, concat_node 
                
    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, n_resid_layers = 1):
        super(GeneralConv, self).__init__()
        self.base_conv = Convolution(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, use_RTE, n_resid_layers)
    def forward(self, meta_xs, node_type, edge_index,edge_type, edge_feature):
        return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_feature)
