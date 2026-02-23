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
from .layers import ResidualMessageMLP
from .layers import GatedMLP
from .layers import GATLayer
class Convolution(MessagePassing):
    def __init__(self, in_dim, out_dim, n_resid_layers = 1, n_mlp_layers = 1, graphtype ='crystal', **kwargs):
        super(Convolution, self).__init__(node_dim=0, aggr='add', **kwargs)
        self.graphtype = graphtype
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.resid_hidden  = 128
        self.n_resid_layers = n_resid_layers
        self.n_mlp_layers = n_mlp_layers
        self.mlp_hidden = 64
        self.bond_linears = GatedMLP(in_dim*3, out_dim, hidden_dim = self.resid_hidden,n_layers = self.n_resid_layers) # used to be ResidualMessageMLP
        self.node_linears = GatedMLP(in_dim*3+1, out_dim,hidden_dim = self.resid_hidden, n_layers = self.n_resid_layers) # used to be ResidualMessageMLP
        self.line_bond_linears =  GatedMLP(in_dim*2, out_dim, hidden_dim = self.resid_hidden,n_layers = self.n_resid_layers)
        self.line_node_linears =  GatedMLP(in_dim*2, out_dim, hidden_dim = self.resid_hidden,n_layers = self.n_resid_layers)
        self.norm = nn.LayerNorm(out_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crystal_gat_layer = GATLayer(in_dim*3+1, out_dim) 
        self.line_gat_layer = GATLayer(in_dim*2, out_dim) 
    def forward(self, node_inp, edge_index, edge_feature, global_state = None, cells = None, coords = None, graphtype = 'crystal'):
        
        concat_edge, concat_node = self.message(edge_index[0],edge_index[1], node_inp, node_inp, edge_feature, global_state, cells, coords, graphtype)
        
        return concat_node

    def average_edge_vector(self, edge_index, edge_attr):
        src = edge_index[0]  
        dest = edge_index[1] 
        num_nodes = max([int(src.max()) + 1, int(dest.max()) + 1])       
        out = scatter_mean(edge_attr, src, dim=0, dim_size=num_nodes)
        out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
        return out
    
    def message(self, edge_index_i, edge_index_j, node_inp_i, node_inp_j, edge_feature, global_state=None, cells=None, coords=None, graphtype = 'crystal'):
        '''
            j: source, i: target; <j, i>
        '''
        
        edge_index = [edge_index_i, edge_index_j]
        node_inp_i_rollout = node_inp_i[edge_index_i]
        node_inp_j_rollout = node_inp_j[edge_index_j] 
        
        try:
            concat_edge = torch.concat([node_inp_i_rollout, node_inp_j_rollout, edge_feature],dim = 1)
        except Exception as e:
            print(e) 

        concat_edge = self.bond_linears.forward(concat_edge)
        agg_edges = self.average_edge_vector(edge_index, concat_edge)
        
        if graphtype == 'crystal':
            
            cells = cells.reshape(cells.shape[0],1)
            global_state = global_state.reshape(global_state.shape[0],1)
            try:
                concat_node = torch.concat([node_inp_i, agg_edges, coords, global_state],dim = 1)
            except Exception as e:
                print(e)
          
            
            concat_node = self.node_linears.forward(concat_node)
            #concat_node = self.crystal_gat_layer.forward(concat_node, edge_index) 
            concat_node = self.norm(concat_node)
        else:
            try:
                concat_node = torch.concat([node_inp_i, agg_edges],dim=1) 
            except Exception as e:
                print(e)
            concat_node = self.line_node_linears.forward(concat_node)
            #concat_node = self.line_gat_layer.forward(concat_node, edge_index) 
            concat_node = self.norm(concat_node) 
       
        return concat_edge, concat_node 
                
   
class GeneralConv(nn.Module):
    def __init__(self, in_hid, out_hid, n_resid_layers = 1, n_mlp_layers = 1, graphtype='crystal'):
        super(GeneralConv, self).__init__()
        self.base_conv = Convolution(in_hid, out_hid, n_resid_layers, n_mlp_layers, graphtype)
    def forward(self, meta_xs, edge_index, edge_feature, global_state, cells, coords, graphtype):
        return self.base_conv(meta_xs, edge_index, edge_feature, global_state, cells, coords, graphtype)
