import torch
import torch.nn as nn

def mlp(in_dim, out_dim, hidden_dim=None, num_layers=2, activation=nn.ReLU):
    """
    Args:
        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: width of hidden layers (defaults to out_dim)
        num_layers: number of hidden layers (not counting final layer)
        activation: activation class
    """
    hidden_dim = hidden_dim or out_dim
    layers = []

    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(activation())

    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation())

    layers.append(nn.Linear(hidden_dim, out_dim))

    return nn.Sequential(*layers)

class ResidualMessageMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, n_layers=1, activation=nn.LeakyReLU):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.activation = activation()

        self.proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.hidden_layers = nn.ModuleList(layers)

        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        identity = self.proj(x)

        out = x
        # pass through hidden layers
        for layer in self.hidden_layers:
            out = self.activation(layer(out))

        # residual
        out = out + identity
        out = self.activation(out)

        return self.fc_out(out)

class GatedMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, n_layers=2, activation=nn.ReLU):
        """
        Args:
            in_dim: input dimension
            out_dim: output dimension
            hidden_dim: width of hidden layers (defaults to out_dim)
            num_layers: number of hidden layers (not counting final layer)
            activation: activation class
        """
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.num_layers = n_layers
        self.activation = activation()

        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.gates.append(nn.Linear(in_dim, hidden_dim))

        # Remaining hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.gates.append(nn.Linear(hidden_dim, hidden_dim))

        # Final output layer
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        for layer, gate in zip(self.layers, self.gates):
            h = self.activation(layer(x))
            g = torch.sigmoid(gate(x))
            x = h * g  # element-wise gating
        return self.out_layer(x)


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout = 0.6, alpha = 0.2):
        super(GATLayer, self).__init__()

        self.W = nn.Linear(in_features, out_features, bias = False) 
        
        self.a = nn.Linear(out_features*2, 1, bias = False)
        
        self.leakyrelu = nn.LeakyReLU(alpha) 

        self.dropout = nn.Dropout(dropout) 

    def forward(self, node, edge_index): 

        transformed_node = self.W(node)
        N = node.size(0)

        source_idx, target_idx = edge_index
        node_i = transformed_node[source_idx]
        node_j = transformed_node[target_idx]
        combined_nodes = torch.cat([node_i, node_j], dim = 1) 
        raw_attention = self.leakyrelu(self.a(combined_nodes)).squeeze()

        attention = self.edge_softmax(raw_attention, source_idx, N) 
        attention = self.dropout(attention) 
        
        out = torch.zeros_like(transformed_node)
        out.index_add_(0, source_idx, attention.view(-1, 1) * node_j)

        return out

    def edge_softmax(self, logits, index, num_nodes): 

        logits_max = torch.zeros(num_nodes).to(logits.device)
        logits_max.index_reduce_(0, index, logits, reduce = 'amax', include_self = False)

        exp = (logits - logits_max[index]).exp()
        sum_exp = torch.zeros(num_nodes).to(logits.device)
        sum_exp.index_add_(0, index, exp) 
        
        return exp / (sum_exp[index] + 1e-10)


