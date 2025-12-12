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



