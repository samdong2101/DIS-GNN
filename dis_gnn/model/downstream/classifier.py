import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        """
        Simple binary classifier.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in the hidden layer.
        """
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim,hidden_dim*2)
        self.fc_hidden2 = nn.Linear(hidden_dim*2, hidden_dim*4)
        self.fc_hidden3 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc_hidden4 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # single output for binary classification
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.fc_hidden2(x)
        x = self.relu(x)
        x = self.fc_hidden3(x)
        x = self.relu(x)
        x = self.fc_hidden4(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.fc1.in_features}, hidden_dim={self.fc1.out_features})"
