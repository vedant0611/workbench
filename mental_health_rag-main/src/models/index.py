# models.py
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, hidden_channels*3)
        self.conv4 = GCNConv(hidden_channels*3, hidden_channels*4)
        self.conv5 = GCNConv(hidden_channels*4, hidden_channels*4)
        self.conv6 = GCNConv(hidden_channels*4, hidden_channels*3)
        self.conv7 = GCNConv(hidden_channels*3, hidden_channels*2)
        self.conv8 = GCNConv(hidden_channels*2, hidden_channels)
        self.conv9 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv6(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv7(x, edge_index)
        x = F.relu(x)
        x = self.conv8(x, edge_index)
        x = F.relu(x)
        x = self.conv9(x, edge_index)
        x = global_mean_pool(x, data.batch)  # Pooling for graph-level representation
        return x
