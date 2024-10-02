import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from src.models.index import GNNModel
import os

cwd = os.getcwd()

model = GNNModel(1, 32, 3)
model.load_state_dict(torch.load(os.path.join(cwd, 'assets/model_path', 'causality.pt'), map_location=torch.device('cpu')))
model.eval()

# model = torch.load(cwd+'/assets/model_path/causality.pt')
node_map = torch.load(cwd+"/assets/model_path/node_map.pt")
filtered_edges = np.load(cwd+'/assets/model_path/filtered_edges.npy')

edge_index = torch.tensor([[node_map[edge[0]], node_map[edge[1]]] for edge in filtered_edges], dtype=torch.long).t().contiguous()

labels = {
    0:"a",
    1:"b",
    2:"c"
}

def create_user_graph(user_responses, node_map, edge_index):
    assert len(user_responses) == len(node_map), "Number of responses must match number of nodes"
    
    # Convert responses to tensor and normalize using training data statistics
    x = torch.tensor(user_responses, dtype=torch.float).view(-1, 1)
    mean = x.mean()
    std = x.std()
    x = (x - mean) / (std + 1e-6)  # Normalize
    
    # Create Data object
    user_graph = Data(x=x, edge_index=edge_index)
    return user_graph

def causality_preds(user_resp):

    user_graph = create_user_graph(user_resp, node_map, edge_index)


    with torch.no_grad():
    # Add batch dimension
    # user_graph = user_graph.unsqueeze(0)  # Shape: [1, 9, 1] for node features
    # user_graph.batch = torch.zeros(user_graph.num_nodes, dtype=torch.long)  # Batch tensor
    
        output = model(user_graph)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    predicted_label = labels[predicted_class]

    return predicted_label, user_graph
