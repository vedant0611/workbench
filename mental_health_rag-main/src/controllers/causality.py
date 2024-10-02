import numpy as np
from src.models.preds import causality_preds
from sentence_transformers import SentenceTransformer

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')
def causality(data):
    data_lst = data['list']
    data_enc = model.encode(data_lst)
    data_1 = [np.mean(i) for i in data_enc]
    result,graph = causality_preds(data_1)

    return result,graph
