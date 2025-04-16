import sys
import nmslib
import numpy as np
from .model import CustomModel
import torch


def find_nearest_neighbors(query_word, k=100):
    path = "C:/Users/LEGION/Desktop/NLP Project/Spelling-Correction-and-Next-Word-Generation/backend/spelling_correction/saved_model_3"
    index = nmslib.init(method="hnsw", space="cosinesimil")
    index.loadIndex(f"{path}/word_index.bin", load_data=True)
    words = np.load(f"{path}/word_list.npy", allow_pickle=True)
    model_path = f"{path}/char2vec.pth"

    model = CustomModel()

    model.load_state_dict(torch.load(model_path, weights_only=True))
    query_vector = model.get_embedding(model.get_OHE(query_word).to(model.device))  
    ids, distances = index.knnQuery(query_vector.detach().cpu(), k=k)

    return [words[i] for i in ids]


