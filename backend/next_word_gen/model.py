import torch
import nmslib
import torch.nn as nn
import numpy as np
import json
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, vocab_size=9399, embedding_dim=100, hidden_dim=150):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output

def load_model(path,vocab_size):
    model_path = f"{path}/lstm_next_word_model.pth"
    next_word_model = LSTMModel(vocab_size)
    next_word_model.load_state_dict(torch.load(model_path, weights_only=True))
    return next_word_model

def get_word_embds(words,word_embds,wtoi):
    embd = []
    for word in words:
        wrd_idx = wtoi.get(word,0)
        embd.append(word_embds[wrd_idx])
    return embd

def predict_top_k_words_from_embds(words, model, word_embds, wtoi, device, k=5):
    embd_seq = get_word_embds(words, word_embds, wtoi)
    embd_seq = np.array(embd_seq)

    input_tensor = torch.tensor(embd_seq, dtype=torch.float32).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        lstm_out, (hidden, cell) = model.lstm(input_tensor)
        output = model.fc(hidden[-1])  # (1, vocab_size)
        probs = F.softmax(output, dim=1)  # Convert logits to probabilities

        _ , topk_indices = torch.topk(probs, k)

    idx_to_word = {v: k for k, v in wtoi.items()}
    topk_words = [idx_to_word.get(idx.item(), "<unk>") for idx in topk_indices[0]]

    return topk_words


def predict_next_word(words,k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "C:/Users/LEGION/Desktop/NLP Project/Spelling-Correction-and-Next-Word-Generation/backend/next_word_gen/saved_model"
    embd_path = f"{path}/word_embeddings.npy"
    vocabs_path = f"{path}/vocab.json"
    wtoi = None
    with open(vocabs_path, 'r') as f:
        wtoi = json.load(f)
    word_embds = np.load(embd_path,allow_pickle=True)

    model = load_model(path,vocab_size=len(wtoi))
    ans = predict_top_k_words_from_embds(words=words, model=model, word_embds=word_embds, wtoi=wtoi,device=device,k=k)
    return ans