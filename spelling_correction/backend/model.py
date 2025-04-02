import torch.nn as nn
import torch

class CustomModel(nn.Module):
    def __init__(self, vocab_size=27, emb_dim=100, num_epochs=15, lr=0.001):
        super().__init__()

        self.vocab = "abcdefghijklmnopqrstuvwxyz"
        self.vocab_size = len(self.vocab) + 1
        self.ctoi = {char: idx for idx, char in enumerate(self.vocab)}

        self.num_epochs = num_epochs
        self.vocab_size = vocab_size
        self.lstm1 = nn.LSTM(input_size=self.vocab_size, hidden_size=emb_dim, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.fc = nn.Linear(1, 1)

        self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  

        
        self.create_OHE()

    def create_OHE(self):
        """Creates a One-Hot Encoding matrix for the vocabulary."""
        self.OHE = torch.zeros((self.vocab_size, self.vocab_size))
        for i in range(self.vocab_size):
            self.OHE[i, i] = 1

    def get_OHE(self, word):
        """Converts a word into a one-hot encoding tensor."""
        emb = [self.OHE[self.ctoi.get(char, self.vocab_size - 1)] for char in word]
        return torch.stack(emb)
    
    def fit(self, batched_data):
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for x1, x2, target_batch in batched_data:
                x1, x2, target_batch = x1.to(self.device), x2.to(self.device), target_batch.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.forward(x1, x2)
                loss = self.loss_fn(outputs, target_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"{epoch+1}/{self.num_epochs} - Loss: {epoch_loss / len(batched_data)}")

    def get_embedding(self, x):
        out1, _ = self.lstm1(x)
        out2, (hn, _) = self.lstm2(out1)
        return hn.squeeze(0) 

    def forward(self, x1, x2):
        emb1 = self.get_embedding(x1)
        emb2 = self.get_embedding(x2)

        diff = emb1 - emb2
        squared_norm = torch.sum(diff ** 2, dim=1, keepdim=True)

        out = torch.sigmoid(self.fc(squared_norm))
        return out
    
    def save_model(self, model_name):
        torch.save(self.state_dict(), model_name)
        print(f"Model saved to {model_name}")

    def load_model(self, model_name):
        self.load_state_dict(torch.load(model_name))
        self.eval()  # Set the model to evaluation mode after loading
        print(f"Model loaded from {model_name}")
