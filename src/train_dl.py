import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import re
import os

# 1. Load Data
print("Loading data...")
df = pd.read_csv('../data/data_preprocessed.csv')

# Drop NA
df = df.dropna(subset=['text_processed', 'sentiment', 'label'])

# Asumsikan label sudah ada dari df['label'] yang bernilai 0, 1, atau 2
# Jika belum, kita map: Negatif -> 0, Netral -> 1, Positif -> 2
label_map = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
if 'sentiment' in df.columns and ('label' not in df.columns or df['label'].dtype == 'object'):
    df['dl_label'] = df['sentiment'].map(label_map)
else:
    df['dl_label'] = df['label']

texts = df['text_processed'].astype(str).values
labels = df['dl_label'].astype(int).values

# 2. Build Vocabulary
print("Building vocabulary...")
vocab = {'<PAD>': 0, '<UNK>': 1}
max_len = 50

# Sederhana tokenizer
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

for text in texts:
    tokens = tokenize(text)
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

print(f"Vocab size: {len(vocab)}")

os.makedirs('../model_dl', exist_ok=True)
with open('../model_dl/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

# 3. Create Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])
        indices = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        # Pad or truncate
        if len(indices) < self.max_len:
            indices = indices + [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

dataset = TextDataset(texts, labels, vocab, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Model Definition
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        # embedded: [batch_size, seq_len, embed_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # hidden[-1]: [batch_size, hidden_dim]
        output = self.fc(hidden[-1])
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = LSTMClassifier(len(vocab), embed_dim=64, hidden_dim=128, output_dim=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training
print("Training model...")
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(predictions.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {100 * correct / total:.2f}%")

# 6. Save Model
model_path = '../model_dl/lstm_model.pt'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
