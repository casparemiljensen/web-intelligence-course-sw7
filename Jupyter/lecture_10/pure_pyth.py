import os
import re
import string
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool

# Ensure the GPU is utilized
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set global variables for text processing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Parallelized text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

# Dataset class for PyTorch
class TextDataset(Dataset):
    def __init__(self, data, word2idx, window_size):
        self.data = data
        self.word2idx = word2idx
        self.window_size = window_size
        self.pairs = self.generate_pairs()

    def generate_pairs(self):
        pairs = []
        for tokens in self.data:
            indices = [self.word2idx[word] for word in tokens if word in self.word2idx]
            for i, center in enumerate(indices):
                for j in range(-self.window_size, self.window_size + 1):
                    if j != 0 and 0 <= i + j < len(indices):
                        pairs.append((center, indices[i + j]))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx], dtype=torch.long)

# Define the Skip-gram model
class SkipGramModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, ns=0):
        super(SkipGramModel, self).__init__()
        self.ns = ns
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)
        if ns > 0:
            self.context_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_word, context_word=None, negative_samples=None):
        center_embedding = self.embeddings(center_word)
        if self.ns > 0:
            pos_embedding = self.context_embeddings(context_word)
            pos_score = torch.sum(center_embedding * pos_embedding, dim=1)
            neg_embedding = self.context_embeddings(negative_samples)
            neg_score = torch.bmm(neg_embedding, center_embedding.unsqueeze(2)).squeeze(2)
            return pos_score, neg_score
        else:
            output = self.linear(center_embedding)
            return output

    def save_model(self, path):
        i = 1
        unique_path = path
        while os.path.exists(unique_path):
            unique_path = f"{path[:-4]}_{i}.pth"
            i += 1
        torch.save(self.state_dict(), unique_path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

# Load and preprocess the dataset
data = pd.read_csv("your_dataset.csv")
texts = data["text_column"]

# Preprocess using multiprocessing
with Pool() as pool:
    tokenized_texts = list(tqdm(pool.imap(preprocess_text, texts), total=len(texts)))

# Build vocabulary
vocab = Counter([word for tokens in tokenized_texts for word in tokens])
word2idx = {word: idx for idx, (word, _) in enumerate(vocab.items())}
id2word = {idx: word for word, idx in word2idx.items()}

# Create dataset and dataloader
window_size = 2
dataset = TextDataset(tokenized_texts, word2idx, window_size)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# Initialize model, loss, and optimizer
embedding_dim = 100
ns = 5
model = SkipGramModel(len(vocab), embedding_dim, ns=ns).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 3
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        batch = batch.to(device)
        center, context = batch[:, 0], batch[:, 1]

        if ns > 0:
            negative_samples = torch.randint(0, len(vocab), (len(batch), ns), device=device)
            pos_score, neg_score = model(center, context, negative_samples)
            pos_loss = -torch.log(torch.sigmoid(pos_score)).mean()
            neg_loss = -torch.log(torch.sigmoid(-neg_score)).mean()
            loss = pos_loss + neg_loss
        else:
            output = model(center)
            loss = loss_function(output, context)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# Save model
model.save_model("skipgram_model.pth")
