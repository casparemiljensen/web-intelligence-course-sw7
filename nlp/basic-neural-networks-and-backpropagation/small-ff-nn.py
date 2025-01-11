import torch
import torch.nn as nn
import torch.optim as optim

# Define the network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 1)  # Weights: w13, w23
        self.output = nn.Linear(1, 1)  # Weight: who
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.hidden(x))
        y_hat = self.sigmoid(self.output(h))
        return y_hat

# Create the model
model = SimpleNN()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Example data
x_train = torch.tensor([[1.0, 2.0]])  # Input
y_train = torch.tensor([[1.0]])       # Target

# Training step
optimizer.zero_grad()
y_pred = model(x_train)
loss = criterion(y_pred, y_train)
loss.backward()
optimizer.step()

print(f"Updated weights: {list(model.parameters())}")
