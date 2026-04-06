import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

# Paths
DATA_DIR = "data/processed"
MODEL_PATH = "model.pt"
METRICS_PATH = "metrics.json"

# Load data
train_images, train_labels = torch.load(os.path.join(DATA_DIR, "train.pt"))
test_images, test_labels = torch.load(os.path.join(DATA_DIR, "test.pt"))

# Normalize (optional but good practice)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN()

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 2
BATCH_SIZE = 64

# Training loop
model.train()
for epoch in range(EPOCHS):
    for i in range(0, len(train_images), BATCH_SIZE):
        x_batch = train_images[i:i+BATCH_SIZE]
        y_batch = train_labels[i:i+BATCH_SIZE]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    outputs = model(test_images)
    _, predicted = torch.max(outputs, 1)
    total += test_labels.size(0)
    correct += (predicted == test_labels).sum().item()

accuracy = correct / total

print(f"Test Accuracy: {accuracy:.4f}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)

# Save metrics
metrics = {"accuracy": accuracy}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Model saved to {MODEL_PATH}")
print(f"Metrics saved to {METRICS_PATH}")
