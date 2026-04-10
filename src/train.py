import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

DATA_DIR = "data/processed"
MODEL_PATH = "model.pt"
METRICS_PATH = "metrics.json"

with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

EPOCHS = params["epochs"]
LR = params["lr"]
BATCH_SIZE = params["batch_size"]
ACTIVATION_NAME = params.get("activation", "relu").lower()
INIT_NAME = params.get("init", "default").lower()
OPTIMIZER_NAME = params.get("optimizer", "adam").lower()
MOMENTUM = params.get("momentum", 0.9)

train_images, train_labels = torch.load(os.path.join(DATA_DIR, "train.pt"))
test_images, test_labels = torch.load(os.path.join(DATA_DIR, "test.pt"))


def get_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class SimpleCNN(nn.Module):
    def __init__(self, activation_name: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.activation = get_activation(activation_name)
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def initialize_weights(model: nn.Module, init_name: str) -> None:
    if init_name == "default":
        return

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if init_name == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif init_name == "he":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unsupported initialization: {init_name}")

            if module.bias is not None:
                nn.init.zeros_(module.bias)


def get_optimizer(name: str, model: nn.Module, lr: float, momentum: float):
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    if name == "sgd_momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def main() -> None:
    model = SimpleCNN(activation_name=ACTIVATION_NAME)
    initialize_weights(model, INIT_NAME)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        OPTIMIZER_NAME,
        model,
        LR,
        MOMENTUM
    )

    model.train()
    instrumentation_logged = False

    for epoch in range(EPOCHS):
        for i in range(0, len(train_images), BATCH_SIZE):
            x_batch = train_images[i:i + BATCH_SIZE]
            y_batch = train_labels[i:i + BATCH_SIZE]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # Inspect only one mini-batch
            if not instrumentation_logged:
                print("\n=== Forward Pass Inspection ===")
                print("Sample outputs (first 5 rows):")
                print(outputs[:5])
                print(f"Batch loss before backward: {loss.item():.4f}")

            # Backward pass
            loss.backward()

            if not instrumentation_logged:
                print("\n=== Backward Pass Inspection ===")
                print(f"Conv weight gradient norm: {model.conv.weight.grad.norm().item():.6f}")
                print(f"FC weight gradient norm: {model.fc.weight.grad.norm().item():.6f}")

                conv_weight_norm_before = model.conv.weight.data.norm().item()
                fc_weight_norm_before = model.fc.weight.data.norm().item()

            # Optimizer step
            optimizer.step()

            if not instrumentation_logged:
                conv_weight_norm_after = model.conv.weight.data.norm().item()
                fc_weight_norm_after = model.fc.weight.data.norm().item()

                print("\n=== Parameter Update Inspection ===")
                print(f"Conv weight norm before update: {conv_weight_norm_before:.6f}")
                print(f"Conv weight norm after update:  {conv_weight_norm_after:.6f}")
                print(f"FC weight norm before update:   {fc_weight_norm_before:.6f}")
                print(f"FC weight norm after update:    {fc_weight_norm_after:.6f}")

                instrumentation_logged = True

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

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

    torch.save(model.state_dict(), MODEL_PATH)

    metrics = {
        "accuracy": round(accuracy, 4),
        "activation": ACTIVATION_NAME,
        "init": INIT_NAME,
        "optimizer": OPTIMIZER_NAME,
        "momentum": MOMENTUM,
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()