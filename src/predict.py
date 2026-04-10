import os
import json
import torch
import torch.nn as nn

DATA_DIR = "data/processed"
MODEL_PATH = "model.pt"
OUTPUT_PATH = "predictions.json"

# ✅ 重新定义模型结构（避免 import train.py 导致训练被执行）
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

# ✅ 加载测试数据
test_images, test_labels = torch.load(os.path.join(DATA_DIR, "test.pt"))

# ✅ 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ✅ 进行预测
with torch.no_grad():
    outputs = model(test_images)
    predicted = torch.argmax(outputs, dim=1)

# ✅ 保存前10个预测结果（用于展示）
results = []
for i in range(10):
    results.append({
        "index": i,
        "predicted": int(predicted[i].item()),
        "actual": int(test_labels[i].item())
    })

# ✅ 保存到 JSON 文件
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"Predictions saved to {OUTPUT_PATH}")