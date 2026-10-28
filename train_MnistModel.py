import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import os
import openpyxl
from datetime import datetime

from models import MnistModel
import config


device = config.device

lr = config.lr
label_smoothing = config.label_smoothing
weight_decay = config.weight_decay
epochs = config.epochs
hyperparams = {
    "训练的开始时间": None,
    "识别的准确度": None,
    "学习率": lr,
    "标签的平滑度": 0.1,
    "权重的衰减": weight_decay,
    "训练的轮数": epochs, 
}


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
data_dir = config.mnist_dataset_path
train_dataset = torchvision.datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform,
)
train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, 
                          shuffle=config.train_shuffle)

test_dataset = torchvision.datasets.MNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transform,
)
test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size,
                         shuffle=config.test_shuffle)


model = MnistModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr, 
                       weight_decay=config.weight_decay)
criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)


start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
model.train()
print("---开始训练---")
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)


        optimizer.zero_grad()


        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


        if batch_idx % 100 == 0:
            msg = f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}"
            print(msg)


model.eval()
correct = 0
total = 0

print("---开始测试---")
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)


        output = model(data)
        predicted = output.argmax(dim=1)


        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = correct / total


hyperparams["训练的开始时间"] = start_time
hyperparams["识别的准确度"] = f"{accuracy * 100}%"


if os.path.exists(config.excel_path):
    wb = openpyxl.load_workbook(config.excel_path)
    sheet = wb.active
else:
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.append(["训练的开始时间", "识别的准确度", "学习率", "标签的平滑度",
                   "权重的衰减", "训练的轮数"])

new_data = [hyperparams["训练的开始时间"], hyperparams["识别的准确度"], 
            hyperparams["学习率"], hyperparams["标签的平滑度"], 
            hyperparams["权重的衰减"], hyperparams["训练的轮数"]]
sheet.append(new_data)

wb.save(config.excel_path)


pth_name = f"{start_time}"
os.makedirs(config.checkpoints_path, exist_ok=True)
pth_path = os.path.join(config.checkpoints_path, pth_name)
torch.save(model.state_dict(), pth_path)