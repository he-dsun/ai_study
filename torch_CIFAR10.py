import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

training_dataloader = DataLoader(dataset=training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # 卷积层1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 全连接层1
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        # 全连接层2
        self.fc2 = nn.Linear(512, 10)
#之前成功率一直上不去，发现是因为CIFAR10与FashionMNIST有不同，要引入卷积层；并且Flatten会导致原有特征的丢失
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        logits = self.fc2(x)
        return logits

model = NeuralNetwork().to(device)

learning_rate = 1e-3
batch_size = 64
epochs = 5

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_val, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

            global_step = epoch * len(dataloader) + batch
            writer.add_scalar('Training Loss', loss_val, global_step)# 横坐标设定为训练次数

def test_loop(dataloader, model, loss_fn, epoch):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    writer.add_scalar('Test Accuracy', correct, epoch)
    writer.add_scalar('Test Avg Loss', test_loss, epoch)

    return correct

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#这个优化器相比SGD在训练速度上有优势
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)#lr衰减
best_acc = 0.0

writer = SummaryWriter('runs/cifar10_experiment')

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(training_dataloader, model, loss_fn, optimizer, t)
    current_acc = test_loop(test_dataloader, model, loss_fn, t)

    scheduler.step()        # 每一轮结束更新学习率

    if current_acc > best_acc:
        best_acc = current_acc
        torch.save(model.state_dict(), 'best_model_weights.pth')

print("Done!")

writer.close()  #训练结束后关闭 writer

loaded_model = NeuralNetwork().to(device)
loaded_model.load_state_dict(torch.load('best_model_weights.pth', map_location=device, weights_only=True))
loaded_model.eval()

sample_X, sample_y = test_data[0][0], test_data[0][1]
sample_X = sample_X.unsqueeze(0).to(device)

with torch.no_grad():
    pred_logits = loaded_model(sample_X)
    predicted_class = pred_logits.argmax(1).item()
    
print(f"抽取图片")
print(f"真实类别: {sample_y}")
print(f"模型预测: {predicted_class}")