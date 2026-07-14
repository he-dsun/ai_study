import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
# batch_size=64: 每次吐出 64 条数据
# shuffle=True: 打乱数据顺序
training_dataloader = DataLoader(dataset=training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#print(f"Using {device} device")       
#启用加速器CUDA
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
#这里有三个全连接层，前两个后面跟着 ReLU 激活函数，最后一个输出层没有激活函数
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
#   
model = NeuralNetwork().to(device)#实例化模型
#print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
#print(f"Predicted class: {y_pred}")

#设置超参数
learning_rate = 1e-3
batch_size = 64
epochs = 5

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
   
    model.train()                               #这一步开启了训练模式，启用 Batch Normalization 和 Dropout（随机丢弃神经元）
    for batch, (X, y) in enumerate(dataloader): #X：当前批次的特征数据（[64, 1, 28, 28] 的矩阵张量），y：当前批次的真实标签（是什么东西）
        X, y = X.to(device), y.to(device)       #注意：将数据移动到cuda上,和模型位置保持一致
        pred = model(X)
        loss = loss_fn(pred, y)

        #反向传播和优化
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):

    model.eval()        #开启评估模式
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0


    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()        #.item()：从张量中提取浮点数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()#先判断再转换为浮点数，最后求和

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct

loss_fn = nn.CrossEntropyLoss()         #这个损失函数包含了Softmax(n.LogSoftmax)和负对数似然(nn.NLLLoss)的功能
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   #尝试不同优化器

best_acc = 0.0 # 记录历史最高准确率

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(training_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    current_acc = test_loop(test_dataloader, model, loss_fn)

    if current_acc > best_acc:
        best_acc = current_acc
        torch.save(model.state_dict(), 'best_model_weights.pth')

print("Done!")


loaded_model = NeuralNetwork().to(device)

loaded_model.load_state_dict(torch.load('best_model_weights.pth', map_location=device, weights_only=True))

loaded_model.eval()

sample_X, sample_y = test_data[0][0], test_data[0][1]
sample_X = sample_X.unsqueeze(0).to(device) # 增加 Batch 维度，变成 [1, 1, 28, 28]

with torch.no_grad():
    pred_logits = loaded_model(sample_X)
    predicted_class = pred_logits.argmax(1).item()
    
print(f"抽取测试集第 1 张图片:")
print(f"真实类别: {sample_y}")
print(f"模型预测: {predicted_class}")