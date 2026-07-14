import torch
import torch.nn as nn


x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = torch.tensor([[4.0], [7.0], [10.0], [13.0]])

# 定义模型： y = wx + b 的线性层
model = nn.Linear(in_features=1, out_features=1)

# 定义损失函数：MSELoss (均方误差)，计算预测值与真实值差的平方的平均数
criterion = nn.MSELoss()

# 定义优化器：SGD (随机梯度下降)
optimizer = torch.optim.SGD(model.parameters(), lr=0.014)

#  核心训练
epochs = 600 # 迭代学习 600 次

for epoch in range(epochs):
    # 前向传播 (Forward)
    # 把 x 输入模型，系统根据当前的随机状态 w 和 b，算出预测值 y_pred
    y_pred = model(x)
    
    # 计算损失 
    loss = criterion(y_pred, y_true)
    
    # 清零梯度
    # 计算梯度时默认是“累加”的，所以每次算新梯度前必须手动清零之前的记录
    optimizer.zero_grad()
     
    # 反向传播 (Backward)
 
    loss.backward()
    
    # 参数更新 (Update)
    optimizer.step()
    
    # 打印
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("\n训练结束，最终参数：")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.item():.4f}")
#这里都是跟着AI到示例一步一步理解，并手敲上去；但是还没到能自己“默”的程度，只熟悉了流程