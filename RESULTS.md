# 训练结果

本文档记录了项目在 CIFAR-10 和 Fashion-MNIST 数据集上的训练结果。

## CIFAR-10

### 模型架构
- 简单的卷积神经网络（CNN）
- 2 个卷积层，带 ReLU 激活和最大池化
- 2 个全连接层

### 训练配置
- 批次大小 (batch_size): 64
- 学习率 (learning_rate): 0.001
- 优化器: Adam
- 学习率调度器: StepLR (step_size=10, gamma=0.1)
- 训练轮数 (epochs): 5
- 随机种子: 42

### 训练结果
- 最佳准确率: 71.4%
- 最终准确率: 71.2%

### 实验记录
所有实验都保存在本地 `runs/` 目录中，包含：
- TensorBoard 可视化文件
- 最佳模型权重
- 最终模型权重
- 实验配置和结果日志

---

## Fashion-MNIST

### 模型架构
- 简单的全连接网络（MLP）
- 3 个全连接层，带 ReLU 激活

### 训练配置
- 批次大小 (batch_size): 64
- 学习率 (learning_rate): 0.001
- 优化器: Adam
- 训练轮数 (epochs): 10

### 训练结果
- 最佳准确率: ~88-90%


```bash
# CIFAR-10
python torch_CIFAR10.py

# Fashion-MNIST (TensorBoard 版本)
python fashion_mnist_mlp_tensorboard.py
```

## 可视化训练过程

启动 TensorBoard 查看训练曲线：
然后访问 http://localhost:6006
- 准确率：![alt text](image.png)
- 损失：![alt text](image-1.png)
