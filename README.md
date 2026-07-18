# PyTorch 深度学习实践项目

这个项目包含使用 PyTorch 进行深度学习的实践代码，主要包括 Fashion-MNIST 和 CIFAR-10 图像分类任务的实现。

## 项目结构

```
study_vesion/
├── torch_CIFAR10.py          # CIFAR-10 图像分类训练脚本
├── fashion_mnist_mlp_simple.py  # Fashion-MNIST 简单全连接网络
├── fashion_mnist_mlp_tensorboard.py  # 集成 TensorBoard 的 Fashion-MNIST
├── fashion_mnist_mlp_wandb.py  # 集成 WandB 的 Fashion-MNIST
├── verify_env.py             # 环境验证脚本
├── ENVIRONMENT.md            # 详细环境配置指南
├── requirements.txt          # Python 依赖列表
└── .gitignore               # Git 忽略配置
```

## 功能特点

### CIFAR-10 训练 (`torch_CIFAR10.py`)
- 使用卷积神经网络（CNN）进行图像分类
- 集成 TensorBoard 进行训练可视化
- 自动记录实验日志和模型权重
- 固定随机种子保证可复现性
- 自动保存最佳模型和最终模型

### Fashion-MNIST 训练
- 提供三个版本：
  - `fashion_mnist_mlp_simple.py` - 基础版本，无可视化
  - `fashion_mnist_mlp_tensorboard.py` - 集成 TensorBoard
  - `fashion_mnist_mlp_wandb.py` - 集成 WandB（Weights & Biases）

## 环境配置

详细的环境配置说明请参考 [ENVIRONMENT.md](ENVIRONMENT.md)

### 快速开始

#### 1. 创建 Conda 环境

```bash
conda create -n torch_env python=3.10 -y
conda activate torch_env
```

#### 2. 安装依赖

```bash
# 安装 PyTorch（选择适合你的版本）
# CUDA 11.8 版本
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# CPU 版本
# pip install torch==2.7.1+cpu torchvision==0.22.1+cpu torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt
```

#### 3. 验证环境

```bash
python verify_env.py
```

## 使用方法

### 运行 CIFAR-10 训练

```bash
python torch_CIFAR10.py
```

### 运行 Fashion-MNIST 训练

```bash
# 简单版本
python fashion_mnist_mlp_simple.py

# TensorBoard 版本
python fashion_mnist_mlp_tensorboard.py

# WandB 版本
wandb login
python fashion_mnist_mlp_wandb.py
```

### 查看 TensorBoard 训练日志

```bash
tensorboard --logdir runs
```

然后在浏览器打开 http://localhost:6006

## 实验记录

每次运行训练会自动在 `runs/` 目录下创建实验记录文件夹，包含：
- TensorBoard 事件文件
- 最佳模型权重 (`best_model_weights.pth`)
- 最终模型权重 (`final_model_weights.pth`)
- 实验日志 (`experiment_log.txt`)

## 模型性能

### CIFAR-10
- 模型：简单 CNN
- 最佳准确率：~71.4%

### Fashion-MNIST
- 模型：简单全连接网络（MLP）
- 最佳准确率：~88-90%

## 数据集

项目会自动下载以下数据集：
- Fashion-MNIST：自动下载到 `data/` 目录
- CIFAR-10：自动下载到 `data/` 目录

## 技术栈

- **PyTorch**: 深度学习框架
- **TorchVision**: 计算机视觉工具库
- **TensorBoard**: 训练可视化
- **WandB**: 实验跟踪（可选）
- **NumPy**: 数值计算
- **Pillow**: 图像处理

## 文件说明

| 文件 | 描述 |
|------|------|
| torch_CIFAR10.py | CIFAR-10 分类主脚本 |
| fashion_mnist_mlp_simple.py | 基础 Fashion-MNIST MLP |
| fashion_mnist_mlp_tensorboard.py | Fashion-MNIST + TensorBoard |
| fashion_mnist_mlp_wandb.py | Fashion-MNIST + WandB |
| verify_env.py | 环境验证脚本 |
| ENVIRONMENT.md | 详细环境配置文档 |
| requirements.txt | 项目依赖列表 |

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 作者

深度学习实践项目
