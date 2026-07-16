# 环境配置指南

本文档提供了项目所需的完整环境配置步骤，包括 Conda 环境创建、PyTorch 安装、依赖管理等。

## 系统要求

- **操作系统**：Windows 10/11, Linux, or macOS
- **Python 版本**：3.10+
- **CUDA 版本**：11.8（如需 GPU 加速）

---

## 第一步：创建 Conda 环境

### 1.1 创建新的 Conda 环境

```bash
# 创建名为 torch_env 的环境，指定 Python 3.10
conda create -n torch_env python=3.10 -y
```

### 1.2 激活环境

```bash
# Windows/Linux/macOS
conda activate torch_env
```

---

## 第二步：安装 PyTorch 和相关依赖

### 2.1 安装 PyTorch（带 CUDA 支持）

如果你有 NVIDIA GPU 并配置好了 CUDA 驱动，请使用以下命令：

```bash
# CUDA 11.8 版本（推荐，兼容性好）
conda install pytorch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2.2 仅 CPU 版本（无 GPU）

如果没有 NVIDIA GPU 或不想配置 CUDA：

```bash
# CPU 版本
conda install pytorch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 cpuonly -c pytorch
```

### 2.3 使用 pip 安装替代方案

如果你更倾向于使用 pip：

```bash
# CUDA 11.8 版本
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# CPU 版本
pip install torch==2.7.1+cpu torchvision==0.22.1+cpu torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

---

## 第三步：安装其他项目依赖

### 3.1 安装必要的包

```bash
pip install tensorboard
pip install wandb
```

### 3.2 安装所有依赖（完整列表）

```bash
pip install numpy==2.2.6
pip install pillow==12.2.0
pip install matplotlib==3.10.9
```

---

## 第四步：验证环境配置

### 4.1 验证 GPU/CUDA 是否可用

创建一个测试文件 `verify_env.py`，内容如下：

```python
import torch
import sys

print("=" * 60)
print("环境验证信息")
print("=" * 60)

# Python 信息
print(f"Python 版本: {sys.version}")

# PyTorch 信息
print(f"PyTorch 版本: {torch.__version__}")

# CUDA 信息
cuda_available = torch.cuda.is_available()
print(f"CUDA 可用: {cuda_available}")

if cuda_available:
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 测试简单的 CUDA 运算
    device = torch.device("cuda")
    x = torch.rand(3, 3).to(device)
    print(f"\nGPU 计算测试通过:")
    print(f"  张量设备: {x.device}")
    print(f"  张量计算:\n{x @ x.T}")
else:
    print("\n使用 CPU 模式")
    device = torch.device("cpu")
    x = torch.rand(3, 3).to(device)
    print(f"  张量设备: {x.device}")

print("\n" + "=" * 60)
print("环境验证完成!")
print("=" * 60)
```

运行验证：

```bash
python verify_env.py
```

### 4.2 快速验证命令

你也可以直接在 Python 交互式环境中运行：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 第五步：WandB 配置（可选）

### 5.1 安装 WandB

```bash
pip install wandb
```

### 5.2 登录 WandB

```bash
wandb login
```

按照提示，输入你的 WandB API key（可从 https://wandb.ai/settings 获取）

### 5.3 验证 WandB 安装

```bash
python -c "import wandb; print('WandB 版本:', wandb.__version__)"
```

---

## 完整安装步骤（一键复制）

### Windows PowerShell

```powershell
# 1. 创建并激活环境
conda create -n torch_env python=3.10 -y
conda activate torch_env

# 2. 安装 PyTorch（选择适合你系统的版本）
# CUDA 11.8 版本
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# CPU 版本（二选一）
# pip install torch==2.7.1+cpu torchvision==0.22.1+cpu torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# 3. 安装其他依赖
pip install tensorboard wandb numpy pillow matplotlib

# 4. 验证环境
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Linux/macOS

```bash
# 1. 创建并激活环境
conda create -n torch_env python=3.10 -y
conda activate torch_env

# 2. 安装 PyTorch（CUDA 11.8）
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖
pip install tensorboard wandb numpy pillow matplotlib

# 4. 验证环境
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 常见问题

### Q: CUDA 不可用怎么办？
A: 
1. 检查是否正确安装了 NVIDIA 显卡驱动
2. 验证 CUDA 工具包版本是否和 PyTorch 版本匹配
3. 可以尝试使用 CPU 版本的 PyTorch

### Q: Conda 命令找不到？
A: 确保你已经安装了 Anaconda 或 Miniconda，并将其添加到系统 PATH 中。

### Q: 如何退出 conda 环境？
A: 使用命令 `conda deactivate`

---

## 项目依赖清单

| 包名 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 编程语言 |
| torch | 2.7.1+cu118 | 深度学习框架 |
| torchvision | 0.22.1+cu118 | 计算机视觉工具库 |
| torchaudio | 2.7.1+cu118 | 音频处理工具库 |
| tensorboard | 2.21.0 | 训练可视化 |
| wandb | latest | 实验跟踪（可选） |
| numpy | 2.2.6 | 数值计算 |
| pillow | 12.2.0 | 图像处理 |
| matplotlib | 3.10.9 | 图表可视化 |

---

## 使用当前项目

环境配置完成后，就可以运行项目中的代码了：

```bash
# 运行 CIFAR-10 训练脚本
python torch_CIFAR10.py
```

## 最后更新

文档最后更新: 2026-07-16
