# MLP（多层感知机）笔记

---

## 1. 什么是 MLP（Multi-Layer Perceptron）？
MLP（多层感知机）是一种 **前馈神经网络（Feedforward Neural Network, FNN）**，由多个 **全连接层（Fully Connected Layers, FC）** 组成，能够进行非线性映射，广泛用于分类、回归等任务。

### 1.1 结构
MLP 由三部分组成：
- **输入层（Input Layer）**：接收数据输入
- **隐藏层（Hidden Layers）**：至少包含一个隐藏层，负责特征变换
- **输出层（Output Layer）**：根据任务输出结果

MLP 的基本结构如下：
> 输入层 → 隐藏层（带激活函数） → 隐藏层（带激活函数） → 输出层

其中，每个隐藏层都包含：
- **线性变换（权重+偏置）**
- **激活函数（非线性变换，如 ReLU、Sigmoid、Tanh）**

---

## 2. MLP 的数学公式
MLP 的每一层计算可以表示为：
$$
h = f(Wx + b)
$$
其中：
- \( x \)：输入向量（上一层的输出）
- \( W \)：权重矩阵
- \( b \)：偏置向量
- \( f(\cdot) \)：激活函数（如 ReLU、Sigmoid）
- \( h \)：当前层的输出

对于 **多层 MLP**，其计算过程如下：
$$
h_1 = f(W_1 x + b_1)
$$
$$
h_2 = f(W_2 h_1 + b_2)
$$
$$
y = W_3 h_2 + b_3
$$
其中，最后一层（输出层）通常不使用激活函数，或者根据任务选择适合的激活函数：
- **回归任务**：不使用激活函数或 `ReLU`
- **分类任务**：
  - **二分类**：`Sigmoid`
  - **多分类**：`Softmax`

---

## 3. MLP 的激活函数
为了引入非线性，MLP 需要使用 **激活函数**，常见的有：

| 激活函数 | 公式 | 特点 |
|----------|------|------|
| **ReLU（Rectified Linear Unit）** | \( f(x) = \max(0, x) \) | 计算高效，收敛快，但可能出现神经元死亡问题 |
| **Sigmoid** | \( f(x) = \frac{1}{1 + e^{-x}} \) | 适用于二分类，易出现梯度消失 |
| **Tanh** | \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \) | 适用于零均值数据，但仍有梯度消失问题 |
| **GELU（Gaussian Error Linear Unit）** | \( f(x) = x \Phi(x) \) | Transformer 中广泛使用 |

---

## 4. 反向传播（Backpropagation）
MLP 通过 **反向传播（Backpropagation, BP）** 进行训练，即：
1. **前向传播（Forward Pass）**：计算每层的输出
2. **计算损失（Loss）**：比较预测值与真实值
3. **反向传播（Backward Pass）**：
   - 计算损失相对于权重的梯度
   - 通过 **链式法则（Chain Rule）** 计算梯度
   - 使用 **梯度下降（SGD、Adam）** 更新权重

损失函数常见选择：
- **回归任务**：均方误差（MSE）
- **分类任务**：交叉熵损失（Cross Entropy Loss）

---

## 5. MLP 与 CNN、Transformer 的比较
| **模型** | **特点** | **应用场景** |
|----------|----------|------------|
| **MLP** | 逐点计算，无空间信息 | 结构化数据、基础神经网络 |
| **CNN（卷积神经网络）** | 共享卷积核，保留局部特征 | 计算机视觉（CV）、图像处理 |
| **Transformer（自注意力）** | 捕捉全局依赖关系 | NLP、视觉Transformer（ViT） |

MLP 主要用于：
- 结构化数据（如表格数据）
- 经典神经网络任务（如小规模图像分类）

---

## 6. MLP 的改进
### 6.1 MLP-Mixer
- 采用 **通道混合（Channel Mixing）+ Token 混合（Token Mixing）**
- 适用于计算机视觉任务，效果接近 CNN 和 Transformer

### 6.2 ResMLP
- 引入 **残差连接（Residual Connection）**，提高梯度传播能力

### 6.3 Vision MLP
- 结合 MLP 和 Transformer，用于图像处理任务

---

## 7. 代码示例
### 7.1 使用 PyTorch 实现 MLP
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建 MLP 模型
model = MLP(input_dim=10, hidden_dim=32, output_dim=1)
print(model)
