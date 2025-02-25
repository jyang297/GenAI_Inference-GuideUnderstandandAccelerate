# MLP(多层感知机)笔记

---

## 1. 什么是 MLP(Multi-Layer Perceptron)？
MLP(多层感知机)是一种 **前馈神经网络(Feedforward Neural Network, FNN)**，由多个 **全连接层(Fully Connected Layers, FC)** 组成，能够进行非线性映射，广泛用于分类、回归等任务。

### 1.1 结构
MLP 由三部分组成:
- **输入层(Input Layer)**:接收数据输入
- **隐藏层(Hidden Layers)**:至少包含一个隐藏层，负责特征变换
- **输出层(Output Layer)**:根据任务输出结果

MLP 的基本结构如下:
> 输入层 → 隐藏层(带激活函数) → 隐藏层(带激活函数) → 输出层

其中，每个隐藏层都包含:
- **线性变换(权重+偏置)**
- **激活函数(非线性变换，如 ReLU、Sigmoid、Tanh)**

---

## 2. MLP 的数学公式
MLP 的每一层计算可以表示为:
$$
h = f(Wx + b)
$$
其中:
- \( x \):输入向量(上一层的输出)
- \( W \):权重矩阵
- \( b \):偏置向量
- \( f(\cdot) \):激活函数(如 ReLU、Sigmoid)
- \( h \):当前层的输出

对于 **多层 MLP**，其计算过程如下:
$$
h_1 = f(W_1 x + b_1)
$$
$$
h_2 = f(W_2 h_1 + b_2)
$$
$$
y = W_3 h_2 + b_3
$$
其中，最后一层(输出层)通常不使用激活函数，或者根据任务选择适合的激活函数:
- **回归任务**:不使用激活函数或 `ReLU`
- **分类任务**:
  - **二分类**:`Sigmoid`
  - **多分类**:`Softmax`

---

## 3. MLP 的激活函数
为了引入非线性，MLP 需要使用 **激活函数**，常见的有:

| 激活函数 | 公式 | 特点 |
|----------|------|------|
| **ReLU(Rectified Linear Unit)** | \( f(x) = \max(0, x) \) | 计算高效，收敛快，但可能出现神经元死亡问题 |
| **Sigmoid** | \( f(x) = \frac{1}{1 + e^{-x}} \) | 适用于二分类，易出现梯度消失 |
| **Tanh** | \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \) | 适用于零均值数据，但仍有梯度消失问题 |
| **GELU(Gaussian Error Linear Unit)** | \( f(x) = x \Phi(x) \) | Transformer 中广泛使用 |

---

## 4. 反向传播(Backpropagation)
MLP 通过 **反向传播(Backpropagation, BP)** 进行训练，即:
1. **前向传播(Forward Pass)**:计算每层的输出
2. **计算损失(Loss)**:比较预测值与真实值
3. **反向传播(Backward Pass)**:
   - 计算损失相对于权重的梯度
   - 通过 **链式法则(Chain Rule)** 计算梯度
   - 使用 **梯度下降(SGD、Adam)** 更新权重

损失函数常见选择:
- **回归任务**:均方误差(MSE)
- **分类任务**:交叉熵损失(Cross Entropy Loss)



---

## 6. MLP 的改进
### 6.1 MLP-Mixer
- 采用 **通道混合(Channel Mixing)+ Token 混合(Token Mixing)**
- 适用于计算机视觉任务，效果接近 CNN 和 Transformer

### 6.2 ResMLP
- 引入 **残差连接(Residual Connection)**，提高梯度传播能力

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
```


## What is None-linear and why it matters?
非线性(Non-linearity) 是深度学习中最关键的特性之一。如果没有非线性，神经网络的能力会受到极大的限制，无法学习复杂的数据分布和高维特征关系。

### What will happen if it is linear?

假设一个神经网络的所有层都是线性变换，那么它的数学表示如下:




### Do I need activation function after every MLP?
MLP 最后是否需要激活函数 取决于任务的类型:

#### 回归任务(Regression) → 不使用激活函数

- 目标:预测一个连续值(如房价、温度、股票价格)

如果使用 ReLU、Sigmoid、Tanh 等激活函数，会限制输出范围，导致预测结果不符合实际需求。
```python
output_layer = nn.Linear(hidden_dim, output_dim)  # 直接输出连续数值

```
- 影响:如果最后一层用了 ReLU，那么预测值只能是 非负数，而有些回归任务可能需要预测 负值。

#### 二分类任务(Binary Classification) → 使用 Sigmoid

- 目标:预测类别 0 或 1(例如:垃圾邮件检测)

由于需要输出 概率值($0\sim 1$)，必须使用 Sigmoid:
```python
output_layer = nn.Sequential(
    nn.Linear(hidden_dim, 1),
    nn.Sigmoid()  # 将输出限制到 (0,1) 之间
)
```
#### 多分类任务(Multi-Class Classification) → 使用 Softmax

- 目标:预测多个类别(如手写数字 0-9)。

Softmax 让所有类别的概率归一化为 1:
```python
output_layer = nn.Sequential(
    nn.Linear(hidden_dim, num_classes),
    nn.Softmax(dim=1)  # 多分类任务
)
```
### What will happen without Actication Function
如果最后一层是
$$
y=Wx + b
$$
- 如果**没有激活函数**:
    - 输出的值是无约束的，可以是任意实数((-∞, +∞))。
    - 适用于**回归任务**，因为我们不希望预测值被限制在某个区间内。
- 如果**有激活函数**:
    - Sigmoid:将输出限制在 $(0,1)$。
    - Tanh:将输出限制在 $(-1,1)$。
    - ReLU:输出范围是 $[0, +∞)$。

#### 会不会影响非线性？
不会 

即使 MLP 最后没有激活函数，整个网络仍然是 非线性的，因为:

- 隐藏层已经有非线性激活函数(ReLU、Tanh 等):
- 整个 MLP 通过深度层次累积复杂的非线性特征:


## ## **2. 如果神经网络是线性的，会发生什么？**
假设一个神经网络的所有层都是**线性变换**:
$$
y = W_3(W_2(W_1 x + b_1) + b_2) + b_3
$$
可以合并成:
$$
y = (W_3 W_2 W_1) x + (W_3 b_2 + W_2 b_1 + b_3)
$$
### **问题**
- 这个公式 **等效于单层线性模型**，即**无论多少层，整个网络还是线性变换！**
- **网络的深度没有意义，无法学习复杂映射。**

---

## 非线性如何增强神经网络的表达能力？
### 线性 vs 非线性
- **线性变换**:$$ y = Wx + b $$
  - 只能做**简单缩放和平移**，无法学习**复杂曲线**
- **非线性变换**:
  - $$ y = \text{ReLU}(Wx + b) $$
  - **允许网络学习非线性关系，例如:**
    - 分类任务中的**复杂决策边界**
    - 计算机视觉中的**边缘检测**
    - NLP 任务中的**复杂语义关系**