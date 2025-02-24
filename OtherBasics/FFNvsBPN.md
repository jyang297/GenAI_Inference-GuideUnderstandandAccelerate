# **Feedforward Network vs. Backpropagation Network 对比笔记**

## **1. 概述**
在神经网络中，**Feedforward Network（前馈网络）** 和 **Backpropagation Network（反向传播网络）** 是两个核心概念。前者描述了神经网络的结构，而后者是训练神经网络的方法。

FFN和BPN其实是指同一个东西，只是从模型结构以及训练完成后的inference的角度来看是FFN，而从训练时梯度计算的角度来看是BPN。

---

## **2. 主要区别**
| **特性** | **Feedforward Network（前馈神经网络）** | **Backpropagation Network（反向传播神经网络）** |
|----------|--------------------------------|--------------------------------|
| **定义** | 数据从输入层 → 隐藏层 → 输出层 **单向流动** 的神经网络 | 使用 **反向传播算法（Backpropagation, BP）** 训练神经网络的方式 |
| **信息流动方向** | **前向传播（Forward Pass）**，数据从输入层流向输出层 | **前向传播 + 反向传播（Backward Pass）**，通过梯度下降优化权重 |
| **目标** | **计算输出**（进行特征提取和映射） | **优化模型参数**（调整权重，使损失最小化） |
| **数学表达** | $ y = f(Wx + b) $ | 误差计算：$ L = \text{Loss}(y, \hat{y}) $，梯度更新： $ W = W - \eta \cdot \frac{\partial L}{\partial W} $ |
| **是否涉及梯度计算** | 否，仅执行向前计算 | 是，使用 **链式法则** 计算梯度 |
| **训练阶段** | 仅计算输出，不调整权重 | 通过 **梯度下降（SGD、Adam）** 迭代更新权重 |
| **核心作用** | 提供模型结构，负责特征变换 | 作为 **训练方法**，优化模型权重 |
| **使用场景** | 任何神经网络，包括 MLP、CNN、Transformer | 训练神经网络，如 MLP、CNN、RNN、Transformer |
| **关键组件** | - 线性变换（权重 + 偏置）  - 激活函数（ReLU, Sigmoid, Tanh, GELU） | - 误差计算（Loss Function） - 梯度计算（∂L/∂W） - 参数更新（Optimizer） |

---

## **3. 详细解析**
### **3.1 Feedforward Network（前馈神经网络）**
#### **定义**
- **数据单向流动**，从输入层流向输出层
- 主要用于**前向推理（Forward Inference）**
- 常见的前馈网络包括：
  - MLP（多层感知机）
  - CNN（卷积神经网络）
  - Transformer Encoder

#### **数学公式**
$$
h = f(Wx + b)
$$
其中：
- \( x \) 是输入
- \( W \) 是权重矩阵
- \( b \) 是偏置项
- \( f(\cdot) \) 是激活函数（ReLU, Sigmoid, Tanh）

#### **工作流程**
1. **输入层** 接收数据
2. **隐藏层** 进行非线性变换
3. **输出层** 生成最终预测值

---

### **3.2 Backpropagation Network（反向传播网络）**
#### **定义**
- 反向传播是 **神经网络的训练方法**，用于优化权重
- 通过计算 **损失函数的梯度** 来调整参数
- 训练目标是**最小化损失函数**

#### **数学公式**
1. **前向传播（Forward Pass）**
   $$
   y = f(Wx + b)
   $$
2. **计算损失**
   $$
   L = \text{Loss}(y, \hat{y})
   $$
3. **计算梯度（Backward Pass）**
   $$
   \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}
   $$
4. **更新权重（Gradient Descent）**
   $$
   W = W - \eta \cdot \frac{\partial L}{\partial W}
   $$
   其中：
   - \( L \) 是损失函数（如 MSE、交叉熵）
   - \( \eta \) 是学习率（Learning Rate）

#### **训练步骤**
1. **前向传播** 计算网络输出
2. **计算损失**（预测值与真实值之间的误差）
3. **反向传播** 计算梯度
4. **更新权重**（使用梯度下降优化）

---

## **4. 关系总结**
- **Feedforward Network** 是 **网络结构**，用于计算输出
- **Backpropagation Network** 是 **训练方法**，用于优化参数
- **前馈网络用于推理，反向传播用于训练**

它们是 **互补关系**，一个提供网络架构，另一个优化网络权重。

---

## **5. 代码示例**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义前馈神经网络（MLP）
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = FeedforwardNN(input_dim=10, hidden_dim=32, output_dim=1)
print(model)


## **Feedforward Network vs. Backpropagation Network（PyTorch 代码对比）**

---

### Feedforward Network（前馈神经网络）
前馈网络只进行**前向传播（Forward Pass）**，即从输入到输出的计算过程，不涉及梯度计算和参数更新。

```python
import torch
import torch.nn as nn

# 定义前馈网络
class FeedforwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = FeedforwardNetwork(input_dim=5, hidden_dim=10, output_dim=1)

# 生成随机输入数据
x = torch.rand(1, 5)  # 1个样本，5个特征

# 前向传播
output = model(x)
print("Feedforward Output:", output)

## 定义反向传播训练函数
```python
def train(model, data, target, criterion, optimizer):
    optimizer.zero_grad()      # 清空之前的梯度
    output = model(data)       # 前向传播
    loss = criterion(output, target)  # 计算损失
    loss.backward()            # 反向传播，计算梯度
    optimizer.step()           # 更新权重
    return loss.item()
# 目标输出

target = torch.rand(1, 1)  # 1个样本，1个输出

# 损失函数 & 优化器

criterion = nn.MSELoss()         # 均方误差损失（回归任务）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 使用 SGD 优化器
# 训练一步

loss = train(model, x, target, criterion, optimizer)
print("Loss after Backpropagation:", loss)
```
