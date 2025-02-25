# Self-Attention 机制详解
Self-Attention（自注意力）是 Transformer 的核心组件，它允许模型在处理序列时关注不同位置的单词，从而更有效地捕捉句子内部的依赖关系。

## 1. Self-Attention 的计算步骤
假设输入序列长度为 \( n \)，每个单词经过嵌入后变成 \( d_{\text{model}} \) 维的向量，形成输入矩阵：
\[
X \in \mathbb{R}^{n \times d_{\text{model}}}
\]
整个计算过程包括以下几步：

### 1.1 计算 Query、Key 和 Value
使用 **可训练参数矩阵** \( W_Q, W_K, W_V \) 计算 **Query (Q)**、**Key (K)** 和 **Value (V)**：
\[
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\]
- \( W_Q, W_K, W_V \) 的形状均为 \( d_{\text{model}} \times d_k \)
- **Q, K, V 的形状**: \( n \times d_k \)

```python
import torch
import torch.nn.functional as F

# 假设 batch_size=1, 序列长度=5, 嵌入维度=8
X = torch.rand(1, 5, 8)

# 定义权重矩阵 (可训练参数)
W_Q = torch.rand(8, 8)
W_K = torch.rand(8, 8)
W_V = torch.rand(8, 8)

# 计算 Q, K, V
Q = torch.matmul(X, W_Q)  # [1, 5, 8]
K = torch.matmul(X, W_K)  # [1, 5, 8]
V = torch.matmul(X, W_V)  # [1, 5, 8]
