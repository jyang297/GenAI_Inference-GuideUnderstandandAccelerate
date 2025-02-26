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
```

### 1.2 计算 Attention Scores（相似性）

计算 Query 和 Key 之间的 **相似性分数**：
\[
\text{Scores} = QK^T
\]
- \( QK^T \) 的形状为 \( (n \times d_k) \times (d_k \times n) = n \times n \)

```python
# 计算相似性分数
scores = torch.matmul(Q, K.transpose(-2, -1))  # [1, 5, 5]
```
> The reason why compute $QK^T$
> - **Query** represents information pf **Current Word**
> - **Key** represents information of **All Words**
> - Because of using dotproduct for similarity, the higher, the more relevant.

## 1.3 Scaling
> 本部分参考了 https://zhuanlan.zhihu.com/p/503321685

为了防止梯度消失/爆炸， 我们需要对Score进行缩放:
$$
Scaled = \frac{QK^T}{\sqrt{d_k}}
$$
where,
- $d_k$ means the dimension of keys

因为Softmax是用于将输入向量转化为一个和为1的概率分布, 以向量作为输入并以向量输出. 因此, Softmax的梯度实际上是其Jacobian Matrix



### Example
假设有输入为
> Hello how are you

我们将其转化为Token:
> $t_0$ = "Hello " 
> $t_1$ = "how "
> $t_2$ = "are "
> $t_3$ = "you "

根据设计的Embedding不同, 我们假设此时选取的$d_{model}$为4, 此时可以得到对应的$X_{Hello}$, 这个示例中没有必要写出$X$具体是多少，反正要被丢给$W_Q,W_K,W_V$去生成$Q,K,V$,于是我们得到:


$$ 
Q_{Hello}=[10,10,10,10],
$$
对于$K$, 我们弄得极端一点:
$$
K_{Hello}=[8,8,8,8]\\
K_{how}=[0,0,0,1]\\
K_{are}=[0,0,0,0]\\
K_{you}=[0,0,1,0]
$$
于是对应的:
$$
\text{dot}_0=QK^T=Q\cdot K = 80+80+80+80=320\\
\text{dot}_1=QK^T=Q\cdot K = 10=10\\
\text{dot}_2=QK^T=Q\cdot K = 0+0=0\\
\text{dot}_3=QK^T=Q\cdot K = 10+0=10
$$

由此我们可以得到$logits=z=[320,10,0,10]$
送给Softmax计算可得:
$$
softmax=\frac{e^{z_i}}{\sum_j{e^{z_j}}}
$$
可以看出得到的Attention Weights差不多就是$Attention=[1,0,0,0]$


