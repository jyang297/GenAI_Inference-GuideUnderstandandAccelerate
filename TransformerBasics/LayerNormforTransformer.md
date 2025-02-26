# **残差连接 (Residual Connection) 和 Layer Normalization**

在 Transformer 结构中，每个子层（如 **Self-Attention** 和 **Feed-Forward Network, FFN**）都包含 **残差连接 (Residual Connection)** 和 **层归一化 (Layer Normalization)**，它们的作用是提高训练稳定性，加速收敛，并增强模型的深度学习能力。

---

## **1. 残差连接 (Residual Connection)**

### **1.1 作用**
残差连接 (Residual Connection) 是 Transformer 结构的关键组件之一，其主要作用包括：
- **缓解梯度消失问题**：直接让梯度从深层传播到浅层，使深层神经网络更容易优化。
- **信息传递**：保留原始输入信息，让网络在学习新表示的同时不丢失重要的原始特征。
- **加速训练**：在多层堆叠的 Transformer 结构中，残差连接有助于更快地优化权重。

### **1.2 计算方式**
在 Transformer 中，残差连接通常用于每个子层 (Self-Attention 或 FFN) 之后，如下所示：

$$
\text{Output} = x + \text{SubLayer}(x)
$$

其中：
- $ x $ 是输入数据
- $ \text{SubLayer}(x) $ 是 Transformer 层中的某个子层（如 Self-Attention 或 FFN）
- $ x + \text{SubLayer}(x) $ 是残差连接，将输入直接加到子层输出上

### **1.3 残差连接示例**
以 Self-Attention 层为例，计算方式如下：

$$
\text{SelfAttentionOutput} = \text{MultiHeadAttention}(x)
$$
$$
\text{ResidualOutput} = x + \text{SelfAttentionOutput}
$$

然后进入 Layer Normalization：

$$
\text{LayerNormOutput} = \text{LayerNorm}(\text{ResidualOutput})
$$

---

## **2. 层归一化 (Layer Normalization)**

### **2.1 作用**
Layer Normalization (LayerNorm) 主要用于：
- **稳定训练**：减少梯度更新的变化，使不同批次的数据更易训练。
- **加速收敛**：使得模型可以更快达到优化状态，提高训练效率。
- **提升泛化能力**：减少数据分布变化带来的影响，提高模型在不同数据集上的表现。

### **2.2 计算方式**
Layer Normalization 归一化的是 **每个样本的所有特征维度**，计算如下：

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

其中：
- $ x_i $ 是当前样本的第 $ i $ 个特征
- $ \mu $ 是当前样本所有特征的均值：
  $$
  \mu = \frac{1}{d} \sum_{j=1}^{d} x_j
  $$
- $ \sigma^2 $ 是方差：
  $$
  \sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2
  $$
- $ \epsilon $ 是一个很小的数值（如 $ 10^{-6} $），防止除零错误
- $ \hat{x}_i $ 是归一化后的值

最终，LayerNorm 还会学习 **可训练参数** $ \gamma $ 和 $ \beta $ 进行缩放和平移：

$$
\text{LN}(x) = \gamma \hat{x} + \beta
$$

其中：
- $ \gamma $ 控制缩放（scale）
- $ \beta $ 控制偏移（shift）

---

## **3. Transformer 中的残差连接和 LayerNorm 组合**
在 Transformer 中，每个子层都遵循 **"Add & Norm"** 结构：
1. **残差连接** (Add)：将输入与子层输出相加
2. **层归一化** (Norm)：对相加后的结果进行归一化

示例代码：
```python
import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Self-Attention + Residual Connection + LayerNorm
        attn_output, _ = self.self_attention(x, x, x)
        x = self.layer_norm(x + attn_output)
        
        # Feed Forward + Residual Connection + LayerNorm
        ffn_output = self.ffn(x)
        x = self.layer_norm(x + ffn_output)

        return x
```


## What happened in nn.MultiheadAttention(embed_dim, num_heads)?
The nn.MultiheadAttention module implements multi-head self-attention as used in the Transformer model.
```python
nn.MultiheadAttention(embed_dim, num_heads)
```
where,
- embed_dim=d_model: The dimensionality of the input embeddings (i.e., the size of each token vector).
- num_heads: The number of attention heads.


When nn.MultiheadAttention(embed_dim=d_model, num_heads=8) is initialized, it does the following:

- Splits the embedding dimension into num_heads independent attention heads.

    - Each head will have a dimension of: \frac{\text{embed_dim}}{\text{num_heads}}
    - If d_model = 512 and num_heads = 8, then each head operates on vectors of size: $512/8=64$

- Creates learnable weight matrices for **Query**, **Key**, and **Value**:
    - Each head has its own linear projection matrices: $W_Q$, $W_K$, and $W_V$
    - The shape of these matrices: $[embed_dim, embed_dim]$
- Computes Attention Scores:
    - For each head:
    $$
    \text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
    $$
    - Each head attends to different positions in the sequence.

- Concatenates Head Outputs:
    The outputs from all attention heads are concatenated back into a $d_{model}$ dimension vector.

-  Apply Final Transformation:
    - A final projection matrix (W_O) transforms the concatenated attention output.


