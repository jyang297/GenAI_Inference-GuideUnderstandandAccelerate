# Transformer 基础复习

## 1. Transformer 结构概览
Transformer 由 **Encoder-Decoder 结构** 组成，适用于自然语言处理任务（如机器翻译、文本摘要等）。其关键组件包括：
- **Encoder（编码器）**：由 `N` 层堆叠而成，每层包含 **Self-Attention + FFN + 残差连接**。
- **Decoder（解码器）**：与编码器类似，但在 Self-Attention 之外还额外包含 **Encoder-Decoder Attention** 机制。

**主要流程**
1. **输入嵌入 (Input Embedding)** → 词向量 + 位置编码
2. **编码器 (Encoder)**
   - **多头自注意力 (Multi-Head Self-Attention)**
   - **前馈神经网络 (Feed-Forward Network, FFN)**
3. **解码器 (Decoder)**
   - **Masked Multi-Head Self-Attention**（防止看到未来词）
   - **Encoder-Decoder Attention**
   - **前馈神经网络 (FFN)**
4. **输出层 (Output Layer)** → 通过 Softmax 计算概率分布

---

## 2. Self-Attention 机制
自注意力（Self-Attention）是一种**捕捉输入序列不同位置之间关系**的方法，用于替代传统 RNN 处理序列数据。

### 计算过程
假设有一个长度为 `n` 的输入序列，每个单词通过嵌入后变成 `d_model` 维的向量：
1. **计算 Query (Q), Key (K), Value (V)**：
   ```
   Q = XW_Q,  K = XW_K,  V = XW_V
   ```
2. **计算 Attention Scores（相似性）**
   ```
   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
   ```

**最终输出是加权的 Value 矩阵**，用于捕捉重要信息。

---

## 3. Multi-Head Attention
Multi-Head Attention 通过多个**独立的 Self-Attention 头**来增强表达能力。

### 计算方式
1. 设有 `h` 个头，每个头独立计算：
   ```
   head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
   ```
2. **拼接多个头的输出**：
   ```
   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
   ```

**优点**
- 每个头可以关注不同的语义关系
- 提高模型的表达能力

---

## 4. Position-wise Feed-Forward Network (FFN)
用于进一步增强特征表达能力。

**FFN 结构**
```math
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```
- 第一层是 **全连接层** (`d_model → d_ff`)
- 使用 **ReLU** 激活函数
- 第二层是 **全连接层** (`d_ff → d_model`)
- 作用：非线性变换 + 提高模型能力

通常，`d_ff = 4 × d_model`。

---

## 5. 残差连接 (Residual Connections) 和 Layer Normalization
Transformer 结构中，每个子层（Self-Attention、FFN）都采用 **残差连接 (Residual Connection)** 和 **层归一化 (LayerNorm)**，帮助训练更稳定。

**计算方式**
```math
LayerNorm(x + SubLayer(x))
```
- **残差连接** (`x + SubLayer(x)`) 让原始信息更容易传播，避免梯度消失
- **LayerNorm** 使得激活值分布更稳定，加速训练

---

## 6. 位置编码 (Positional Encoding)
由于 Transformer 没有循环结构，需要额外提供 **位置信息** 来区分不同 token 的顺序。

### 常见方法
#### 1. Sinusoidal Positional Encoding
```math
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
```
```math
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
```
- 位置编码会随 token 位置变化，提供连续性信息
- 不需要训练参数

#### 2. Learned Positional Encoding
直接用**可训练参数**学习位置嵌入：
```math
PE = W_{pos}
```
适用于 **小数据集**，但**泛化能力较弱**。

---

## 总结
| 组件 | 作用 |
|------|------|
| **Self-Attention** | 计算每个 token 与其他 token 之间的依赖关系 |
| **Multi-Head Attention** | 让模型从不同角度学习信息 |
| **FFN** | 提供额外的特征变换，提高表达能力 |
| **残差连接 + LayerNorm** | 让梯度更稳定，提高训练效果 |
| **位置编码** | 提供位置信息，解决 Transformer 无序问题 |

---


