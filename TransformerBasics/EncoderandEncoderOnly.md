# Transformer Encoder 结构

## 1. Transformer 结构概览
Transformer 由 **Encoder-Decoder** 结构组成，其中 Encoder 负责处理输入数据。

### **Encoder 主要组成部分**
1. **多头自注意力 (Multi-Head Self-Attention)**
2. **前馈神经网络 (Feed-Forward Network, FFN)**
3. **残差连接 (Residual Connections) + 层归一化 (Layer Normalization)**
4. **位置编码 (Positional Encoding)**

**数据流：**
输入 → 词嵌入 + 位置编码 → 多头自注意力 → 残差连接 + LayerNorm → FFN → 残差连接 + LayerNorm → 输出

---

## 2. Transformer Encoder 详细解析

### **2.1 词嵌入 (Token Embedding)**
- 句子中的每个单词都会转换成 `d_model` 维向量，例如：
  - `"I" → [0.2, 0.8, ..., 0.1]`
  - `"like" → [0.3, 0.5, ..., 0.2]`

### **2.2 位置编码 (Positional Encoding)**
Transformer 采用 **位置编码** 来表示词语顺序：

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

### **2.3 多头自注意力 (Multi-Head Self-Attention)**
- 计算方式：
  $$
  \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
  $$
- 作用：
  - 让每个词关注句子中**其他相关的词**
  - 通过**多头机制**关注不同层面的信息

### **2.4 残差连接 + 层归一化**
- 计算方式：
  $$
  \text{LayerNorm}(x + \text{SubLayer}(x))
  $$
- 作用：
  - 残差连接让梯度稳定
  - LayerNorm 防止分布漂移

### **2.5 前馈神经网络 (FFN)**
- 计算方式：
  $$
  \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
  $$
- 作用：
  - 进一步增强 Transformer 的非线性表达能力

---

## 3. Encoder-Only 架构
**适用于文本理解任务，如：**
- 文本分类（BERT、RoBERTa）
- 情感分析
- 语义搜索（S-BERT）
- 命名实体识别（NER）

### **3.1 典型架构**
Encoder-Only 结构 **只包含 Transformer Encoder，没有 Decoder**。

**常见模型：**
- **BERT（Bidirectional Encoder Representations from Transformers）**
- **RoBERTa（Robustly optimized BERT）**
- **ALBERT（A Lite BERT）**
- **DistilBERT（轻量级 BERT）**

### **3.2 BERT 结构**
1. **输入**：文本（或文本对）
2. **编码**：经过多个 Transformer Encoder 层
3. **输出**：
   - `CLS` 位置的向量（用于分类任务）
   - 每个 Token 的向量（用于问答、NER 等任务）

### **3.3 BERT 的预训练任务**
1. **Masked Language Model (MLM)**
   - 训练时随机 Mask 一些 Token，让模型预测它们
   - 例如：
     - 输入：`"I love [MASK] food."`
     - 目标：`"I love Thai food."`
2. **Next Sentence Prediction (NSP)**
   - 预测两个句子是否相关：
     - `"I went to the store."` → `"I bought some milk."` ✅
     - `"I went to the store."` → `"The weather is nice."` ❌

---

## 4. 总结

### **4.1 Transformer Encoder 结构**
| 组件 | 作用 |
|------|------|
| **Self-Attention** | 计算每个 token 与其他 token 之间的依赖关系 |
| **Multi-Head Attention** | 让模型从不同角度学习信息 |
| **FFN** | 提供额外的特征变换，提高表达能力 |
| **残差连接 + LayerNorm** | 让梯度更稳定，提高训练效果 |
| **位置编码** | 提供位置信息，解决 Transformer 无序问题 |

### **4.2 Encoder-Only 适合的任务**
| 任务类型 | 适用模型 |
|----------|---------|
| **文本分类** | BERT、RoBERTa |
| **情感分析** | BERT、DistilBERT |
| **命名实体识别（NER）** | BERT、ALBERT |
| **文本相似度** | Sentence-BERT (S-BERT) |
| **搜索（向量检索）** | ColBERT |
| **问答（非生成）** | DPR (Dense Passage Retrieval) |

