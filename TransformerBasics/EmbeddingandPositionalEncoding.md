# Embedding

Transformer 系列模型通常规定了：**Embedding层**的输出维度 = 模型主干隐藏层维度（$d_{model}$) 

这两个部分在训练中是“共同适配、协同学习”的，embedding 输出的特征分布正好是主干网络后续注意力层、前馈层所期望的分布。

## Special Token and Vocabualry

多语言模型 有时会在相同主干网络前加不同的词表与对应 embedding（比如英、法、德三个 embedding，最终经过同一个 Transformer）。
这样做法往往需要在预训练或微调时进行特殊设计，让主干能适应多套 embedding 并对其进行联合训练。

在一些领域（如医学、生物、法律），人们可能会“在已有词表基础上新增专用 Token”，并初始化这些新 Token 的嵌入向量，做一次 微调 让它与原模型对齐。
这不是完全替换 embedding，而是 增量式 地对词表做扩充，然后把新增部分嵌入纳入训练。

## Max Token and Max SequenceLength

位置编码的 Transformer 模型里，通常会有一个最大可处理的序列长度（如 512、1024、2048 等）。我们可以把它叫做 $L_{max}$

这意味着：在训练/推理阶段，你的输入序列（Token 数）不能超过这个上限，否则就超出了模型的预设范围。

> 为什么需要这个上限？
> - 位置编码（Position Embedding）只预定义或训练到了某个固定长度；
> - 模型计算复杂度通常是$O(n^2)$ 或更高对序列长度 而言所以如果无限长就会计算不可行


对于不足最大长度的序列，一般用 padding + Attention Mask 的方式处理
- Padding：如果你的实际（序列）长度 n 小于$L_{max}$,往往会在后面补一些特殊的[PAD] token（或 0 token ID），让整条输入长度凑齐到固定的批量大小（例如在训练或推理时的批处理，需要把不同句子对齐到同一长度）。
- Attention Mask：Transformer 会有一组 mask（掩码）来指示哪些位置属于“真实 token”，哪些是“padding token”。在计算自注意力时，会让网络忽略掉 padding 的部分，不产生实际的注意力交互，也不影响损失的计算。

这样做可以在一次并行计算中处理不同长度的句子，只是要保证它们都 pad 到同样的序列长度，然后在注意力计算中使用 mask 来区分真实 token 和填充 token。

## max token 在实际中如何起作用
- 限制输入最大长度：
    
    不管是 max_seq_length=512（常见于 BERT）还是 max_tokens=4096（OpenAI API, GPT-3.5/4），本质上它就是一个硬性上限：如果你的输入超过这个长度，就需要截断、分段，或者采用别的策略；否则就会报错或无法处理。

- 对于小于此长度的输入：
    - 一般会在后面用“补 0”或[PAD]的方式对齐,（对齐到一个统一的 batch 长度或接近到 max_seq_length）。
    - 然后配合Attention Mask确保这些填充位置不会对模型输出造成影响

## 无限上下文
### Infini-attention 
 
#### 核心理念
- 局部片段处理：​将输入序列划分为固定长度的片段（segments），在每个片段内计算标准的因果点积注意力。这种方式确保了对当前片段内信息的充分捕捉。
- 压缩记忆引入：​在处理当前片段时，利用查询向量（Query）从压缩记忆中检索相关的长期上下文信息。压缩记忆通过关联矩阵的形式存储先前片段的键（Key）和值（Value）状态，并在每个片段处理后更新。
