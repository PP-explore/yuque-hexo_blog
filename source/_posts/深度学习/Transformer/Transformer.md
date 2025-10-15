---
title: Transformer
date: '2024-06-17 13:59:33'
updated: '2025-09-17 15:07:02'
categories:
  - 人工智能
tags:
  - 深度学习
  - Transformer
cover: /images/custom-cover.jpg
recommend: true
---
[Transformer学习资源&顺序推荐 - 有氧 - 博客园](https://www.cnblogs.com/youtmdyang/p/16172480.html)



Transformer是一种用于自然语言处理（NLP）和其他**序列到序列**（sequence-to-sequence）任务的深度学习模型架构，它在2017年由Vaswani等人首次提出。Transformer架构引入了自注意力机制（self-attention mechanism），这是一个关键的创新，使其在处理序列数据时表现出色。

以下是Transformer的一些重要组成部分和特点：

:::info
+ 自注意力机制（Self-Attention）：这是Transformer的核心概念之一，它使模型能够同时考虑输入序列中的所有位置，而不是像循环神经网络（RNN）或卷积神经网络（CNN）一样逐步处理。自注意力机制允许模型根据输入序列中的不同部分来赋予不同的注意权重，从而更好地捕捉语义关系。
+ 多头注意力（Multi-Head Attention）：Transformer中的自注意力机制被扩展为多个注意力头，每个头可以学习不同的注意权重，以更好地捕捉不同类型的关系。多头注意力允许模型并行处理不同的信息子空间。
+ 堆叠层（Stacked Layers）：Transformer通常由多个相同的编码器和解码器层堆叠而成。这些堆叠的层有助于模型学习复杂的特征表示和语义。
+ 位置编码（Positional Encoding）：由于Transformer没有内置的序列位置信息，它需要额外的位置编码来表达输入序列中单词的位置顺序。
+ 残差连接和层归一化（Residual Connections and Layer Normalization）：这些技术有助于减轻训练过程中的梯度消失和爆炸问题，使模型更容易训练。
+ 编码器和解码器：Transformer通常包括一个编码器用于处理输入序列和一个解码器用于生成输出序列，这使其适用于序列到序列的任务，如机器翻译。

:::

# <font style="color:rgb(77, 77, 77);">模型架构</font>
<font style="color:rgb(77, 77, 77);">今天，我们来揭示 Transformers 背后的核心概念：注意力机制、</font>[编码器-解码器](https://so.csdn.net/so/search?q=%E7%BC%96%E7%A0%81%E5%99%A8-%E8%A7%A3%E7%A0%81%E5%99%A8&spm=1001.2101.3001.7020)<font style="color:rgb(77, 77, 77);">架构、多头注意力等等。</font>![](/images/264b170065b0226e32948dbbda65157e.png)

![](/images/271d777328519759c126973e756fe805.jpeg)<font style="color:rgb(77, 77, 77);"></font>

<font style="color:#DF2A3F;">源码对照模型架构</font>

## <font style="color:rgb(79, 79, 79);">使用位置编码表示序列的顺序</font>
为什么要用位置编码？

如果不添加位置编码，那么无论单词在什么位置，它的注意力分数都是确定的。这不是我们想要的。

为了理解单词顺序，Transformer为每个输入的词嵌入添加了一个向量，这样能够更好的表达词与词之间的关系。词嵌入与位置编码相加，而不是拼接，他们的效率差不多，但是拼接的话维度会变大，所以不考虑。（这里位置向量如何得到，以哪种计算方法得到，以及词嵌入与位置编码如何结合是可以尝试实验的点，可以看以下链接[https://www.zhihu.com/question/347678607](https://www.zhihu.com/question/347678607)思考这个问题）

![](/images/ce47d38b1e4fa605ec0224be2b5a64d1.png)



<font style="color:rgba(0, 0, 0, 0.75);">为了让模型理解单词的顺序，我们添加了位置编码向量，这些向量的值遵循特定的模式。</font>

![](/images/9d4b6fdf240b053fe1bf54c44aef5ee0.png)

## <font style="color:rgb(79, 79, 79);">理解注意力机制</font>
<font style="color:rgb(77, 77, 77);">注意力机制是神经网络中一个迷人的概念，特别是在涉及到像 NLP 这样的任务时。它就像给模型一个聚光灯，让它能够集中注意力在输入序列的某些部分，同时忽略其他部分，就像我们人类在理解句子时关注特定的单词或短语一样。</font>

<font style="color:rgb(77, 77, 77);">现在，让我们深入了解一种特定类型的注意力机制*，称为自注意力，也称为内部注意力。想象一下，当你阅读一句话时，你的大脑会自动突出显示重要的单词或短语来理解意思。这就是神经网络中自注意力的基本原理。它使序列中的每个单词都能“关注”其他单词，包括自己在内，以更好地理解上下文。</font>

注意力机制有三个核心变量：查询值 Query，键值 Key 和 真值 Value。

Query查询相对应的key, 然后如果query想要关联多个key,那么我们就需要一个注意力分数,分配给不同的key不同的注意力权重, 从直观上讲,可以认为key与query关联性越高,则被赋予的注意力权重就越大.

其采用的计算注意力分数的方法是:

通过词向量能够表征语义信息，从而让语义相近的词在向量空间中距离更近，语义较远的词在向量空间中距离更远。我们往往用欧式距离来衡量词向量的相似性，但我们同样也可以用点积来进行度量：

v⋅w=∑iviwiv⋅w=i∑viwi

根据词向量的定义，语义相似的两个词对应的词向量的点积应该大于0，而语义不相似的词向量点积应该小于0。

那么，我们就可以用点积来计算词之间的相似度。假设我们的 Query 为“fruit”，对应的词向量为 q；我们的 Key 对应的词向量为 ![](/images/8e4b53ab28264dc5ea68530fe57214a2.png),则我们可以计算 Query 和每一个键的相似程度：

![](/images/82e6ef743b49c3c3dba680c81bde506e.png)

 K 即为将所有 Key 对应的词向量堆叠形成的矩阵。基于矩阵乘法的定义，x 即为 q 与每一个 k 值的点积.得到的 x 即反映了 Query 和每一个 Key 的相似程度，我们再通过一个 Softmax 层将其转化为和为 1 的权重：

![](/images/023c96d36a4222c80cf66ef31c24a372.png)

向量就能够反映 Query 和每一个 Key 的相似程度，同时又相加权重为 1，也就是我们的注意力分数了。

可以得到注意力机制计算的基本公式：

![](/images/f22d0221c15d62fdfc9b250f50d5ff51.png)

如果 Q 和 K 对应的维度 dkdk 比较大，softmax 放缩时就非常容易受影响，使不同值之间的差异较大，从而影响梯度的稳定性。因此，我们要将 Q 和 K 乘积的结果做一个放缩：

![](/images/fcf8a01c18887a8329ef6494cb2d5910.png)

```yaml
'''注意力计算函数'''
def attention(query, key, value, dropout=None):
    '''
    args:
    query: 查询值矩阵
    key: 键值矩阵
    value: 真值矩阵
    '''
    # 获取键向量的维度，键向量的维度和值向量的维度相同
    d_k = query.size(-1) 
    # 计算Q与K的内积并除以根号dk
    # transpose——相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        # 采样
     # 根据计算结果对value进行加权求和
    return torch.matmul(p_attn, value), p_attn

```

## <font style="color:rgb(79, 79, 79);">自注意力</font>
<font style="color:#000000;">注意力机制的本质是对两段序列的元素依次进行相似度计算，寻找出</font>**<font style="color:#000000;">一个序列的每个元素对另一个序列</font>**<font style="color:#000000;">的每个元素的相关度，然后基于相关度进行加权，即分配注意力。</font>

<font style="color:#000000;">实际应用中，我们往往只需要计算 Query 和 Key 之间的注意力结果，很少存在额外的真值 Value。也就是说，我们其实只需要拟合两个文本序列。在经典的 注意力机制中，Q 往往来自于一个序列，K 与 V 来自于另一个序列，都通过参数矩阵计算得到，从而可以拟合这两个序列之间的关系。例如在 Transformer 的 Decoder 结构中，Q 来自于 Decoder 的输入，K 与 V 来自于 Encoder 的输出，从而拟合了编码信息与历史信息之间的关系，便于综合这两种信息实现未来的预测。</font>

<font style="color:#000000;">所以Transformer的Encoder结构中, 使用的就是注意力机制的变种------自注意力机制.</font>

:::info
<font style="color:#000000;">即是计算本身序列中每个元素对其他元素的注意力分布，即在计算过程中，Q、K、V 都由同一个输入通过不同的参数矩阵计算得到。</font>

:::

<font style="color:#000000;">在 Encoder 中，Q、K、V 分别是</font>**<font style="color:#000000;">输入对参数矩阵 </font>****<font style="color:#000000;">Wq、Wk、Wv</font>**_**<font style="color:#000000;">Wq</font>**_**<font style="color:#000000;">、</font>**_**<font style="color:#000000;">Wk</font>**_**<font style="color:#000000;">、</font>**_**<font style="color:#000000;">Wv</font>**_**<font style="color:#000000;"> 做积</font>**<font style="color:#000000;">得到，从而拟合输入语句中每一个 token 对其他所有 token 的关系。</font>

<font style="color:#000000;">通过自注意力机制，我们可以找到一段文本中每一个 token 与其他所有 token 的相关关系大小，从而建模文本之间的依赖关系。在代码中的实现，self-attention 机制其实是通过给 Q、K、V 的输入传入同一个参数实现的</font><font style="color:rgb(238, 254, 255);background-color:rgb(29, 31, 32);">：</font>

```java
attention(x, x, x)
```

## <font style="color:#000000;">掩码自注意力机制  Mask Self-Attention</font>
<font style="color:#000000;">使用注意力掩码的自注意力机制。掩码的作用是遮蔽一些特定位置的 token，模型在学习的过程中，会忽略掉被遮蔽的token。</font>

<font style="color:#000000;">核心动机是让模型只能使用历史信息进行预测而不能看到未来信息。</font>

<font style="color:#000000;">Transformer 模型也是通过不断根据之前的 token 来预测下一个 token，直到将整个文本序列补全。</font>

<font style="color:#000000;">那么我们可以采用掩码机制来让训练并行化, 构造一个上三角掩码矩阵:</font>

```yaml
<BOS> 【MASK】【MASK】【MASK】【MASK】
<BOS>    I   【MASK】 【MASK】【MASK】
<BOS>    I     like  【MASK】【MASK】
<BOS>    I     like    you  【MASK】
<BOS>    I     like    you   </EOS>

```

<font style="color:#000000;">在具体实现中，我们通过以下代码生成 Mask 矩阵：</font>

```python
# 创建一个上三角矩阵，用于遮蔽未来信息。
# 先通过 full 函数创建一个 1 * seq_len * seq_len 的矩阵
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
# triu 函数的功能是创建一个上三角矩阵
mask = torch.triu(mask, diagonal=1)
```

<font style="color:#000000;">生成的 Mask 矩阵会是一个上三角矩阵，上三角位置的元素均为 -inf，其他位置的元素置为0。</font>

<font style="color:#000000;">在注意力计算时，我们会将计算得到的注意力分数与这个掩码做和，再进行 Softmax 操作：</font>

```python
# 此处的 scores 为计算得到的注意力分数，mask 为上文生成的掩码矩阵
scores = scores + mask[:, :seqlen, :seqlen]
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```

<font style="color:#000000;">做 Softmax 操作，</font>`<font style="color:#000000;">-inf</font>`<font style="color:#000000;"> 的值在经过 Softmax 之后会被置为 0，从而忽略了上三角区域计算的注意力分数，从而实现了注意力遮蔽。</font>

## <font style="color:#000000;">多头注意力机制</font>
<font style="color:#000000;"> 	Transformer 使用了多头注意力机制（Multi-Head Attention），即同时对一个语料进行多次注意力计算，每次注意力计算都能拟合不同的关系，将最后的多次结果拼接起来作为最后的输出，即可更全面深入地拟合语言信息。</font>

<font style="color:#000000;">就是将原始的输入序列进行多组的自注意力处理；然后再将每一组得到的自注意力结果拼接起来，再通过一个线性层进行处理，得到最终的输出。</font>

![](/images/e4fffa2bd8d32f3c7c578b077e0147ec.png)

<font style="color:#000000;">我们可以通过矩阵运算巧妙地实现并行的多头计算，其核心逻辑在于使用三个组合矩阵来代替了n个参数矩阵的组合，也就是矩阵内积再拼接其实等同于拼接矩阵再内积。具体实现可以参考下列代码：</font>

![](/images/859cd8416ff5ff8369a19b85e262062d.png)

<font style="color:#000000;">前馈层只需要一个矩阵，则把得到的8个矩阵拼接在一起，然后用一个附加的权重矩阵</font>![image](/images/f9a64a46870a19bf45220c1abefc7eec.svg)<font style="color:#000000;">与它们相乘。</font>

![](/images/6a6c96b09e1bd0b827796ec72b7c9e7a.png)

![](/images/b19bab71843a83c43b11f729576761e7.png)

```python
import torch.nn as nn
import torch

'''多头自注意力计算模块'''
class MultiHeadAttention(nn.Module):

    def __init__(self, args: ModelArgs, is_causal=False):
        # 构造函数
        # args: 配置对象
        super().__init__()
        
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.dim % args.n_heads == 0
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x dim
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        # 不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        # 输出权重矩阵，维度为 dim x dim（head_dim = dim / n_heads）
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # 获取批次大小和序列长度，[batch_size, seq_len, n_embd]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, dim) -> (B, T, dim)
        #通过线性层随机权重来复制nhead份
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, dim // n_head)，然后交换维度，变成 (B, n_head, T, dim // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 做 Dropout
        scores = self.attn_dropout(scores)
        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, dim // n_head)，再拼接成 (B, T, n_head * dim // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

```

<font style="color:#000000;">总结整个流程：</font>![](/images/74f87b91c20608543934baa83a518e58.png)

## <font style="color:#000000;">Encoder-Decoder</font>
<font style="color:#000000;">在 Transformer 中，使用注意力机制的是其两个核心组件——Encoder（编码器）和 Decoder（解码器）。事实上，后续基于 Transformer 架构而来的预训练语言模型基本都是对 Encoder-Decoder 部分进行改进来构建新的模型架构，例如只使用 Encoder 的 BERT、只使用 Decoder 的 GPT 等。</font>

### <font style="color:#000000;">Seq2Seq 模型</font>
> <font style="color:#000000;">Seq2Seq，即序列到序列，是一种经典 NLP 任务。具体而言，是指模型输入的是一个自然语言序列 </font><font style="color:#000000;">input=(x1,x2,x3...xn)</font><font style="color:#000000;">input</font><font style="color:#000000;">=(</font><font style="color:#000000;">x</font><font style="color:#000000;">1,</font><font style="color:#000000;">x</font><font style="color:#000000;">2,</font><font style="color:#000000;">x</font><font style="color:#000000;">3...</font><font style="color:#000000;">xn</font><font style="color:#000000;">)</font><font style="color:#000000;"> ，输出的是一个可能不等长的自然语言序列 </font><font style="color:#000000;">output=(y1,y2,y3...ym)</font><font style="color:#000000;">output</font><font style="color:#000000;">=(</font><font style="color:#000000;">y</font><font style="color:#000000;">1,</font><font style="color:#000000;">y</font><font style="color:#000000;">2,</font><font style="color:#000000;">y</font><font style="color:#000000;">3...</font><font style="color:#000000;">ym</font><font style="color:#000000;">)</font><font style="color:#000000;"> 。事实上，Seq2Seq 是 NLP 最经典的任务，几乎所有的 NLP 任务都可以视为 Seq2Seq 任务。</font>
>

<font style="color:#000000;"></font><font style="color:#000000;">Seq2Seq 任务，一般的思路是对自然语言序列进行编码再解码。所谓编码，就是将输入的自然语言序列通过隐藏层编码成能够表征语义的向量（或矩阵），可以简单理解为更复杂的词向量表示。而解码，就是对输入的自然语言序列编码得到的向量或矩阵通过隐藏层输出，再解码成对应的自然语言目标序列。通过编码再解码，就可以实现 Seq2Seq 任务。</font>

<font style="color:#000000;">Transformer 中的 Encoder，就是用于上述的编码过程；Decoder 则用于上述的解码过程。Transformer 结构:</font>

![](/images/954dc10bf312983280a658549f823299.jpg)

<font style="color:#000000;">Transformer 由 Encoder 和 Decoder 组成，每一个 Encoder（Decoder）又由 6个 Encoder（Decoder）Layer 组成</font>

<font style="color:#000000;">输入源序列会进入 Encoder 进行编码，到 Encoder Layer 的最顶层再将编码结果输出给 Decoder Layer 的每一层，通过 Decoder 解码后就可以得到输出目标序列了</font>

<font style="color:#000000;">以下是 Encoder 和 Decoder 内部传统神经网络的经典结构——前馈神经网络（FNN）、层归一化（Layer Norm）和残差连接（Residual Connection），然后进一步分析 Encoder 和 Decoder 的内部结构。</font>

<font style="color:#000000;"></font>

## <font style="color:#000000;">Feed Forward Neural Network  前馈神经网络(全连接层)</font>
> <font style="color:#000000;">Transformer 的前馈神经网络是由两个线性层中间加一个 RELU 激活函数组成的，以及前馈神经网络还加入了一个 Dropout 层来防止过拟合。</font>
>

![](/images/35ba72ba754b841862cc397e9932abba.png)<font style="color:#000000;">  
</font><font style="color:#000000;">全连接层是一个两层的神经网络，先线性变换，然后ReLU非线性，再线性变换。  
</font><font style="color:#000000;">这两层网络就是为了将输入的Z映射到更加高维的空间中然后通过非线性函数ReLU进行筛选，筛选完后再变回原来的维度。  
</font><font style="color:#000000;">经过6个encoder后输入到decoder中。</font>

```python
class MLP(nn.Module):
    '''前馈神经网络'''
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和RELU激活函数
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))
    

```

### <font style="color:rgb(79, 79, 79);">Add&Normalize（相加和层归一化）</font>
<font style="color:#000000;">在经过多头注意力机制得到矩阵Z之后，并没有直接传入全连接神经网络，而是经过了一步Add&Normalize。</font>![](/images/2653248039915cbef9705824874e7882.png)

<font style="color:#000000;">Add & Norm 层由 Add 和 Norm 两部分组成，其计算公式如下：</font>

![](/images/90f442f99b24239f5b814b710cb5b1d3.png)

<font style="color:#000000;">其中 X表示 Multi-Head Attention 或者 Feed Forward 的输入，MultiHeadAttention(X) 和 FeedForward(X) 表示输出 (输出与输入 X 维度是一样的，所以可以相加)。</font>

1. <font style="color:#000000;">Add  ResNet残差神经网络</font>
2. <font style="color:rgb(238, 254, 255);background-color:rgb(29, 31, 32);">由于 Transformer 模型结构较复杂、层数较深，为了避免模型退化，Transformer 采用了残差连接的思想来连接每一个子层。残差连接，即下一层的输入不仅是上一层的输出，还包括上一层的输入。残差连接允许最底层信息直接传到最高层，让高层专注于残差的学习。</font>

<font style="color:#000000;">Add，就是在z的基础上加了一个残差块X，加入残差块的目的是为了防止在深度神经网络的训练过程中发生退化的问题，退化的意思就是深度神经网络通过增加网络的层数，Loss逐渐减小，然后趋于稳定达到饱和，然后再继续增加网络层数，Loss反而增大。</font>

![](/images/c6d17cbdb795e6944b3a94fbdaf6dd66.png)

在 Encoder 中，在第一个子层，输入进入多头自注意力层的同时会直接传递到该层的输出，然后该层的输出会与原输入相加，再进行标准化。在第二个子层也是一样。即：

![](/images/f2f63237bd791b01cf606ed867caadb8.png)

<font style="color:#000000;"></font>

<font style="color:#000000;">要恒等映射，我们只需要让F（X）=0就可以了。x经过线性变换（随机初始化权重一般偏向于0），输出值明显会偏向于0，而且经过激活函数Relu会将负数变为0，过滤了负数的影响。</font>

<font style="color:#000000;">这样当网络自己决定哪些网络层为冗余层时，使用ResNet的网络很大程度上解决了学习恒等映射的问题，用学习残差F(x)=0更新该冗余层的参数来代替学习h(x)=x更新冗余层的参数。</font>

3. <font style="color:#000000;">层归一化 Layer Nrom </font>

> <font style="color:#000000;">归一化核心是为了让不同层输入的取值范围或者分布能够比较一致。由于深度神经网络中每一层的输入都是上一层的输出，因此多层传递下，对网络中较高的层，之前的所有神经层的参数变化会导致其输入的分布发生较大的改变。也就是说，随着神经网络参数的更新，各层的输出分布是不相同的，且差异会随着网络深度的增大而增大。但是，需要预测的条件分布始终是相同的，从而也就造成了预测的误差。</font>
>

<font style="color:#000000;">归一化目的：</font>

<font style="color:#000000;">1、加快训练速度</font>

<font style="color:#000000;">2、提高训练的稳定性</font>

<font style="color:#000000;">使用到的归一化方法是Layer Normalization。</font>

![](/images/b5d6664bfc513f074ff404dd6ac13bf0.png)

<font style="color:#DF2A3F;">LN</font><font style="color:#000000;">是在同一个样本中不同神经元之间进行归一化，而</font><font style="color:#DF2A3F;">BN</font><font style="color:#000000;">是在同一个batch中不同样本之间的同一位置的神经元之间进行归一化。</font>

<font style="color:#000000;">BN是对于相同的维度进行归一化，但是咱们NLP中输入的都是词向量，一个300维的词向量，单独去分析它的每一维是没有意义地，因此这里选用的是LN。</font>

#### <font style="color:#000000;">Layer Norm</font>
<font style="color:rgb(238, 254, 255);background-color:rgb(29, 31, 32);">相较于 Batch Norm 在每一层统计所有样本的均值和方差，Layer Norm 在每个样本上计算其所有层的均值和方差，从而使每个样本的分布达到稳定。Layer Norm 的归一化方式其实和 Batch Norm 是完全一样的，只是统计统计量的维度不同。</font>

```python
class LayerNorm(nn.Module):
    ''' Layer Norm 层'''
    def __init__(self, features, eps=1e-6):
    super().__init__()
    # 线性矩阵做映射
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps
    
    def forward(self, x):
    # 在统计每个样本所有维度的值，求均值和方差
    mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
    std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
    # 注意这里也在最后一个维度发生了广播
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

```

+ **LayerNorm** 是在最后一个维度 `dim` 上进行归一化的。
+ 对于输入张量 `x ∈ [batch, seq, dim]`，会 **逐个样本、逐个序列位置** 来做归一化。
+ 也就是说：
+ 固定 `batch_idx` 和 `seq_idx`，取出向量 `x[batch_idx, seq_idx, :] ∈ [dim]`。
+ 在这个 `[dim]` 向量上计算：

![](/images/bd72b982d17fcd00f550038fd5a151c1.png)

 然后对该向量的每个元素做：  ![](/images/09886c8f4e0d3fc88b5e2b6f30a0ee99.png)

 最后再乘以可训练参数 `γ`，加上 `β`：![](/images/85a090358b925c97e560d4324ae33eeb.png)  

γ∈R d_model：可学习缩放参数（初始化为1）

β∈R d_model：可学习平移参数（初始化为0）



因此，对于一个 batch 里大小 `[batch, seq, dim]` 的张量：

    - 一共会执行 `batch × seq` 次归一化操作；
    - 每次操作的是一个长度为 `dim` 的向量。
    -  广播机制：计算出的均值 `μ` 和标准差 `σ` 是标量（针对每个 `[dim]` 向量），它们会广播到该向量的所有 `dim` 位置去参与计算。  

#### <font style="color:#000000;">Batch Norm</font>
批归一化是指在一个 mini-batch 上进行归一化，<font style="background-color:#FBDE28;">也就是在某个相同词嵌入维度的切面上进行归一化</font>，相当于对一个 batch 对样本拆分出来一部分，首先计算样本的均值：



![](/images/31539ee7b6300c50b6a9a2869cfb67a9.png)  
先计算样本的均值![](/images/6b5d6a3dde14a56de159a31bef9b42fa.png)其中，ZjiZji 是样本 i 在第 j 个维度上的值，m 就是 mini-batch 的大小。

再计算样本的方差：![](/images/ae2d322f411fe14cba8accaf2ebbb9c1.png)

最后，对每个样本的值减去均值再除以标准差来将这一个 mini-batch 的样本的分布转化为标准正态分布：![](/images/3875f4e5e00123eb5d52be38a4ee2be1.png)

此处加上 ϵ 这一极小量是为了避免分母为0。

缺点：

+ 当显存有限，mini-batch 较小时，Batch Norm 取的样本的均值和方差不能反映全局的统计分布信息，从而导致效果变差；
+ 对于在时间维度展开的 RNN，不同句子的同一分布大概率不同，所以 Batch Norm 的归一化会失去意义；
+ 在训练时，Batch Norm 需要保存每个 step 的统计信息（均值和方差）。在测试时，由于变长句子的特性，测试集可能出现比训练集更长的句子，所以对于后面位置的 step，是没有训练的统计量使用的；
+ 应用 Batch Norm，每个 step 都需要去保存和计算 batch 统计量，耗时又耗力

####  Pre-LN  or Post-LN  
Post-LN (原始Transformer结构)

结构: x -> Sublayer(x) -> x + Sublayer(x) -> LayerNorm -> output

解释: 输入 x 先通过子层（如多头注意力），然后与原始的 x 进行残差连接，最后再进行层归一化。这是论文《Attention Is All You Need》中提出的原始设计。

Pre-LN (现在更主流的结构)

结构: x -> LayerNorm -> Sublayer(LayerNorm(x)) -> x + Sublayer(LayerNorm(x)) -> output

解释: 输入 x 首先经过层归一化，然后将归一化后的结果送入子层，最后再进行残差连接。

:::info
要先 LayerNorm 再算 Attention的核心原因：**数值稳定性 + 更容易训练深层网络。**

:::

残差结构的初衷是：

y=f(x)+x

这样就算 `f(x)` 学不到什么，梯度也能直接从 `y` 回传到 `x`，因为有一个 **恒等映射 (identity mapping)**。

    - 前向：模型至少能复制输入，不会完全塌掉。
    - 反向：梯度可以顺畅传播，避免梯度消失。

**Post-LN里面y=LayerNorm(x+f(x))  ， LayerNorm 会对 **`**(x + f(x))**`** 重新做均值和方差归一化  ，结果导致 残差的恒等映射不再是“干净的 x”，而是被归一化过的版本  ， 所以反向传播时，梯度不能直接“原封不动”地流回去，会经过 LayerNorm 的缩放、平移，可能导致梯度消失或放大。  **

**Pre-LN 里 y=x+f(LayerNorm(x)) ，**`**x**`**直接加到输出，没有被 LayerNorm 改动  ，就算 **`**f**`** 部分训练不好，**`**y ≈ x**`** 依然能稳定传播。梯度也能沿着这条干净的残差路径直接传回去，不会衰减太多。**

## <font style="color:rgb(77, 77, 77);">查询、键和值向量</font>
<font style="color:rgb(77, 77, 77);">接下来，模型为序列中的每个单词</font>**<font style="color:rgb(77, 77, 77);">计算三个向量</font>**<font style="color:rgb(77, 77, 77);">：查询向量、键向量和值向量。</font>

<font style="color:#DF2A3F;">从</font><font style="color:rgb(77, 77, 77);">每个编码器的输入向量（每个单词的词向量，即Embedding，可以是任意形式的词向量，比如说word2vec，GloVe，one-hot编码）中</font><font style="color:#DF2A3F;">生成三个向量</font><font style="color:rgb(77, 77, 77);">，即查询向量、键向量和一个值向量。</font>

<font style="color:rgb(77, 77, 77);">（这三个向量是</font><font style="color:#DF2A3F;">通过</font><font style="color:rgb(77, 77, 77);">词嵌入与</font><font style="color:#DF2A3F;">三个权重矩阵即W^Q,W^K,W^V </font><font style="color:rgb(77, 77, 77);">相乘后创建出来的）</font>

<font style="color:rgb(77, 77, 77);">新向量在维度上往往比词嵌入向量更低(由权重矩阵最后一个维度相关)。（512->64）在训练过程中，模型学习这些向量，每个向量都有不同的作用。</font>

1. <font style="color:rgb(77, 77, 77);">查询向量表示单词的查询，即模型在序列中寻找的内容。</font>
2. <font style="color:rgb(77, 77, 77);">键向量表示单词的键，即序列中其他单词应该注意的内容。</font>
3. <font style="color:rgb(77, 77, 77);">值向量表示单词的值，即单词对输出所贡献的信息。</font>

![](/images/e218b1283d5d0b840329b699c1c7f4fe.png)

<font style="color:rgb(77, 77, 77);">更一般的，将以上所得到的查询向量、键向量、值向量组合起来就可以得到三个向量矩阵Query、Keys、Values。</font>

![](/images/acfdffa3fa56cfe94ffeca17e20f6f1b.png)

### <font style="color:rgb(77, 77, 77);">注意力分数</font>
<font style="color:rgb(77, 77, 77);">一旦模型计算了每个单词的查询、键和值向量，它就会为序列中的每一对单词计算注意力分数。这些</font><font style="color:#DF2A3F;">分数</font><font style="color:rgb(77, 77, 77);">是</font><font style="color:#DF2A3F;">通过所有</font><font style="color:rgb(77, 77, 77);">输入句子的单词的</font><font style="color:#DF2A3F;">键向量</font><font style="color:rgb(77, 77, 77);">与“Thinking”的查询向量相</font><font style="color:#DF2A3F;">点积</font><font style="color:rgb(77, 77, 77);">来计算的。这通常通过取</font><font style="color:#DF2A3F;">查询向量和键向量的点积</font><font style="color:rgb(77, 77, 77);">来实现，以评估单词之间的相似性。</font>

![](/images/c8c25ed3bd1240d83384c630ce2df4c9.png)

接下来是将分数除以8(<font style="background-color:#FBDE28;">8是论文中使用的键向量的维数64的平方根</font>，这会让梯度更稳定。这里也可以使用其它值，8只是默认值，这样做是为了防止内积过大,影响后面softmax分配权重)，然后通过softmax传递结果。



### <font style="color:rgb(77, 77, 77);">SoftMax 归一化</font>
[Transformer模型-softmax的简明介绍：转为概率分布，马太效应_softmax在transformer-CSDN博客](https://blog.csdn.net/ank1983/article/details/137197538)

<font style="color:rgb(77, 77, 77);">然后，使用 softmax 函数对注意力分数进行归一化，以获得</font>**<font style="color:#DF2A3F;">注意力权重</font>**<font style="color:rgb(77, 77, 77);">。这些权重表示每个单词应该关注序列中其他单词的程度。随着模型处理输入序列的每个单词，自注意力会关注整个输入序列的所有单词，帮助模型对本单词更好地进行编码。softmax的作用是使所有单词的分数归一化，得到的分数都是正值且和为1。注意力权重较高的单词被认为对正在执行的任务更为关键。</font>

### <font style="color:rgb(77, 77, 77);">加权求和</font>
<font style="color:rgb(77, 77, 77);">最后，将</font><font style="color:#DF2A3F;">每个值向量乘以softmax分数</font><font style="color:rgb(77, 77, 77);">(最终的注意力分数)。使用注意力权重计算值向量的加权和。这产生了每个序列中单词的</font>[自注意力机制](https://so.csdn.net/so/search?q=%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&spm=1001.2101.3001.7020)<font style="color:rgb(77, 77, 77);">输出，捕获了来自其他单词的上下文信息。</font>

![](/images/e98eb09bdf4048ea667b063eee0ecdf1.png)

整体的计算图如图所示：

![](/images/a1a40569ed257913ea4ee0f647fa670f.png)

<font style="color:rgb(77, 77, 77);">最终得到了自注意力，并将得到的向量传递给</font>**<font style="color:rgb(77, 77, 77);">前馈神经网络</font>**<font style="color:rgb(77, 77, 77);">。以上合为一个公式计算自注意力层的输出。</font>

<font style="color:rgb(77, 77, 77);">下面是一个计算注意力分数的简单解释：</font>

```python
# 安装 PyTorch
!pip install torch==2.2.1+cu121

# 导入库
import torch
import torch.nn.functional as F

# 示例输入序列
input_sequence = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

# 生成 Key、Query 和 Value 矩阵的随机权重
random_weights_key = torch.randn(input_sequence.size(-1), input_sequence.size(-1))
random_weights_query = torch.randn(input_sequence.size(-1), input_sequence.size(-1))
random_weights_value = torch.randn(input_sequence.size(-1), input_sequence.size(-1))

# 计算 Key、Query 和 Value 矩阵
key = torch.matmul(input_sequence, random_weights_key)
query = torch.matmul(input_sequence, random_weights_query)
value = torch.matmul(input_sequence, random_weights_value)

# 计算注意力分数
attention_scores = torch.matmul(query, key.T) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))

# 使用 softmax 函数获得注意力权重
attention_weights = F.softmax(attention_scores, dim=-1)

# 计算 Value 向量的加权和
output = torch.matmul(attention_weights, value)

print("自注意力机制后的输出:")
print(output)
```





## <font style="color:rgb(77, 77, 77);">解码器</font><font style="color:rgb(79, 79, 79);">Decoder整体结构</font>
![](/images/c9c9444eb8b019faa8b3ba3eceb6c98c.png)

和Encoder Block一样，Decoder也是由6个decoder堆叠而成的，Nx=6。包含两个 Multi-Head Attention 层<font style="color:rgb(238, 254, 255);background-color:rgb(29, 31, 32);">和一个前馈神经网络组成</font>。第一个 Multi-Head Attention 层采用了 Masked 操作。第二个 Multi-Head Attention 层的K, V矩阵使用 Encoder 的编码信息矩阵C进行计算，而Q使用上一个 Decoder block 的输出计算。

<font style="color:rgb(77, 77, 77);">解码器层包括以下组件：</font>

1. <font style="color:rgba(0, 0, 0, 0.75);">输出嵌入右移：在处理输入序列之前，模型将输出嵌入向右移动一个位置。这确保解码器中的每个标记在训练期间都能从先前生成的标记接收到正确的上下文。</font>
2. <font style="color:rgba(0, 0, 0, 0.75);">位置编码：与编码器类似，模型将位置编码添加到输出嵌入中，以合并标记的顺序信息。这种编码帮助解码器根据标记在序列中的位置进行区分。</font>
3. <font style="color:rgba(0, 0, 0, 0.75);">掩码的多头自注意力机制：解码器采用掩码的多头自注意力机制，以便注意输入序列的相关部分和先前生成的标记。在训练期间，模型应用掩码以防止注意到未来的标记，确保每个标记只能注意到前面的标记。</font>
4. <font style="color:rgba(0, 0, 0, 0.75);">编码器-解码器注意力机制：除了掩码的自注意力机制外，解码器还包括编码器-解码器注意力机制。这种机制使解码器能够注意到输入序列的相关部分，有助于生成受输入上下文影响的输出标记。</font>
5. <font style="color:rgba(0, 0, 0, 0.75);">位置逐点前馈网络：在注意力机制之后，解码器对每个标记独立地应用位置逐点前馈网络。这个网络捕捉输入和先前生成的标记中的复杂模式和关系，有助于生成准确的输出序列。</font>

### Embedding层
Embedding 层其实是一个存储固定大小词典的嵌入向量查找表。

在输入神经网络之前，我们往往会先让自然语言输入通过分词器 tokenizer，分词器的作用是把自然语言输入切分成 token 并转化成一个固定的 index。实际情况下，tokenizer 的工作会比这更复杂。分词有多种不同的方式，可以切分成词、切分成子词、切分成字符等，而词表大小则往往高达数万数十万。

Embedding 层的输入往往是一个形状为 （batch_size，seq_len，1）的矩阵，第一个维度是一次批处理的数量，第二个维度是自然语言序列的长度，第三个维度则是 token 经过 tokenizer 转化成的 index 值。

Embedding 内部其实是一个可训练的（Vocab_size，embedding_dim）的权重矩阵，词表里的每一个值，都对应一行维度为 embedding_dim 的向量。对于输入的值，会对应到这个词向量，然后拼接成（batch_size，seq_len，embedding_dim）的矩阵输出。

下面是直接使用pytorch的embedding层：`self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)`

### 位置编码
![](/images/5e6ec9cc2942500414b06e91091876e5.png)

对一个长度为 4 的句子"I like to code"，其词向量假设为【4，4】，每一行代表的就是一个词，则位置编码可以表示为：

![](/images/8f38881d10681b85e575b0f36db086ec.png)

这样的位置编码主要有两个好处：

1. 使 PE 能够适应比训练集里面所有句子更长的句子，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。
2. 可以让模型容易地计算出相对位置，对于固定长度的间距 k，PE(pos+k) 可以用 PE(pos) 计算得到。因为 Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)。当然具体的数学理论推到比较复杂
3. 10000为实验所得结果， 2i/d_model= i/(d_model/2)保证偶数维度使用不同的sin，奇数维度使用不同的cos

```python

class PositionalEncoding(nn.Module):
    '''位置编码模块'''

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        # self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x

```



## 最终的Transformer
```python
class Transformer(nn.Module):
   '''整体模型'''
    def __init__(self, args):
        super().__init__()
        # 必须输入词表大小和 block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),
            wpe = PositionalEncoding(args),
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    '''统计所有参数的数量'''
    def get_num_params(self, non_embedding=False):
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    '''初始化权重'''
    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    '''前向计算函数'''
    def forward(self, idx, targets=None):
        # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("idx",idx.size())
        # 通过 Embedding 层
        tok_emb = self.transformer.wte(idx)
        print("tok_emb",tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb) 
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        # 然后通过 Encoder
        print("x after wpe:",x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:",enc_out.size())
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        if targets is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

```

## <font style="color:rgb(79, 79, 79);">六、模型的训练与评估</font>
<font style="color:rgb(77, 77, 77);">训练Transformer模型涉及优化其参数以最小化损失函数，通常使用梯度下降和反向传播。一旦训练完成，就会使用各种指标评估模型的性能，以评估其解决目标任务的有效性。</font>

**<font style="color:rgb(77, 77, 77);">训练过程</font>**

<font style="color:rgb(77, 77, 77);">梯度下降和反向传播：</font>

+ <font style="color:rgba(0, 0, 0, 0.75);">在训练期间，将输入序列输入模型，并生成输出序列。</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">将模型的预测与地面真相进行比较，涉及使用损失函数（例如交叉熵损失）来衡量预测值与实际值之间的差异。</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">梯度下降用于更新模型的参数，使损失最小化的方向。</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">优化器根据这些梯度调整参数，迭代更新它们以提高模型性能。</font>

<font style="color:rgb(77, 77, 77);">学习率调度：</font>

+ <font style="color:rgba(0, 0, 0, 0.75);">可以应用学习率调度技术来动态调整训练期间的学习率。</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">常见策略包括热身计划，其中学习率从低开始逐渐增加，以及衰减计划，其中学习率随时间降低。</font>

**<font style="color:rgb(77, 77, 77);">评估指标</font>**

<font style="color:rgb(77, 77, 77);">困惑度：</font>

+ <font style="color:rgba(0, 0, 0, 0.75);">困惑度是用于评估语言模型性能的常见指标，包括Transformer。</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">它衡量模型对给定标记序列的预测能力。</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">较低的困惑度值表示更好的性能，理想值接近词汇量大小。</font>

<font style="color:rgb(77, 77, 77);">BLEU分数：</font>

+ <font style="color:rgba(0, 0, 0, 0.75);">BLEU（双语评估研究）分数通常用于评估机器翻译文本的质量。</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">它将生成的翻译与一个或多个由人类翻译人员提供的参考翻译进行比较。</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">BLEU分数范围从0到1，较高的分数表示更好的翻译质量。</font>

### <font style="color:rgb(79, 79, 79);">七、训练和评估的实现</font>
<font style="color:rgb(77, 77, 77);">让我们使用PyTorch对Transformer模型进行训练和评估的基本代码实现：</font>

```plain
# Transformer 模型的训练和评估
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# 训练循环
transformer.train()

for epoch in range(10):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:]
    .contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"第 {epoch+1} 轮：损失= {loss.item():.4f}")


# 虚拟数据
src_data = torch.randint(1, src_vocab_size, (5, max_len))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (5, max_len))  # (batch_size, seq_length)

# 评估循环
transformer.eval()
with torch.no_grad():
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:]
    .contiguous().view(-1))
    print(f"\n虚拟数据的评估损失= {loss.item():.4f}")



```

### <font style="color:rgb(79, 79, 79);">八、高级主题和应用</font>
<font style="color:rgb(77, 77, 77);">Transformers 在自然语言处理（NLP）领域引发了大量先进概念和应用。让我们深入探讨其中一些主题，包括不同的注意力变体、BERT（来自 Transformers 的双向编码器表示）和 GPT（生成式预训练 Transformer），以及它们的实际应用。</font>

**<font style="color:rgb(77, 77, 77);">不同的注意力变体</font>**

<font style="color:rgb(77, 77, 77);">注意力机制是 Transformer 模型的核心，使其能够专注于输入序列的相关部分。各种注意力变体的提议旨在增强 Transformer 的能力。</font>

1. <font style="color:rgba(0, 0, 0, 0.75);">缩放点积注意力：是原始 Transformer 模型中使用的标准注意力机制。它将查询和键向量的点积作为注意力分数，同时乘以维度的平方根进行缩放。</font>
2. <font style="color:rgba(0, 0, 0, 0.75);">多头注意力：注意力的强大扩展，利用多个注意力头同时捕捉输入序列的不同方面。每个头学习不同的注意力模式，使模型能够并行关注输入的各个部分。</font>
3. <font style="color:rgba(0, 0, 0, 0.75);">相对位置编码：引入相对位置编码以更有效地捕捉标记之间的相对位置关系。这种变体增强了模型理解标记之间顺序关系的能力。</font>

**<font style="color:rgb(77, 77, 77);">BERT（来自 Transformers 的双向编码器表示）</font>**

<font style="color:rgb(77, 77, 77);">BERT 是一个具有里程碑意义的基于 Transformer 的模型，在 NLP 领域产生了深远影响。它通过掩码语言建模和下一句预测等目标，在大规模文本语料库上进行预训练。BERT 学习了单词的深层上下文表示，捕捉双向上下文，使其在广泛的下游 NLP 任务中表现良好。</font>

<font style="color:rgb(77, 77, 77);">代码片段 - BERT 模型:</font>

```plain
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
print(outputs)


1
2
3
4
5
6
7
8
```

**<font style="color:rgb(77, 77, 77);">GPT（生成式预训练 Transformer）</font>**

<font style="color:rgb(77, 77, 77);">GPT 是一个基于 Transformer 的模型，以其生成能力而闻名。与双向的 BERT 不同，GPT 采用仅解码器的架构和自回归训练来生成连贯且上下文相关的文本。研究人员和开发人员已经成功地将 GPT 应用于各种任务，如文本完成、摘要、对话生成等。</font>

<font style="color:rgb(77, 77, 77);">代码片段 - GPT 模型:</font>

```plain
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_text = "Once upon a time, "
inputs=tokenizer(input_text,return_tensors='pt')
output=tokenizer.decode(
    model.generate(
        **inputs,
        max_new_tokens=100,
      )[0],
      skip_special_tokens=True
  )
input_ids = tokenizer(input_text, return_tensors='pt')

print(output)


```

### <font style="color:rgb(79, 79, 79);">八、总结</font>
<font style="color:rgb(77, 77, 77);">Transformer 通过其捕捉上下文和理解语言的能力，彻底改变了自然语言处理（NLP）领域。</font>

<font style="color:rgb(77, 77, 77);">通过注意力机制、编码器-解码器架构和多头注意力，它们使得诸如机器翻译和情感分析等任务得以在前所未有的规模上实现。随着我们继续探索诸如 BERT 和 GPT 等模型，很明显，Transformer 处于语言理解和生成的前沿。</font>

<font style="color:rgb(77, 77, 77);">它们对 NLP 的影响深远，而与 Transformer 一起的发现之旅将揭示出该领域更多令人瞩目的进展。</font>

**<font style="color:rgb(77, 77, 77);">研究论文</font>**

+ <font style="color:rgba(0, 0, 0, 0.75);">《Attention is All You Need》</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">《Language Models are Unsupervised Multitask Learners》</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">Attention in transformers, visually explained</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">Transformer Neural Networks, ChatGPT’s foundation</font>
