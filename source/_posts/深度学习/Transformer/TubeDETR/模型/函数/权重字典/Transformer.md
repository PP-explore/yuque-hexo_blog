---
title: Transformer
date: '2025-08-05 17:39:31'
updated: '2025-08-22 20:35:14'
categories:
  - 人工智能
tags:
  - 深度学习
  - TubeDETR
cover: /images/custom-cover.jpg
recommend: true
---
## transformer.py
#### **<font style="color:rgba(0, 0, 0, 0.9);">双流编码器-解码器结构</font>**
+ **<font style="color:rgba(0, 0, 0, 0.9);">Encoder</font>**<font style="color:rgba(0, 0, 0, 0.9);">：处理视觉特征（视频帧）和文本特征的联合编码</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">使用</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">TransformerEncoder</font>`<font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">堆叠多层（默认6层）</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">关键创新：将视频空间特征（</font>`<font style="color:rgba(0, 0, 0, 0.9);">src</font>`<font style="color:rgba(0, 0, 0, 0.9);">）和文本特征（</font>`<font style="color:rgba(0, 0, 0, 0.9);">text_memory_resized</font>`<font style="color:rgba(0, 0, 0, 0.9);">）在序列维度拼接</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">Decoder</font>**<font style="color:rgba(0, 0, 0, 0.9);">：基于编码结果进行跨模态注意力</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">使用</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">TransformerDecoder</font>`<font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">处理时空-文本联合特征</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">支持返回中间层结果（</font>`<font style="color:rgba(0, 0, 0, 0.9);">return_intermediate_dec</font>`<font style="color:rgba(0, 0, 0, 0.9);">）</font>

#### **<font style="color:rgba(0, 0, 0, 0.9);">多模态融合机制</font>**
```python
src = torch.cat([src, text_memory_resized], dim=0)  # 视觉+文本序列拼接
mask = torch.cat([mask, text_attention_mask], dim=1)  # 对应mask拼接
```

+ **<font style="color:rgba(0, 0, 0, 0.9);">视觉分支</font>**<font style="color:rgba(0, 0, 0, 0.9);">：视频帧通过CNN提取特征后展平为序列</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">文本分支</font>**<font style="color:rgba(0, 0, 0, 0.9);">：使用RoBERTa编码文本，通过</font>`<font style="color:rgba(0, 0, 0, 0.9);">FeatureResizer</font>`<font style="color:rgba(0, 0, 0, 0.9);">调整维度匹配视觉特征</font>

#### <font style="color:rgba(0, 0, 0, 0.9);"></font>**<font style="color:rgba(0, 0, 0, 0.9);">时空注意力设计</font>**
+ **<font style="color:rgba(0, 0, 0, 0.9);">时间编码</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>`<font style="color:rgba(0, 0, 0, 0.9);">TimeEmbedding</font>`<font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">处理视频时序信息（可学习或正弦位置编码）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">快速路径</font>**<font style="color:rgba(0, 0, 0, 0.9);">（Fast分支）：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">可选模式：</font>`<font style="color:rgba(0, 0, 0, 0.9);">gating</font>`<font style="color:rgba(0, 0, 0, 0.9);">（门控融合）、</font>`<font style="color:rgba(0, 0, 0, 0.9);">transformer</font>`<font style="color:rgba(0, 0, 0, 0.9);">（轻量时序编码）、</font>`<font style="color:rgba(0, 0, 0, 0.9);">pool</font>`<font style="color:rgba(0, 0, 0, 0.9);">（空间池化）</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">与主路径（Slow路径）特征融合增强时序建模</font>

### 参数学习
```plain

class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        video_max_len=0,
        stride=0,
        no_tsa=False,
        return_weights=False,
        fast=False,
        fast_mode="",
        learn_time_embed=False,
        rd_init_tsa=False,
        no_time_embed=False,
    ):
        """
        :param d_model: transformer embedding dimension   控制所有注意力层的向量维度
        :param nhead: transformer number of heads         将注意力分散到8个并行计算的头上
        :param num_encoder_layers: transformer encoder number of layers    
        :param num_decoder_layers: transformer decoder number of layers
        :param dim_feedforward: transformer dimension of feedforward
        :param dropout: transformer dropout
        :param activation: transformer activation
        :param return_intermediate_dec: whether to return intermediate outputs of the decoder
        :param pass_pos_and_query: if True tgt is initialized to 0 and position is added at every layer
        :param text_encoder_type: Hugging Face name for the text encoder
        :param freeze_text_encoder: whether to freeze text encoder weights
        :param video_max_len: maximum number of frames in the model
        :param stride: temporal stride k
        :param no_tsa: whether to use temporal self-attention
        :param return_weights: whether to return attention weights
        :param fast: whether to use the fast branch
        :param fast_mode: which variant of fast branch to use
        :param learn_time_embed: whether to learn time encodings
        :param rd_init_tsa: whether to randomly initialize temporal self-attention weights
        :param no_time_embed: whether to use time encodings
        """
        super().__init__()

```

##### <font style="color:rgba(0, 0, 0, 0.9);">基础Transformer配置</font>
| <font style="color:rgba(0, 0, 0, 0.9);">参数</font> | <font style="color:rgba(0, 0, 0, 0.9);">类型</font> | <font style="color:rgba(0, 0, 0, 0.9);">默认值</font> | <font style="color:rgba(0, 0, 0, 0.9);">作用原理</font> |
| :---: | :---: | :---: | :---: |
| `<font style="color:rgba(0, 0, 0, 0.9);">d_model</font>` | <font style="color:rgba(0, 0, 0, 0.9);">int</font> | <font style="color:rgba(0, 0, 0, 0.9);">512</font> | <font style="color:rgba(0, 0, 0, 0.9);">所有注意力层的向量维度，决定模型容量</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">nhead</font>` | <font style="color:rgba(0, 0, 0, 0.9);">int</font> | <font style="color:rgba(0, 0, 0, 0.9);">8</font> | <font style="color:rgba(0, 0, 0, 0.9);">多头注意力的头数，并行计算注意力的分支</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">num_encoder_layers</font>` | <font style="color:rgba(0, 0, 0, 0.9);">int</font> | <font style="color:rgba(0, 0, 0, 0.9);">6</font> | <font style="color:rgba(0, 0, 0, 0.9);">编码器堆叠的Transformer层数</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">num_decoder_layers</font>` | <font style="color:rgba(0, 0, 0, 0.9);">int</font> | <font style="color:rgba(0, 0, 0, 0.9);">6</font> | <font style="color:rgba(0, 0, 0, 0.9);">解码器堆叠的Transformer层数</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">dim_feedforward</font>` | <font style="color:rgba(0, 0, 0, 0.9);">int</font> | <font style="color:rgba(0, 0, 0, 0.9);">2048</font> | <font style="color:rgba(0, 0, 0, 0.9);">FFN层的隐藏维度（通常为4*d_model）</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">dropout</font>` | <font style="color:rgba(0, 0, 0, 0.9);">float</font> | <font style="color:rgba(0, 0, 0, 0.9);">0.1</font> | <font style="color:rgba(0, 0, 0, 0.9);">随机失活比例，防止过拟合</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">activation</font>` | <font style="color:rgba(0, 0, 0, 0.9);">str</font> | <font style="color:rgba(0, 0, 0, 0.9);">"relu"</font> | <font style="color:rgba(0, 0, 0, 0.9);">FFN层的激活函数</font> |


##### <font style="color:rgba(0, 0, 0, 0.9);">数学关系：</font>
+ <font style="color:rgba(0, 0, 0, 0.9);">每个注意力头的维度 =</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">d_model // nhead</font>`
+ <font style="color:rgba(0, 0, 0, 0.9);">FFN层计算：</font>`<font style="color:rgba(0, 0, 0, 0.9);">Linear(d_model→dim_feedforward)→ReLU→Linear(dim_feedforward→d_model)</font>`

##### <font style="color:rgba(0, 0, 0, 0.9);">输出控制参数</font>
| <font style="color:rgba(0, 0, 0, 0.9);">参数</font> | <font style="color:rgba(0, 0, 0, 0.9);">作用原理</font> | <font style="color:rgba(0, 0, 0, 0.9);">应用场景</font> |
| :---: | :---: | :---: |
| `<font style="color:rgba(0, 0, 0, 0.9);">return_intermediate_dec</font>` | <font style="color:rgba(0, 0, 0, 0.9);">返回所有解码层输出而非最后一层</font> | <font style="color:rgba(0, 0, 0, 0.9);">深度监督训练</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">return_weights</font>` | <font style="color:rgba(0, 0, 0, 0.9);">返回注意力权重矩阵</font> | <font style="color:rgba(0, 0, 0, 0.9);">可视化/注意力引导损失</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">pass_pos_and_query</font>` | <font style="color:rgba(0, 0, 0, 0.9);">每层都添加位置编码到查询向量</font> | <font style="color:rgba(0, 0, 0, 0.9);">增强空间感知</font> |


##### <font style="color:rgba(0, 0, 0, 0.9);">视频处理专用参数</font>
| <font style="color:rgba(0, 0, 0, 0.9);">参数</font> | <font style="color:rgba(0, 0, 0, 0.9);">功能说明</font> | <font style="color:rgba(0, 0, 0, 0.9);">技术实现</font> |
| :---: | :---: | :---: |
| `<font style="color:rgba(0, 0, 0, 0.9);">video_max_len</font>` | <font style="color:rgba(0, 0, 0, 0.9);">最大视频帧数</font> | <font style="color:rgba(0, 0, 0, 0.9);">限制时序位置编码长度</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">stride</font>` | <font style="color:rgba(0, 0, 0, 0.9);">时间维度下采样率</font> | <font style="color:rgba(0, 0, 0, 0.9);">控制时序注意力计算密度</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">no_tsa</font>` | <font style="color:rgba(0, 0, 0, 0.9);">禁用时序自注意力</font> | <font style="color:rgba(0, 0, 0, 0.9);">简化视频处理</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">learn_time_embed</font>` | <font style="color:rgba(0, 0, 0, 0.9);">可学习的时间编码</font> | <font style="color:rgba(0, 0, 0, 0.9);">替代固定正弦编码</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">rd_init_tsa</font>` | <font style="color:rgba(0, 0, 0, 0.9);">随机初始化时序注意力</font> | <font style="color:rgba(0, 0, 0, 0.9);">打破时序对称性</font> |


##### <font style="color:rgba(0, 0, 0, 0.9);">文本编码器配置</font>
| <font style="color:rgba(0, 0, 0, 0.9);">参数</font> | <font style="color:rgba(0, 0, 0, 0.9);">作用</font> | <font style="color:rgba(0, 0, 0, 0.9);">实现细节</font> |
| :---: | :---: | :---: |
| `<font style="color:rgba(0, 0, 0, 0.9);">text_encoder_type</font>` | <font style="color:rgba(0, 0, 0, 0.9);">预训练文本模型类型</font> | <font style="color:rgba(0, 0, 0, 0.9);">如"roberta-base"</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">freeze_text_encoder</font>` | <font style="color:rgba(0, 0, 0, 0.9);">冻结文本编码器参数</font> | <font style="color:rgba(0, 0, 0, 0.9);">迁移学习时常用</font> |


##### <font style="color:rgba(0, 0, 0, 0.9);">快速模式参数</font>
| <font style="color:rgba(0, 0, 0, 0.9);">参数</font> | <font style="color:rgba(0, 0, 0, 0.9);">功能</font> | <font style="color:rgba(0, 0, 0, 0.9);">优化策略</font> |
| :---: | :---: | :---: |
| `<font style="color:rgba(0, 0, 0, 0.9);">fast</font>` | <font style="color:rgba(0, 0, 0, 0.9);">启用快速推理模式</font> | <font style="color:rgba(0, 0, 0, 0.9);">牺牲精度换速度</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">fast_mode</font>` | <font style="color:rgba(0, 0, 0, 0.9);">快速模式类型</font> | <font style="color:rgba(0, 0, 0, 0.9);">"gating"或"transformer"</font> |


### __init__函数:
```python
def __init__(
    self,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation="relu",
    return_intermediate_dec=False,
    pass_pos_and_query=True,
    text_encoder_type="roberta-base",
    freeze_text_encoder=False,
    video_max_len=0,
    stride=0,
    no_tsa=False,
    return_weights=False,
    fast=False,
    fast_mode="",
    learn_time_embed=False,
    rd_init_tsa=False,
    no_time_embed=False,
):
    super().__init__()

    self.pass_pos_and_query = pass_pos_and_query
    encoder_layer = TransformerEncoderLayer(   #编码器的单个基础层
        d_model, nhead, dim_feedforward, dropout, activation
    )
    encoder_norm = None  # 最终层归一化（通常为LayerNorm）
    self.encoder = TransformerEncoder(
        encoder_layer, num_encoder_layers, encoder_norm, return_weights=True
    )

    decoder_layer = TransformerDecoderLayer(   #解码器单层
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        no_tsa=no_tsa,
    )
    decoder_norm = nn.LayerNorm(d_model)
    self.decoder = TransformerDecoder(    ## 多层堆叠的解码层
        decoder_layer,
        num_decoder_layers,
        decoder_norm,
        return_intermediate=return_intermediate_dec,
        return_weights=return_weights,
    )

    self._reset_parameters()

    self.return_weights = return_weights

    self.learn_time_embed = learn_time_embed
    self.use_time_embed = not no_time_embed
    if self.use_time_embed:
        if learn_time_embed:
            self.time_embed = TimeEmbeddingLearned(video_max_len, d_model)
        else:
            self.time_embed = TimeEmbeddingSine(video_max_len, d_model)

        self.fast = fast
    self.fast_mode = fast_mode
    if fast:
        if fast_mode == "gating":
            self.fast_encoder = nn.Linear(d_model, d_model)
        elif fast_mode == "transformer":
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation
            )
            self.fast_encoder = TransformerEncoder(
                encoder_layer, 1, nn.LayerNorm(d_model), return_weights=True
            )
            self.fast_residual = nn.Linear(d_model, d_model)
        else:
            self.fast_encoder = nn.Linear(d_model, d_model)
            self.fast_residual = nn.Linear(d_model, d_model)

        self.rd_init_tsa = rd_init_tsa
    self._reset_temporal_parameters()

    self.tokenizer = RobertaTokenizerFast.from_pretrained(
        text_encoder_type, local_files_only=False
    )
    self.text_encoder = RobertaModel.from_pretrained(
        text_encoder_type, local_files_only=False
    )

    if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.video_max_len = video_max_len
        self.stride = stride

    def _reset_parameters(self):
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  #对所有权重矩阵执行 ​Xavier均匀分布初始化​（又称Glorot初始化）

    def _reset_temporal_parameters(self):
        for n, p in self.named_parameters():
            if "fast_encoder" in n and self.fast_mode == "transformer":
                if "norm" in n and "weight" in n:
                    nn.init.constant_(p, 1.0)
                elif "norm" in n and "bias" in n:
                    nn.init.constant_(p, 0)
                else:
                    nn.init.constant_(p, 0)

            if self.rd_init_tsa and "decoder" in n and "self_attn" in n:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            if "fast_residual" in n:
                nn.init.constant_(p, 0)
            if self.fast_mode == "gating" and "fast_encoder" in n:
                nn.init.constant_(p, 0)
```

#### 参数初始化
```plain

Transformer模型初始化过程中最关键的一步参数初始化操作，它的设计直接影响模型训练的稳定性和收敛速度。

    def _reset_parameters(self):
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  #对所有权重矩阵执行 ​Xavier均匀分布初始化​（又称Glorot初始化）
```

##### <font style="color:rgba(0, 0, 0, 0.9);">数学本质</font>
<font style="color:rgba(0, 0, 0, 0.9);">对所有权重矩阵执行 </font>**<font style="color:rgba(0, 0, 0, 0.9);">Xavier均匀分布初始化</font>**<font style="color:rgba(0, 0, 0, 0.9);">（又称Glorot初始化），公式为：</font>

![](/images/ff11b5f798d19fc1cc564281f5b3ec1f.png)

<font style="color:rgba(0, 0, 0, 0.9);">其中</font>_<font style="color:rgba(0, 0, 0, 0.9);">nin</font>_<font style="color:rgba(0, 0, 0, 0.9);">和</font>_<font style="color:rgba(0, 0, 0, 0.9);">nout</font>_<font style="color:rgba(0, 0, 0, 0.9);">分别是该层的输入/输出维度。</font>

#### <font style="color:rgba(0, 0, 0, 0.9);">时序编码和可学习的时序编码:</font>
```python
        if self.use_time_embed:
            if learn_time_embed:
                self.time_embed = TimeEmbeddingLearned(video_max_len, d_model)
            else:
                self.time_embed = TimeEmbeddingSine(video_max_len, d_model)
class TimeEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=200, d_model=512):
        super().__init__()
        self.time_embed = nn.Embedding(num_pos_feats, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.time_embed.weight)

    def forward(self, ln):
        return self.time_embed.weight[:ln].unsqueeze(1)


class TimeEmbeddingSine(nn.Module):
    """
    Same as below for temporal dimension
    """

    def __init__(self, max_len=200, d_model=512):
        super().__init__()
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        te = torch.zeros(max_len, 1, d_model)
        te[:, 0, 0::2] = torch.sin(position * div_term)
        te[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("te", te)

    def forward(self, ln):
        pos_t = self.te[:ln]
        return pos_t
```

##### <font style="color:rgba(0, 0, 0, 0.9);">*</font>`<font style="color:rgba(0, 0, 0, 0.9);">ln</font>`<font style="color:rgba(0, 0, 0, 0.9);">的来源**</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">训练时</font>**<font style="color:rgba(0, 0, 0, 0.9);">：从数据标注中获取真实帧数（论文3.4节）</font>

```plain
ln = durations[i]  # 第i个视频的实际长度
```

+ **<font style="color:rgba(0, 0, 0, 0.9);">推理时</font>**<font style="color:rgba(0, 0, 0, 0.9);">：通过时序预测头得到（论文3.3节）</font>

```plain
ln = (t_end - t_start)  # 预测的时间跨度
```

#### <font style="color:rgba(0, 0, 0, 0.9);">文本编码器初始化</font>
##### <font style="color:rgba(0, 0, 0, 0.9);">1. Tokenizer 加载</font>
```plain
self.tokenizer = RobertaTokenizerFast.from_pretrained(
    "roberta-base",  # 默认使用RoBERTa-base
    local_files_only=False  # 允许在线下载
)
```

+ **<font style="color:rgba(0, 0, 0, 0.9);">作用</font>**<font style="color:rgba(0, 0, 0, 0.9);">：将原始文本转换为模型可处理的token ID</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">关键特性</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">支持510 token长度（RoBERTa限制）</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">自动添加[CLS]/[SEP]等特殊token</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">示例输出：</font>

```plain
{"input_ids": [0, 314, 112, ..., 2], "attention_mask": [1,1,...,0]}
```

##### <font style="color:rgba(0, 0, 0, 0.9);">2. 预训练模型加载</font>
```plain
self.text_encoder = RobertaModel.from_pretrained(
    "roberta-base",
    local_files_only=False
)
```

+ **<font style="color:rgba(0, 0, 0, 0.9);">输出维度</font>**<font style="color:rgba(0, 0, 0, 0.9);">：768维（RoBERTa-base的hidden_size）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">层结构</font>**<font style="color:rgba(0, 0, 0, 0.9);">：12层Transformer编码器</font>

##### <font style="color:rgba(0, 0, 0, 0.9);">参数冻结控制</font>
```plain
if freeze_text_encoder:
    for p in self.text_encoder.parameters():
        p.requires_grad_(False)  # 冻结所有参数
```

+ **<font style="color:rgba(0, 0, 0, 0.9);">设计动机</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">保留预训练语言知识（论文4.1节）</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">防止小数据集上的过拟合</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">实验对比</font>**<font style="color:rgba(0, 0, 0, 0.9);">（论文表3）</font>

#### <font style="color:rgba(0, 0, 0, 0.9);">特征维度适配</font>
##### <font style="color:rgba(0, 0, 0, 0.9);">1. 维度匹配问题</font>
+ <font style="color:rgba(0, 0, 0, 0.9);">文本编码器输出：768维</font>
+ <font style="color:rgba(0, 0, 0, 0.9);">视觉编码器维度：512维（d_model）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">解决方案</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>`<font style="color:rgba(0, 0, 0, 0.9);">FeatureResizer</font>`<font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">投影层</font>

##### <font style="color:rgba(0, 0, 0, 0.9);">2. 特征适配器实现</font>
```plain
self.resizer = FeatureResizer(
    input_feat_size=768,  # RoBERTa输出维度
    output_feat_size=512, # 目标维度
    dropout=0.1          # 防止过拟合
)
```

+ **<font style="color:rgba(0, 0, 0, 0.9);">内部结构</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>

```plain
Sequential(
    Linear(768, 512),
    LayerNorm(512),
    Dropout(0.1),
    ReLU()
)
```

#### <font style="color:rgba(0, 0, 0, 0.9);">关键设计思想</font>
##### <font style="color:rgba(0, 0, 0, 0.9);">1. 梯度流控制</font>
| <font style="color:rgba(0, 0, 0, 0.9);">模块</font> | <font style="color:rgba(0, 0, 0, 0.9);">梯度状态</font> | <font style="color:rgba(0, 0, 0, 0.9);">更新策略</font> |
| :---: | :---: | :---: |
| <font style="color:rgba(0, 0, 0, 0.9);">Tokenizer</font> | <font style="color:rgba(0, 0, 0, 0.9);">❌</font><font style="color:rgba(0, 0, 0, 0.9);"> 固定</font> | <font style="color:rgba(0, 0, 0, 0.9);">-</font> |
| <font style="color:rgba(0, 0, 0, 0.9);">Text Encoder</font> | <font style="color:rgba(0, 0, 0, 0.9);">根据freeze_text_encoder</font> | <font style="color:rgba(0, 0, 0, 0.9);">冻结/微调</font> |
| <font style="color:rgba(0, 0, 0, 0.9);">FeatureResizer</font> | <font style="color:rgba(0, 0, 0, 0.9);">✅</font><font style="color:rgba(0, 0, 0, 0.9);"> 可训练</font> | <font style="color:rgba(0, 0, 0, 0.9);">随机初始化</font> |


##### <font style="color:rgba(0, 0, 0, 0.9);">2. 跨模态对齐</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">维度统一</font>**<font style="color:rgba(0, 0, 0, 0.9);">：将文本特征投影到与视觉特征相同的空间</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">共享d_model</font>**<font style="color:rgba(0, 0, 0, 0.9);">：使注意力机制能直接处理多模态特征</font>

##### <font style="color:rgba(0, 0, 0, 0.9);">3. 正则化策略</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">Dropout</font>**<font style="color:rgba(0, 0, 0, 0.9);">：在特征适配器中添加0.1的dropout</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">LayerNorm</font>**<font style="color:rgba(0, 0, 0, 0.9);">：稳定训练过程</font>

### forward函数
```python

    def forward(
        self,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        text=None,
        encode_and_save=True,
        durations=None,
        tpad_mask_t=None,
        fast_src=None,
        img_memory=None,
        query_mask=None,
        text_memory=None,
        text_mask=None,
        memory_mask=None,
    ):
        if encode_and_save:
            # flatten n_clipsxCxHxW to HWxn_clipsxC   tot_clips 是总剪辑数（即时间步数）
            tot_clips, c, h, w = src.shape
            device = src.device

            # nb of times object queries are repeated   durations提供了每个视频的实际长度（帧数
            if durations is not None:
                t = max(durations)      #t取durations中的最大值
                b = len(durations)      #b为batch size（即视频个数）
                bs_oq = tot_clips if (not self.stride) else b * t   #根据是否使用stride（时间步长）来决定bs_oq（object queries的batch大小）果不使用stride，则bs_oq为tot_clips（即总剪辑数）；否则为b*t（即batch_size * 最大帧数）。
            else:
                bs_oq = tot_clips

            src = src.flatten(2).permute(2, 0, 1)       #输出：[H*W, tot_clips, C]
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  #位置编码同等操作
            mask = mask.flatten(1)                          #  [tot_clips, H, W] → [tot_clips, H*W]
            query_embed = query_embed.unsqueeze(1).repeat(
                1, bs_oq, 1
            )  # n_queriesx(b*t)xf

            n_queries, _, f = query_embed.shape
            query_embed = query_embed.view(
                n_queries * t,
                b,
                f,
            )
            if self.use_time_embed:  # add temporal encoding to init time queries
                time_embed = self.time_embed(t).repeat(n_queries, b, 1)
                query_embed = query_embed + time_embed

            # prepare time queries mask
            query_mask = None
            if self.stride:
                query_mask = (
                    torch.ones(
                        b,
                        n_queries * t,
                    )
                    .bool()
                    .to(device)
                )
                query_mask[:, 0] = False  # avoid empty masks
                for i_dur, dur in enumerate(durations):
                    query_mask[i_dur, : (dur * n_queries)] = False

            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = (
                    src + 0.1 * pos_embed,
                    query_embed,
                    None,
                    None,
                )

            if isinstance(text[0], str):
                # Encode the text
                tokenized = self.tokenizer.batch_encode_plus(
                    text, padding="longest", return_tensors="pt"
                ).to(device)
                encoded_text = self.text_encoder(**tokenized)

                # Transpose memory because pytorch's attention expects sequence first
                text_memory = encoded_text.last_hidden_state.transpose(0, 1)
                # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                text_attention_mask = tokenized.attention_mask.ne(1).bool()

                # Resize the encoder hidden states to be of the same d_model as the decoder
                text_memory_resized = self.resizer(text_memory)
            else:
                # The text is already encoded, use as is.
                text_attention_mask, text_memory_resized, tokenized = text

            # encode caption once per video and repeat each caption X times where X is the number of clips in the video
            n_repeat = t if (not self.stride) else math.ceil(t / self.stride)
            assert (
                n_repeat
                == src.shape[1] // text_memory_resized.shape[1]
                == mask.shape[0] // text_attention_mask.shape[0]
            )
            tokenized._encodings = [
                elt for elt in tokenized._encodings for _ in range(n_repeat)
            ]  # repeat batchencodings output (BT)
            text_attention_mask_orig = text_attention_mask
            text_attention_mask = torch.stack(
                [
                    text_attention_mask[i_elt]
                    for i_elt in range(len(text_attention_mask))
                    for _ in range(n_repeat)
                ]
            )
            text_memory_resized_orig = text_memory_resized
            text_memory_resized = torch.stack(
                [
                    text_memory_resized[:, i_elt]
                    for i_elt in range(text_memory_resized.size(1))
                    for _ in range(n_repeat)
                ],
                1,
            )
            tokenized["input_ids"] = torch.stack(
                [
                    tokenized["input_ids"][i_elt]
                    for i_elt in range(len(tokenized["input_ids"]))
                    for _ in range(n_repeat)
                ]
            )
            tokenized["attention_mask"] = torch.stack(
                [
                    tokenized["attention_mask"][i_elt]
                    for i_elt in range(len(tokenized["attention_mask"]))
                    for _ in range(n_repeat)
                ]
            )

            # Concat on the sequence dimension
            src = torch.cat([src, text_memory_resized], dim=0)

            # Concat mask for all frames, will be used for the decoding
            if tpad_mask_t is not None:
                tpad_mask_t_orig = tpad_mask_t
                tpad_mask_t = tpad_mask_t.flatten(1)  # bxtxhxw -> bx(txhxw)
                text_attn_mask_t = torch.stack(
                    [
                        text_attention_mask_orig[i_elt]
                        for i_elt in range(len(text_attention_mask_orig))
                        for _ in range(max(durations))
                    ]
                )
                tpad_mask_t = torch.cat([tpad_mask_t, text_attn_mask_t], dim=1)

            # For mask, sequence dimension is second
            mask = torch.cat([mask, text_attention_mask], dim=1)
            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            pos_embed = torch.cat(
                [pos_embed, torch.zeros_like(text_memory_resized)], dim=0
            )

            if (
                self.fast_mode == "noslow"
            ):  # no space-text attention for noslow baseline
                img_memory, weights = src, None
                text_memory = torch.stack(
                    [
                        text_memory_resized_orig[:, i_elt]
                        for i_elt in range(text_memory_resized_orig.size(1))
                        for _ in range(t)
                    ],
                    1,
                )
            else:  # space-text attention
                img_memory, weights = self.encoder(
                    src, src_key_padding_mask=mask, pos=pos_embed, mask=None
                )
                text_memory = img_memory[-len(text_memory_resized) :]

            if self.fast:
                if (
                    self.fast_mode == "transformer"
                ):  # temporal transformer in the fast branch for this variant
                    fast_src2 = (
                        fast_src.flatten(2)
                        .view(b, t, f, h * w)
                        .permute(1, 0, 3, 2)
                        .flatten(1, 2)
                    )  # (b*t)xfxhxw -> (b*t)xfx(h*w) -> bxtxfx(h*w) -> txbx(h*w)xf -> tx(b*h*w)xf
                    time_embed = self.time_embed(t)
                    time_embed = time_embed.repeat(1, b * h * w, 1)
                    fast_memory, fast_weights = self.fast_encoder(
                        fast_src2, pos=time_embed
                    )
                    fast_memory = (
                        fast_memory.view(t, b, h * w, f)
                        .transpose(0, 1)
                        .view(b * t, h * w, f)
                        .transpose(0, 1)
                    )  # tx(b*h*w)xf -> txbx(h*w)xf -> bxtx(h*w)xf -> (b*t)x(h*w)xf -> (h*w)x(b*t)xf
                else:
                    fast_src2 = fast_src.flatten(2).permute(
                        2, 0, 1
                    )  # (b*t)xfxhxw -> (b*t)xfx(h*w) -> (h*w)x(b*t)xf
                    if (
                        self.fast_mode == "pool"
                    ):  # spatial pool in the fast branch for this baseline
                        fast_mask = tpad_mask_t_orig.flatten(1).transpose(
                            0, 1
                        )  # (h*w)x(b*t)
                        fast_pool_mask = ~fast_mask[:, :, None]
                        sum_mask = fast_pool_mask.float().sum(dim=0).clamp(min=1)
                        fast_src2 = fast_src2 * fast_pool_mask
                        n_visual_tokens = len(fast_src2)
                        fast_src2 = torch.div(fast_src2.sum(dim=0), sum_mask)
                    fast_memory = self.fast_encoder(fast_src2)
                    if self.fast_mode == "pool":
                        fast_memory = fast_memory.unsqueeze(0).repeat(
                            n_visual_tokens, 1, 1
                        )

            if self.stride:  # temporal replication
                device = img_memory.device
                n_tokens, tot_clips, f = img_memory.shape
                pad_img_memory = torch.zeros(n_tokens, b, t, f).to(device)
                pad_pos_embed = torch.zeros(n_tokens, b, t, f).to(device)
                cur_clip = 0
                n_clips = math.ceil(t / self.stride)
                for i_dur, dur in enumerate(durations):
                    for i_clip in range(n_clips):
                        clip_dur = min(self.stride, t - i_clip * self.stride)
                        pad_img_memory[
                            :,
                            i_dur,
                            i_clip * self.stride : i_clip * self.stride + clip_dur,
                        ] = (
                            img_memory[:, cur_clip].unsqueeze(1).repeat(1, clip_dur, 1)
                        )
                        pad_pos_embed[
                            :,
                            i_dur,
                            i_clip * self.stride : i_clip * self.stride + clip_dur,
                        ] = (
                            pos_embed[:, cur_clip].unsqueeze(1).repeat(1, clip_dur, 1)
                        )
                        cur_clip += 1
                img_memory = pad_img_memory.view(
                    n_tokens, b * t, f
                )  # n_tokensxbxtxf -> n_tokensx(b*t)xf
                mask = tpad_mask_t.view(
                    b * t, n_tokens
                )  # bxtxn_tokens -> (b*t)xn_tokens
                mask[:, 0] = False  # avoid empty masks
                pos_embed = pad_pos_embed.view(
                    n_tokens, b * t, f
                )  # n_tokensxbxtxf -> n_tokensx(b*t)xf

                if self.fast:  # aggregate slow and fast features
                    n_visual_tokens = len(fast_memory)
                    if self.fast_mode == "noslow":
                        img_memory = torch.cat([fast_memory, text_memory], 0)
                    elif self.fast_mode == "gating":
                        img_memory2 = img_memory[
                            :n_visual_tokens
                        ].clone() * torch.sigmoid(fast_memory)
                        img_memory[:n_visual_tokens] = (
                            img_memory[:n_visual_tokens] + img_memory2
                        )
                    else:
                        img_memory2 = img_memory[:n_visual_tokens] + fast_memory
                        img_memory2 = self.fast_residual(img_memory2)
                        img_memory[:n_visual_tokens] = (
                            img_memory[:n_visual_tokens] + img_memory2
                        )
                text_memory = img_memory[-len(text_memory_resized) :]

            memory_cache = {
                "text_memory_resized": text_memory_resized,  # seq first
                "text_memory": text_memory,  # seq first
                "text_attention_mask": text_attention_mask,  # batch first
                "tokenized": tokenized,  # batch first
                "img_memory": img_memory,  # seq first
                "mask": mask,  # batch first
                "pos_embed": pos_embed,  # seq first
                "query_embed": query_embed,  # seq first
                "query_mask": query_mask,  # batch first
            }

            return memory_cache

        else:
            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = (
                    src + 0.1 * pos_embed,
                    query_embed,
                    None,
                    None,
                )

            # time-space-text attention
            hs = self.decoder(
                tgt,  # n_queriesx(b*t)xF
                img_memory,  # ntokensx(b*t)x256
                memory_key_padding_mask=mask,  # (b*t)xn_tokens
                pos=pos_embed,  # n_tokensx(b*t)xF
                query_pos=query_embed,  # n_queriesx(b*t)xF
                tgt_key_padding_mask=query_mask,  # bx(t*n_queries)
                text_memory=text_memory,
                text_memory_mask=text_mask,
                memory_mask=memory_mask,
            )  # n_layersxn_queriesx(b*t)xF
            if self.return_weights:
                hs, weights, cross_weights = hs

            if not self.return_weights:
                return hs.transpose(1, 2)
            else:
                return hs.transpose(1, 2), weights, cross_weights

```



### 查询向量初始化
#### 1. **基础查询扩展**
```plain
query_embed = query_embed.unsqueeze(1).repeat(1, bs_oq, 1)  # [n_queries, b*t, f]
```

+ **输入形状**：`[n_queries, f]` （可学习的查询向量）
+ **扩展逻辑**：
    - `unsqueeze(1)`：增加中间维度 → `[n_queries, 1, f]`
    - `repeat(1, bs_oq, 1)`：沿时间维度复制  
`bs_oq = batch_size * t`（时间步数）
+ **物理意义**：为每个时间步创建独立的查询向量

#### <font style="color:rgb(0, 0, 0);">2. </font>**<font style="color:rgb(0, 0, 0);">时间维度重组</font>**
```python
query_embed = query_embed.view(n_queries * t, b, f)  # [n_queries*t, b, f]
```

+ **<font style="color:rgb(0, 0, 0);">维度变换</font>**<font style="color:rgb(0, 0, 0);">：  
</font>`<font style="color:rgb(0, 0, 0);">[n_queries, b*t, f] → [n_queries*t, b, f]</font>`
+ **<font style="color:rgb(0, 0, 0);">目的</font>**<font style="color:rgb(0, 0, 0);">：  
</font><font style="color:rgb(0, 0, 0);">将时间维度从"b*t"平铺形式改为显式分离，便于后续处理</font>

#### 3. **时间编码注入**
```python
if self.use_time_embed:
    time_embed = self.time_embed(t).repeat(n_queries, b, 1)
    query_embed = query_embed + time_embed
```

+ **时间编码生成**：
    - `self.time_embed(t)`：生成`[t, 1, f]`形状的时间编码
    - `.repeat(n_queries, b, 1)`：扩展为`[n_queries*t, b, f]`
+ **融合方式**：元素级相加，使查询向量具有时序感知能力

### TransformerEncoderLayer():
```python
TransformerEncoderLayer 类定义了 ​Transformer 编码器的单个基础层，它封装了编码器的核心计算逻辑
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        #多头注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model 前馈网络(FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)  #获得激活函数

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos    # 加法融合位置信息

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # # 对Q/K注入位置信息
        q = k = self.with_pos_embed(src, pos) 
        # 注意此处value用原始src
        src2, weights = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        ) # 多头注意力
        
        # Post-Norm处理：
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)            # 后置归一化
        # FFN处理：
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)     # 后置归一化
        return src, weights
```

Post-Norm vs Pre-Norm

1. Post-Norm（原始Transformer方案）

```plain
# TransformerEncoderLayer中的实现：
src = src + self.dropout1(src2)  # 残差连接
src = self.norm1(src)            # 后置归一化
```

+ **执行顺序**：`Attention → Dropout → Add → LayerNorm`
+ **数学表达**：xl+1=LayerNorm(xl+Attention(xl))

2. Pre-Norm（现代变体）

```plain
# 常见实现方式：
src2 = self.norm1(src)  # 先归一化
src = src + self.dropout1(self.attention(src2))
```

+ **优势**：更深的网络也能稳定训练（如BERT/GPT）

3. 对比实验数据

| 类型 | 训练稳定性 | 适合深度 | 典型应用 |
| :---: | :---: | :---: | :---: |
| Post-Norm | 较低 | ≤12层 | 原始Transformer |
| Pre-Norm | 高 | 100+层 | BERT、GPT系列 |




### class TransformerDecoderLayer(nn.Module):
```python

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model,                 # 统一特征维度（与Encoder一致）  512
        nhead,                   # 多头注意力头数  8
        dim_feedforward=2048,    # FFN隐藏层维度（通常4倍d_model） 
        dropout=0.1,
        activation="relu",       # FFN激活函数
        no_tsa=False,            # 是否禁用时序自注意力（Temporal Self-Attention）
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)   # 时序自注意力（Temporal Self-Attention）
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)   # 时间对齐交叉注意力（Time-Aligned Cross-Attention） 

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.no_tsa = no_tsa   #no_tsa	表1对比实验	禁用时序建模时退化为空间解码器

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,  #可学习的时间查询向量 time queries 
        memory,  ## 编码器输出的视频-文本特征
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,   # 固定编码
        text_memory=None,   #图3的h_s(v,s)	文本上下文特征
        text_memory_mask=None,
    ):

        q = k = self.with_pos_embed(tgt, query_pos)

        # Temporal Self attention
        if self.no_tsa:
            t, b, _ = tgt.shape
            n_tokens, bs, f = memory.shape
            tgt2, weights = self.self_attn(
                q.transpose(0, 1).reshape(bs * b, -1, f).transpose(0, 1),
                k.transpose(0, 1).reshape(bs * b, -1, f).transpose(0, 1),
                value=tgt.transpose(0, 1).reshape(bs * b, -1, f).transpose(0, 1),
                attn_mask=tgt_mask,
                key_padding_mask=None,
            )
            tgt2 = tgt2.view(b, t, f).transpose(0, 1)
        else:
            tgt2, weights = self.self_attn(
                q,
                k,
                value=tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Time Aligned Cross attention
        t, b, _ = tgt.shape
        n_tokens, bs, f = memory.shape
        tgt_cross = (
            tgt.transpose(0, 1).reshape(bs, -1, f).transpose(0, 1)
        )  # txbxf -> bxtxf -> (b*t)x1xf -> 1x(b*t)xf
        query_pos_cross = (
            query_pos.transpose(0, 1).reshape(bs, -1, f).transpose(0, 1)
        )  # txbxf -> bxtxf -> (b*t)x1xf -> 1x(b*t)xf

        tgt2, cross_weights = self.cross_attn_image(
            query=self.with_pos_embed(tgt_cross, query_pos_cross),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )

        tgt2 = tgt2.view(b, t, f).transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, weights, cross_weights

```

#### **<font style="color:rgba(0, 0, 0, 0.9);">tgt（目标序列）</font>**
+ **<font style="color:rgba(0, 0, 0, 0.9);">定义</font>**<font style="color:rgba(0, 0, 0, 0.9);">：可学习的时间查询向量（time queries）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">初始化</font>**<font style="color:rgba(0, 0, 0, 0.9);">：零张量（见论文3.3节"tgt is initialized to 0"）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">作用</font>**<font style="color:rgba(0, 0, 0, 0.9);">：承载解码过程中的时空状态表示</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">维度</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>`<font style="color:rgba(0, 0, 0, 0.9);">[t, b, d_model]</font>`
    - `<font style="color:rgba(0, 0, 0, 0.9);">t</font>`<font style="color:rgba(0, 0, 0, 0.9);">：时间步数（如200帧）</font>
    - `<font style="color:rgba(0, 0, 0, 0.9);">b</font>`<font style="color:rgba(0, 0, 0, 0.9);">：批次大小（如32个视频）</font>
    - `<font style="color:rgba(0, 0, 0, 0.9);">d_model</font>`<font style="color:rgba(0, 0, 0, 0.9);">：特征维度（如512）</font>

#### **<font style="color:rgba(0, 0, 0, 0.9);">query_pos（位置编码）</font>**
+ **<font style="color:rgba(0, 0, 0, 0.9);">组成</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>

```plain
# 论文3.3节实现
query_pos = learned_object_embed + sin_time_embed(t)
```

    - `<font style="color:rgba(0, 0, 0, 0.9);">learned_object_embed</font>`<font style="color:rgba(0, 0, 0, 0.9);">：可学习的对象编码（跨帧身份标识）</font>
    - `<font style="color:rgba(0, 0, 0, 0.9);">sin_time_embed</font>`<font style="color:rgba(0, 0, 0, 0.9);">：正弦时间位置编码（标记绝对时序）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">特性</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">与tgt同维度</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">[t, b, d_model]</font>`
    - <font style="color:rgba(0, 0, 0, 0.9);">梯度计算冻结（不参与反向传播）</font>

#### <font style="color:rgba(0, 0, 0, 0.9);">维度变换代码</font>
```python
tgt_cross = (
            tgt.transpose(0, 1).reshape(bs, -1, f).transpose(0, 1)
        )  # txbxf -> bxtxf -> (b*t)x1xf -> 1x(b*t)xf
        query_pos_cross = (
            query_pos.transpose(0, 1).reshape(bs, -1, f).transpose(0, 1)
        )  # txbxf -> bxtxf -> (b*t)x1xf -> 1x(b*t)xf
```

##### <font style="color:rgba(0, 0, 0, 0.9);">核心处理目标</font>
<font style="color:rgba(0, 0, 0, 0.9);">这段代码实现论文3.3节提出的 </font>**<font style="color:rgba(0, 0, 0, 0.9);">"时间对齐的交叉注意力"（Time-Aligned Cross-Attention）</font>**<font style="color:rgba(0, 0, 0, 0.9);">，其核心目的是：</font>

1. **<font style="color:rgba(0, 0, 0, 0.9);">强制时间对齐</font>**<font style="color:rgba(0, 0, 0, 0.9);">：确保每个时间查询只关注对应帧的视觉特征</font>
2. **<font style="color:rgba(0, 0, 0, 0.9);">批量计算优化</font>**<font style="color:rgba(0, 0, 0, 0.9);">：在保持时间局部性的前提下实现并行计算</font>
