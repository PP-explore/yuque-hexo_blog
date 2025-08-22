---
title: 模型TubeDETR
date: '2025-08-08 11:04:59'
updated: '2025-08-22 16:19:15'
---
TubeDETR 的核心架构包含三个关键组件：

视觉主干网络：提取视频帧的空间特征

Transformer 编码器：融合视觉和文本特征

Transformer 解码器：生成时空预测结果

![](/images/5291f00ccc7b1211a5036203c018ef71.png)



__init__方法中:

```python
class TubeDETR(nn.Module):
    def __init__(
        self,
        backbone,         # 视觉主干网络（如 ResNet）
        transformer,      # Transformer 模型 
        num_queries,      # 每帧的对象查询数量    1
        aux_loss=False,   # 是否在每个解码层使用辅助损失  默认为true
        video_max_len=200, # 模型支持的最大视频帧数    200
        stride=5,         # 时间步长（用于降低时间分辨率）    5
        guided_attn=False, # 是否使用引导注意力损失       true
        fast=False,        # 是否使用快速分支（用于处理不同分辨率）  true
        fast_mode="",      # 快速分支的具体模式	  ""
        sted=True,         # 是否预测开始和结束时间   
    ):
        """
        :param backbone: 视觉主干模型，用于提取帧级特征
        :param transformer: Transformer 模型，负责特征融合和预测
        :param num_queries: 每帧的对象查询数量（类似 DETR 中的 object queries）
        :param aux_loss: 是否在每个解码层计算辅助损失（用于深度监督）
        :param video_max_len: 模型能处理的最大视频帧数（用于内存预分配）
        :param stride: 时间步长 k（用于降低时间分辨率）
        :param guided_attn: 是否使用引导注意力损失（提升注意力机制的效果）
        :param fast: 是否使用快速分支（处理不同时间/空间分辨率的输入）
        :param fast_mode: 快速分支的具体实现模式（如 "temporal" 或 "spatial"）
        :param sted: 是否预测开始和结束时间（时空端点检测）
        """
        super().__init__()
        
        # 保存基本参数
        self.num_queries = num_queries  # 每帧的查询数量
        self.transformer = transformer  # Transformer 模型
        hidden_dim = transformer.d_model  # Transformer 的隐藏维度
        
        # 边界框预测头 - 3层MLP，输出4个值（cx, cy, w, h）
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # 查询嵌入 - 可学习的查询向量
        # 形状: [num_queries, hidden_dim]
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # 输入投影层 - 将主干网络特征投影到Transformer维度
        # 输入通道: backbone.num_channels (如 ResNet 的2048)
        # 输出通道: hidden_dim (Transformer 的 d_model)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        
        # 主干网络
        self.backbone = backbone
        
        # 辅助损失标志
        self.aux_loss = aux_loss
        
        # 视频处理参数
        self.video_max_len = video_max_len  # 最大视频长度
        self.stride = stride                # 时间步长（用于降低帧率）
        
        # 注意力引导参数
        self.guided_attn = guided_attn  # 是否使用引导注意力
        
        # 快速分支参数
        self.fast = fast                # 是否启用快速分支
        self.fast_mode = fast_mode      # 快速分支的具体模式
        
        # 时空端点检测标志
        self.sted = sted
        if sted:
            # STED 预测头 - 2层MLP，输出2个值（开始概率，结束概率）
            # 使用 dropout=0.5 防止过拟合
            self.sted_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.5)
```

其中涉及到一个多层感知机全连接层的代码:

```python
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0):
        super().__init__()
        self.num_layers = num_layers  #深度
        # 3. 构造一个列表，用来表示中间隐藏层的维度
        # 创建一个列表h，它表示隐藏层的大小。由于总层数为num_layers，那么隐藏层的数量为num_layers-1（因为最后一层是输出层）
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            # 例如 num_layers为3则 [input_dim, hidden_dim,  hidden_dim, hidden_dim,  hidden_dim, output_dim]
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            ## 对非最后一层应用 ReLU 激活
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            ## 应用 Dropout（非最后一层且启用了 Dropout）
            if self.dropout and i < self.num_layers:
                x = self.dropout(x)
        return x
```

nn.Embedding词嵌入向量:[https://www.ppmy.cn/devtools/168335.html?action=onClick](https://www.ppmy.cn/devtools/168335.html?action=onClick)





**前向传播**

<font style="color:#000000;">输入格式：</font>

`<font style="color:#000000;">samples</font>`<font style="color:#000000;">: NestedTensor 结构，包含：</font>

`<font style="color:#000000;">.tensor</font>`<font style="color:#000000;">: 视频帧张量，形状为 </font>`<font style="color:#000000;">[n_frames, 3, H, W]</font>`

`<font style="color:#000000;">.mask</font>`<font style="color:#000000;">: 二进制掩码，形状为 </font>`<font style="color:#000000;">[n_frames, H, W]</font>`<font style="color:#000000;">，标识填充像素</font>

`<font style="color:#000000;">durations</font>`<font style="color:#000000;">: 每个视频的实际帧数列表</font>

`<font style="color:#000000;">captions</font>`<font style="color:#000000;">: 文本描述列表</font>

## <font style="color:rgba(255, 255, 255, 0.9);background-color:rgb(29, 29, 29);">编码阶段(encode_and_save=True)</font>
```python
if encode_and_save:
    assert memory_cache is None
    b = len(durations)  # batch_size
    t = max(durations)  # 最大视频长度(帧数)
    
    # 通过主干网络提取特征
    features, pos = self.backbone(samples)
    # 分解最后一层特征
    src, mask = features[-1].decompose()  
    # src形状: (n_frames)×F×(H/32)×(W/32)
    # mask形状: (n_frames)×(H/32)×(W/32)
```

[骨干模型](https://www.yuque.com/yuqueyonghucfm7kr/awlus4/txhok00z2u3xlar3)



```plain
if self.fast:
    # 不计算梯度(快速分支不反向传播到视觉主干)
    with torch.no_grad():  
        features_fast, pos_fast = self.backbone(samples_fast)
    src_fast, mask_fast = features_fast[-1].decompose()
    src_fast = self.input_proj(src_fast)  # 投影到Transformer维度
```

