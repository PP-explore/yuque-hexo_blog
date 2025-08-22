---
title: main处理
date: '2025-08-05 17:27:58'
updated: '2025-08-22 20:34:07'
categories:
  - 人工智能
tags:
  - 深度学习
  - TubeDETR
cover: /images/custom-cover.jpg
recommend: true
---
### 参数列表:
  超参数列表:

```python
# Training hyper-parameters
parser.add_argument("--lr", default=5e-5, type=float)

#控制模型骨干网络（Backbone）学习率（Learning Rate）
parser.add_argument("--lr_backbone", default=1e-5, type=float)

parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--lr_drop", default=10, type=int)
parser.add_argument(
    "--epoch_chunks",
    default=-1,
    type=int,
    help="If greater than 0, will split the training set into chunks and validate/checkpoint after each chunk",
)
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument(
    "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
)
parser.add_argument(
    "--eval_skip",
    default=1,
    type=int,
    help='do evaluation every "eval_skip" epochs',
)

parser.add_argument(
    "--schedule",
    default="linear_with_warmup",
    type=str,
    choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"),
)
parser.add_argument("--ema", action="store_true")
parser.add_argument("--ema_decay", type=float, default=0.9998)
parser.add_argument(
    "--fraction_warmup_steps",
    default=0.01,
    type=float,
    help="Fraction of total number of steps",
)

```

### <font style="color:#000000;">随机种子:</font>
<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">将随机种子与进程rank相加，确保每个进程的随机种子不同，从而避免分布式训练中数据重复。</font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">设置PyTorch的随机种子，确保PyTorch中的随机操作（如权重初始化、数据采样等）是可重复的。</font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">设置NumPy的随机种子，确保NumPy中的随机操作（如数组随机化）是可重复的。</font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">设置Python标准库</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">random</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">模块的随机种子，确保Python中的随机操作（如随机数生成）是可重复的。</font>

```java
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
```



## <font style="color:rgba(255, 255, 255, 0.9);background-color:rgb(29, 29, 29);">数据加载流程</font>
```python
sampler_train = torch.utils.data.RandomSampler(dataset_train)
```

1. **<font style="color:rgb(0, 0, 0);">初始化</font>**<font style="color:rgb(0, 0, 0);">：</font>
    - <font style="color:rgb(0, 0, 0);">接收</font>`<font style="color:rgb(0, 0, 0);">dataset_train</font>`<font style="color:rgb(0, 0, 0);">（</font>`<font style="color:rgb(0, 0, 0);">VideoModulatedSTGrounding</font>`<font style="color:rgb(0, 0, 0);">实例）</font>
    - <font style="color:rgb(0, 0, 0);">获取数据集长度：</font>`<font style="color:rgb(0, 0, 0);">len(dataset_train)</font>`<font style="color:rgb(0, 0, 0);">（视频样本总数）</font>
2. **<font style="color:rgb(0, 0, 0);">生成索引序列</font>**<font style="color:rgb(0, 0, 0);">：</font>

```plain
# 内部实现等价于
indices = list(range(len(dataset_train)))  # [0, 1, 2, ..., N-1]
random.shuffle(indices)  # 随机打乱，如[2, 0, 1, ..., N-3]
```

3. **<font style="color:rgb(0, 0, 0);">与</font>**`**<font style="color:rgb(0, 0, 0);">VideoModulatedSTGrounding</font>**`**<font style="color:rgb(0, 0, 0);">的关系</font>**<font style="color:rgb(0, 0, 0);">：</font>
    - <font style="color:rgb(0, 0, 0);">不直接调用</font>`<font style="color:rgb(0, 0, 0);">__getitem__</font>`<font style="color:rgb(0, 0, 0);">，仅生成随机索引序列</font>
    - <font style="color:rgb(0, 0, 0);">每个epoch会重新打乱顺序（保证训练随机性）</font>



```python
batch_sampler_train = torch.utils.data.BatchSampler(
sampler_train, args.batch_size, drop_last=True
)
```

**<font style="color:#000000;">具体操作：</font>**

1. **<font style="color:#000000;">输入处理</font>**<font style="color:#000000;">：</font>
    - <font style="color:#000000;">接收</font>`<font style="color:#000000;">sampler_train</font>`<font style="color:#000000;">生成的随机索引序列（如[2, 0, 1, 3, 5, 4,...]）</font>
    - <font style="color:#000000;">按</font>`<font style="color:#000000;">batch_size</font>`<font style="color:#000000;">分组（假设batch_size=2 → [[2,0], [1,3], [5,4],...]）</font>
2. **<font style="color:#000000;">边界处理</font>**<font style="color:#000000;">：</font>
    - `<font style="color:#000000;">drop_last=True</font>`<font style="color:#000000;">：丢弃最后不足batch_size的样本</font>
        * <font style="color:#000000;">例如7个样本，batch_size=2 → 保留3个batch（6个样本）</font>
+ <font style="color:#000000;">此时仍未加载实际数据，仅准备batch索引组合</font>
+ <font style="color:#000000;">例如生成的batch索引[[2,0], [1,3]]表示：</font>
    - <font style="color:#000000;">第一个batch：第2和第0个视频样本</font>
    - <font style="color:#000000;">第二个batch：第1和第3个视频样本</font>

<font style="color:#000000;"></font>

```python
data_loader_train = DataLoader(
    dataset_train,
    batch_sampler=batch_sampler_train,
    collate_fn=partial(utils.video_collate_fn, False, 0),
    num_workers=args.num_workers,
)
```

![](/images/46287b864e87ecde221aac80d7696bdf.png)

`**<font style="color:#000000;">collate_fn</font>**`**<font style="color:#000000;">视频专用处理</font>**

```plain
partial(utils.video_collate_fn, False, 0)
```

+ **<font style="color:#000000;">作用</font>**<font style="color:#000000;">：将多个视频样本整合为一个batch</font>
+ **<font style="color:#000000;">参数</font>**<font style="color:#000000;">：</font>
    - <font style="color:#000000;">**第一个参数</font>`<font style="color:#000000;">False</font>`<font style="color:#000000;">**：</font>`<font style="color:#000000;">do_round=False</font>`<font style="color:#000000;">，不对图像尺寸做对齐到32的倍数（训练时通常为True）</font>
+ <font style="color:#000000;">•**第二个参数</font>`<font style="color:#000000;">0</font>`<font style="color:#000000;">**：</font>`<font style="color:#000000;">div_vid=0</font>`<font style="color:#000000;">，禁用长视频分割功能（完整处理每个视频）</font>

#### **<font style="color:#000000;">核心功能</font>**
+ <font style="color:#000000;">•</font>**<font style="color:#000000;">批量化处理</font>**<font style="color:#000000;">：将多个不同长度的视频样本打包成统一格式的batch</font>
+ <font style="color:#000000;">•</font>**<font style="color:#000000;">智能填充</font>**<font style="color:#000000;">：自动处理不同分辨率/长度的视频帧</font>
+ <font style="color:#000000;">•</font>**<font style="color:#000000;">多模态整合</font>**<font style="color:#000000;">：关联视频帧、文本描述、时间标注等信息</font>

`**<font style="color:#000000;">num_workers</font>**`**<font style="color:#000000;">多进程加载</font>**

+ <font style="color:#000000;">启动多个子进程并行执行</font>`<font style="color:#000000;">__getitem__</font>`
+ <font style="color:#000000;">加速数据加载（特别是视频解码耗时操作）</font>

**<font style="color:#000000;">与</font>**`**<font style="color:#000000;">VideoModulatedSTGrounding</font>**`**<font style="color:#000000;">的完整交互</font>**

### <font style="color:#000000;">训练迭代示例：</font>
```python
for epoch in range(epochs):
    for batch in data_loader_train:  # 每次迭代触发以下流程
        # 1. BatchSampler生成如[3,1]的索引
        # 2. 并行调用：
        #    - dataset_train.__getitem__(3)
        #    - dataset_train.__getitem__(1)
        # 3. 每个__getitem__执行：
        #    - 视频解码
        #    - 时空裁剪/压缩
        #    - 数据增强
        # 4. collate_fn合并两个样本
        # 5. 返回batch数据给模型
```

  
 

## <font style="color:#000000;">模型权重加载机制</font>
<font style="color:#000000;">从检查点(checkpoint)加载模型权重的功能，主要用于两种场景：</font>

1. **<font style="color:#000000;">迁移学习</font>**<font style="color:#000000;">：加载预训练权重后从头开始训练</font>
2. **<font style="color:#000000;">恢复训练</font>**<font style="color:#000000;">：继续之前的训练过程</font>

```python
if args.load:  # 检查是否指定了加载路径
    print("loading from", args.load)
    checkpoint = torch.load(args.load, map_location="cpu")  # 加载检查点文件
```

<font style="color:#000000;">关键点：</font>

+ `<font style="color:#000000;">map_location="cpu"</font>`<font style="color:#000000;">：确保权重先加载到CPU，避免GPU内存问题</font>
+ `<font style="color:#000000;">checkpoint</font>`<font style="color:#000000;">通常包含：</font>
    - `<font style="color:#000000;">model</font>`<font style="color:#000000;">：模型权重</font>
    - `<font style="color:#000000;">model_ema</font>`<font style="color:#000000;">：EMA(指数移动平均)模型权重</font>
    - `<font style="color:#000000;">optimizer</font>`<font style="color:#000000;">：优化器状态(恢复训练时使用)</font>
    - `<font style="color:#000000;">epoch</font>`<font style="color:#000000;">：训练轮次</font>

**<font style="color:#000000;">EMA模型权重处理</font>**

```python
if "model_ema" in checkpoint:  # 优先使用EMA权重
    # 处理object queries数量不匹配的情况
    if args.num_queries < 100 and "query_embed.weight" in checkpoint["model_ema"]:
        checkpoint["model_ema"]["query_embed.weight"] = checkpoint["model_ema"]["query_embed.weight"][:args.num_queries]

    # 删除时间嵌入参数(可能不兼容)
    if "transformer.time_embed.te" in checkpoint["model_ema"]:
        del checkpoint["model_ema"]["transformer.time_embed.te"]

    # 非严格模式加载(允许部分参数不匹配)
    model_without_ddp.load_state_dict(checkpoint["model_ema"], strict=False)
```

<font style="color:#000000;">处理逻辑：</font>

1. **<font style="color:#000000;">Object Queries适配</font>**<font style="color:#000000;">：</font>
    - <font style="color:#000000;">当目标模型的查询数(</font>`<font style="color:#000000;">num_queries</font>`<font style="color:#000000;">)小于源模型时</font>
    - <font style="color:#000000;">只保留前N个query embeddings</font>
    - <font style="color:#000000;">例如：源模型有100个queries，新模型只需20个，则取前20个</font>
2. **<font style="color:#000000;">时间嵌入处理</font>**<font style="color:#000000;">：</font>
    - <font style="color:#000000;">删除时间编码参数</font>`<font style="color:#000000;">time_embed.te</font>`<font style="color:#000000;">，可能因为：</font>
        * <font style="color:#000000;">新模型使用不同的时间编码方式</font>
        * <font style="color:#000000;">需要重新初始化时间相关参数</font>
3. **<font style="color:#000000;">非严格加载</font>**<font style="color:#000000;">：</font>
    - `<font style="color:#000000;">strict=False</font>`<font style="color:#000000;">允许部分参数不加载</font>
    - <font style="color:#000000;">适用于模型结构有变化的情况</font>

**<font style="color:#000000;">特殊初始化处理</font>**

```plain
if "pretrained_resnet101_checkpoint.pth" in args.load:
    model_without_ddp.transformer._reset_temporal_parameters()
if args.ema:
    model_ema = deepcopy(model_without_ddp)
```

<font style="color:#000000;">关键操作：</font>

1. **<font style="color:#000000;">重置时序参数</font>**<font style="color:#000000;">：</font>
    - <font style="color:#000000;">当加载ResNet101预训练权重时</font>
    - <font style="color:#000000;">单独重新初始化transformer的时间相关参数</font>
    - <font style="color:#000000;">确保时空建模从合适的状态开始</font>
2. **<font style="color:#000000;">EMA模型初始化</font>**<font style="color:#000000;">：</font>
    - <font style="color:#000000;">如果需要使用EMA(指数移动平均)</font>
    - <font style="color:#000000;">从主模型深拷贝初始化</font>



## 训练流程
```python
train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            weight_dict=weight_dict,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            max_norm=args.clip_max_norm,
            model_ema=model_ema,
            writer=writer,
        )
```
