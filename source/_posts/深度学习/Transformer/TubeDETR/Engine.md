---
title: Engine
date: '2025-08-06 16:42:10'
updated: '2025-08-22 20:34:11'
categories:
  - 人工智能
tags:
  - 深度学习
  - TubeDETR
cover: /images/custom-cover.jpg
recommend: true
---
```python
# 初始化指标记录器
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100 # 打印频率

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
```

[Util](https://www.yuque.com/yuqueyonghucfm7kr/awlus4/rlxt1rl4yxn3tg8d)里面知乎详细解读这俩工具类

`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">MetricLogger.log_every()</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">是一个</font>**<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">生成器函数</font>**<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">，它：</font>

1. <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">1.</font>**<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">迭代原始数据加载器</font>**<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">：</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">for obj in iterable</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">中的</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">iterable</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">就是传入的</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">data_loader</font>`
2. <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">2.</font>**<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">添加日志功能</font>**<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">：每</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">print_freq</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">步打印一次训练进度和指标</font>
3. <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">3.</font>**<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">透传原始数据</font>**<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">：通过</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">yield obj</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">返回未经修改的batch数据</font>



<font style="color:rgba(255, 255, 255, 0.6);background-color:rgb(29, 29, 29);">模型前向传播</font>
