---
title: DataLoader数据集加载和迭代
date: '2025-08-06 17:12:44'
updated: '2025-08-22 16:19:03'
---
```python
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=partial(utils.video_collate_fn, False, 0),
                num_workers=args.num_workers,
            )

.............

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

..............

    for i, batch_dict in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
```

![](/images/d2b6c647b1c47e72823ba333554b0729.png)

## <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">数据集处理总流程</font>
<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">数据集类</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">VideoModulatedSTGrounding</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">（继承自</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">torch.utils.data.Dataset</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">）的</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">__getitem__</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">：根据索引</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">idx</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">获取一个视频样本。</font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">首先调用build_dataset函数来初步读取数据集:</font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">主要工作有在build函数值中读取json文件然后构建一个数据集类对象VideoModulatedSTGrounding</font>

```plain
    dataset = VideoModulatedSTGrounding(
        vid_dir,
        ann_file,
        transforms=make_video_transforms(
            image_set, cautious=True, resolution=args.resolution
        ),
        is_train=image_set == "train",
        video_max_len=args.video_max_len,
        video_max_len_train=args.video_max_len_train,
        fps=args.fps,
        tmp_crop=args.tmp_crop and image_set == "train",
        tmp_loc=args.sted,
        stride=args.stride,
    )
```

`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">__init__</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">方法：初始化，主要加载注解文件，并预先计算每个视频的帧索引（根据采样率、最大长度等</font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">先迭代json文件中videos字段:</font>

`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);"> [{"original_video_id": "2727682922", "frame_count": 604, "fps": 15, "width": 320, "height": 240, "start_frame": 0, "end_frame": 603, "tube_start_frame": 124, "tube_end_frame": 604, "video_path": "0087/2727682922.mp4", "caption": "what does the adult man in black hold?", "type": "object", "target_id": 4, "video_id": 0, "qtype": "interrogative"}</font>`

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">参数video_max_len会限制前一步以固定采样率得到的视频帧的最大长度</font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">最终</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">vid2imgids[video["video_id"]] = [frame_ids, inter_frames]存储</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">该视频的帧ids</font>

返回给main函数之后,创建dataloader

```python
# 创建DataLoader
data_loader_train = DataLoader(
    dataset_train,
    batch_sampler=batch_sampler_train,
    collate_fn=partial(utils.video_collate_fn, False, 0),
    num_workers=args.num_workers,
)
```

	设置批采样以及随机图像增强 

```python
batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True
            )
```

**每一epoch遍历开始执行:**

迭代dataloader

### <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">步骤1: 批采样器生成一个批次的索引</font>
+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">例如，</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">batch_sampler_train</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">返回一个索引列表</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">[3, 7, 2]</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">（表示取数据集中第3、7、2个样本）</font>

### <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">步骤2: 对每个索引调用</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">dataset_train.__getitem__()</font>`
+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•假设</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">idx=3</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">：调用</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">dataset_train[3]</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">，返回一个元组</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">(images, targets, tmp_target)</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">（或四元组如果启用stride）</font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">在</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">__getitem__</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">中：</font>

+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">步骤1：获取视频的元数据（如caption, video_id等）</font>
+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•步骤2：从vid2imgids提取video_id的帧索引（frame_ids）使用ffmpeg提取视频帧（得到numpy数组，形状为[帧数, 高度, 宽度, 3]）</font>
+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">步骤3：为每一帧准备目标（target）数据，包括边界框等，并标记哪些帧在标注的时间间隔内（inter_frames）</font>
+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">步骤4：应用空间变换（transforms）将帧列表转换为张量（CTHW格式）并调整目标数据</font>
+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">步骤5：可能进行时间裁剪（temporal crop）和训练时的帧数压缩（如果帧数超过video_max_len_train）</font>
+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">步骤6：构建视频级的目标（tmp_target），包含视频ID、问题类型、动作帧索引区间、帧ID列表和描述文本</font>
+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">步骤7：返回</font>
    - <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">图像张量（如果stride>0，会返回两个：一个是原始帧，一个是按步长采样的帧）</font>
    - <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">目标列表（每个元素对应一帧的目标数据）</font>
    - <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•视频级目标（tmp_target）</font>

### <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">步骤3: 将多个样本的元组送入</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">collate_fn</font>`
(div_vid切割视频分段,只在val时对视频进行分割,训练时不分割)

+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">输入：一个批次的样本列表，每个样本是元组（假设3个样本）：</font>

```plain
batch_samples = [
    (images0, targets0, tmp_target0), 
    (images1, targets1, tmp_target1), 
    (images2, targets2, tmp_target2)
]
```

+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">注意：如果</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">div_vid=0</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">，则不会分割视频。</font>

### <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">步骤4:</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);"> </font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">collate_fn</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">处理</font>
<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">在</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">video_collate_fn(do_round=False, div_vid=0, batch=batch_samples)</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">中：</font>

1. <font style="color:#000000;background-color:rgba(255, 255, 255, 0);"></font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">重组数据：</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">batch = list(zip(*batch_samples))</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);"> </font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">得到：</font>

```python
batch_reorg = [
    (images0, images1, images2),              # 所有图像张量
    (targets0, targets1, targets2),           # 所有目标列表
    (tmp_target0, tmp_target1, tmp_target2)    # 所有视频级目标
]
```

2. <font style="color:#000000;background-color:rgba(255, 255, 255, 0);"></font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">处理图像：</font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">final_batch["samples"] = NestedTensor.from_tensor_list(batch_reorg[0], do_round=False)</font>

+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);"></font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">NestedTensor</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">将不同尺寸的视频（不同H,W）进行填充，并生成掩码mask。</font>
+ <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">结果形状：</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">samples.tensor</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);"> -> </font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">(3, C, T, H, W)</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">，其中T不一定相同（但每个视频的T由</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">__getitem__</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">决定）</font>
+ `<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">samples.mask</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);"> -> </font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">(3, T, H, W)</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">，指示有效像素（1有效，0填充）</font>
3. <font style="color:#000000;background-color:rgba(255, 255, 255, 0);"></font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">提取时长：</font>

4. <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">final_batch["durations"] = [len(targets0), len(targets1), len(targets2)]</font>
    - <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">注意：每个</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">targets_i</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">的长度就是该视频的帧数（T_i）。</font>
5. <font style="color:#000000;background-color:rgba(255, 255, 255, 0);"></font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">展平目标列表（帧级目标）：</font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">final_batch["targets"] = [t for clip in batch_reorg[1] for t in clip]</font>

    - <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">将三个视频的目标列表连接成一个列表：  
</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">[targets0[0], targets0[1], ..., targets0[T0-1], targets1[0], ..., targets2[T2-1]]</font>`
    - <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">•</font><font style="color:#000000;background-color:rgba(255, 255, 255, 0);">注意：每个帧级目标包含该帧的</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">image_id</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">、</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">boxes</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">等。</font>
6. <font style="color:#000000;background-color:rgba(255, 255, 255, 0);"></font>

<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">提取视频级元数据：</font>

```plain
final_batch["captions"] = [tmp_target0['caption'], tmp_target1['caption'], tmp_target2['caption']]
final_batch["video_ids"] = [tmp_target0['video_id'], ...]
final_batch["frames_id"] = [tmp_target0['frames_id'], ...]  # 每个视频使用的原始帧ID列表
final_batch["inter_idx"] = [tmp_target0['inter_idx'], ...]   # 动作起止索引（如[5,25]）
```

### <font style="color:#000000;background-color:rgba(255, 255, 255, 0);">最终输出</font>
`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">data_loader_train</font>`<font style="color:#000000;background-color:rgba(255, 255, 255, 0);">的一次迭代返回一个字典，包含：</font>

```plain
{
    'samples': NestedTensor( # 图像数据
        tensor:  形状 (B, C, T, H_max, W_max) 的填充张量,
        mask:    形状 (B, T, H_max, W_max) 的二进制掩码
    ),
    'durations': [T0, T1, T2], # 整数列表，表示每个视频实际帧数（未填充前）
    'targets': [                # 帧级目标列表（已展平）
        # 第一个视频的所有帧目标，然后是第二个，等等
        { 'image_id': 'vid0_100', 'boxes': 张量, ... }, 
        ... # 共 T0 个
        { ... }, ... # 第二个视频的 T1 个目标
        { ... }      # 第三个视频的 T2 个目标
    ],
    'captions': ['caption0', 'caption1', 'caption2'],
    'video_ids': ['vid0', 'vid1', 'vid2'],
    'frames_id': [  # 注意：这是一个列表的列表
        [100, 105, 110, ...], # 视频0的原始帧ID
        [203, 208, ...],       # 视频1
        ... 
    ],
    'inter_idx': [  # 每个视频的动作起始和结束帧索引（在frames_id中的索引位置）
        [start_idx0, end_idx0], # 例如 [2,5] 表示视频0的frames_id中索引2到5是动作帧
        [start_idx1, end_idx1],
        ...
    ]
}
```



[从DETR backbone 的NestedTensor 到DataLoader, Sampler,collate_fn，再到DETR transformer_detr的backbone-CSDN博客](https://blog.csdn.net/qq_35831906/article/details/124524455)

