---
title: TubeDETR 论文及代码阅读
date: '2025-07-21 10:33:38'
updated: '2025-09-22 16:27:11'
categories:
  - 人工智能
tags:
  - 深度学习
  - TubeDETR
cover: /images/custom-cover.jpg
recommend: true
---
详细代码解析请查看 TubeDETR标签 下的文章

### 论文逻辑梳理:
#### **<font style="color:rgba(0, 0, 0, 0.9);">视觉特征</font>**<font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">x₀(v) ∈ ℝ^(T×HW×d)</font>`
| <font style="color:rgba(0, 0, 0, 0.9);">维度</font> | <font style="color:rgba(0, 0, 0, 0.9);">含义</font> | <font style="color:rgba(0, 0, 0, 0.9);">典型值示例</font> |
| :---: | :---: | :---: |
| `<font style="color:rgba(0, 0, 0, 0.9);">T</font>` | <font style="color:rgba(0, 0, 0, 0.9);">视频总帧数</font> | <font style="color:rgba(0, 0, 0, 0.9);">200（长视频降采样后）</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">HW</font>` | <font style="color:rgba(0, 0, 0, 0.9);">单帧特征图展平尺寸</font> | <font style="color:rgba(0, 0, 0, 0.9);">7×7=49（ResNet最后一层特征图）</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">d</font>` | <font style="color:rgba(0, 0, 0, 0.9);">特征通道数</font> | <font style="color:rgba(0, 0, 0, 0.9);">256（论文默认维度）</font> |


+ **<font style="color:rgba(0, 0, 0, 0.9);">生成方式</font>**<font style="color:rgba(0, 0, 0, 0.9);">：  
</font><font style="color:rgba(0, 0, 0, 0.9);">视频帧 → ResNet-101 → 空间平均池化 → 展平为</font>`<font style="color:rgba(0, 0, 0, 0.9);">HW</font>`<font style="color:rgba(0, 0, 0, 0.9);">向量</font>

_<font style="color:rgba(0, 0, 0, 0.6);">例如：2048维ResNet特征经投影层压缩至</font>_`_<font style="color:rgba(0, 0, 0, 0.6);">d=256</font>_`

#### **<font style="color:rgba(0, 0, 0, 0.9);">文本特征</font>**<font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">y₀(s) ∈ ℝ^(L×d)</font>`
| <font style="color:rgba(0, 0, 0, 0.9);">维度</font> | <font style="color:rgba(0, 0, 0, 0.9);">含义</font> | <font style="color:rgba(0, 0, 0, 0.9);">典型值示例</font> |
| :---: | :---: | :---: |
| `<font style="color:rgba(0, 0, 0, 0.9);">L</font>` | <font style="color:rgba(0, 0, 0, 0.9);">查询语句token数</font> | <font style="color:rgba(0, 0, 0, 0.9);">20（RoBERTa分词后长度）</font> |
| `<font style="color:rgba(0, 0, 0, 0.9);">d</font>` | <font style="color:rgba(0, 0, 0, 0.9);">与视觉特征对齐的维度</font> | <font style="color:rgba(0, 0, 0, 0.9);">256</font> |


**<font style="color:rgba(0, 0, 0, 0.9);">生成方式</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>  
<font style="color:rgba(0, 0, 0, 0.9);">文本 → RoBERTa → 取最后一层隐状态 → 线性投影至</font>`<font style="color:rgba(0, 0, 0, 0.9);">d</font>`<font style="color:rgba(0, 0, 0, 0.9);">维</font>

#### <font style="color:rgba(0, 0, 0, 0.9);">特征拼接机制</font>
<font style="color:rgba(0, 0, 0, 0.9);">(1) 拼接操作定义</font>

<font style="color:rgba(0, 0, 0, 0.9);">对于</font>**<font style="color:rgba(0, 0, 0, 0.9);">每一帧</font>**<font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">t ∈ [1,T]</font>`<font style="color:rgba(0, 0, 0, 0.9);">：</font>

1. <font style="color:rgba(0, 0, 0, 0.9);">取出视觉特征</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">x₀(v)[t] ∈ ℝ^(HW×d)</font>`
2. <font style="color:rgba(0, 0, 0, 0.9);">与</font>**<font style="color:rgba(0, 0, 0, 0.9);">整个文本特征</font>**<font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">y₀(s) ∈ ℝ^(L×d)</font>`<font style="color:rgba(0, 0, 0, 0.9);"> 拼接：</font>![](/images/719bdcbc19eceb5315a1b24463a3cd2f.png)

_<font style="color:rgba(0, 0, 0, 0.6);">拼接发生在序列维度（</font>_`_<font style="color:rgba(0, 0, 0, 0.6);">HW+L</font>_`_<font style="color:rgba(0, 0, 0, 0.6);">），特征维度</font>_`_<font style="color:rgba(0, 0, 0, 0.6);">d</font>_`_<font style="color:rgba(0, 0, 0, 0.6);">保持不变</font>_

<font style="color:rgba(0, 0, 0, 0.9);">(2) 拼接后的结构</font>

+ **<font style="color:rgba(0, 0, 0, 0.9);">输出特征</font>**<font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">F(v,s) ∈ ℝ^(T×(HW+L)×d)</font>`<font style="color:rgba(0, 0, 0, 0.9);">：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">每帧包含：</font>`<font style="color:rgba(0, 0, 0, 0.9);">HW</font>`<font style="color:rgba(0, 0, 0, 0.9);">个视觉位置特征 +</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">L</font>`<font style="color:rgba(0, 0, 0, 0.9);">个文本token特征  
</font><font style="color:rgba(0, 0, 0, 0.9);">（如图3中编码器输出的紫色与橙色块拼接）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">物理意义</font>**<font style="color:rgba(0, 0, 0, 0.9);">：  
</font><font style="color:rgba(0, 0, 0, 0.9);">每个空间位置（共</font>`<font style="color:rgba(0, 0, 0, 0.9);">HW</font>`<font style="color:rgba(0, 0, 0, 0.9);">个）都能感知全文本文义（</font>`<font style="color:rgba(0, 0, 0, 0.9);">L</font>`<font style="color:rgba(0, 0, 0, 0.9);">个token）</font>

#### <font style="color:rgba(0, 0, 0, 0.9);">慢速多模态分支（Slow Branch）</font>
#### 拼好帧
##### <font style="color:rgba(0, 0, 0, 0.9);">采样步骤</font>
1. **<font style="color:rgba(0, 0, 0, 0.9);">视频分段</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">将视频划分为 </font>_<font style="color:rgba(0, 0, 0, 0.9);">M</font>_<font style="color:rgba(0, 0, 0, 0.9);"> 个片段（clip），每段连续</font>**<font style="color:rgba(0, 0, 0, 0.9);"> </font>**_**<font style="color:rgba(0, 0, 0, 0.9);">k</font>**_**<font style="color:rgba(0, 0, 0, 0.9);"> 帧（这里是指的视频帧率</font>**<font style="color:rgba(0, 0, 0, 0.9);">,默认 </font>_<font style="color:rgba(0, 0, 0, 0.9);">k</font>_<font style="color:rgba(0, 0, 0, 0.9);">=5，对应1秒）。</font>
    - _<font style="color:rgba(0, 0, 0, 0.9);">例：若</font>__<font style="color:rgba(0, 0, 0, 0.9);"> </font>__<font style="color:rgba(0, 0, 0, 0.9);">T</font>__<font style="color:rgba(0, 0, 0, 0.9);">=</font>__<font style="color:rgba(0, 0, 0, 0.9);">200</font>__<font style="color:rgba(0, 0, 0, 0.9);"> </font>__<font style="color:rgba(0, 0, 0, 0.9);">帧，则</font>__<font style="color:rgba(0, 0, 0, 0.9);"> </font>__<font style="color:rgba(0, 0, 0, 0.9);">M</font>__<font style="color:rgba(0, 0, 0, 0.9);">=</font>__<font style="color:rgba(0, 0, 0, 0.9);">40</font>__<font style="color:rgba(0, 0, 0, 0.9);"> </font>__<font style="color:rgba(0, 0, 0, 0.9);">个片段</font>_<font style="color:rgba(0, 0, 0, 0.9);">。</font>
2. **<font style="color:rgba(0, 0, 0, 0.9);">帧选择策略</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>
    - **<font style="color:rgba(0, 0, 0, 0.9);">每片段中心采样</font>**<font style="color:rgba(0, 0, 0, 0.9);">：取每 </font>_<font style="color:rgba(0, 0, 0, 0.9);">k</font>_<font style="color:rgba(0, 0, 0, 0.9);"> 帧片段的</font>**<font style="color:rgba(0, 0, 0, 0.9);">中间1帧</font>**<font style="color:rgba(0, 0, 0, 0.9);">（如图3顶部箭头所示）。  
</font>_<font style="color:rgba(0, 0, 0, 0.9);">公式</font>_<font style="color:rgba(0, 0, 0, 0.9);">：</font>![](/images/26eb96726513b66c7b060f767e7333cd.png)
    - _<font style="color:rgba(0, 0, 0, 0.9);">替代方案</font>_<font style="color:rgba(0, 0, 0, 0.9);">：随机采样或首帧采样（论文默认中心采样效果最佳）。</font>
3. **<font style="color:rgba(0, 0, 0, 0.9);">处理</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>
    1. <font style="color:rgba(0, 0, 0, 0.9);">拼接特征</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">Concat(xₘᵖ, y₀(s)) ∈ ℝ^(HW+L)×d</font>`
    2. <font style="color:rgba(0, 0, 0, 0.9);">通过Transformer编码器计算跨模态注意力：拼接后的 Input</font>_<font style="color:rgba(0, 0, 0, 0.9);">m</font>_<font style="color:rgba(0, 0, 0, 0.9);"> 输入N层Transformer编码器：</font>
        1. **<font style="color:rgba(0, 0, 0, 0.9);">自注意力机制</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>
            + <font style="color:rgba(0, 0, 0, 0.9);">视觉位置（HW部分）与文本token（L部分）相互计算注意力权重  
</font>_<font style="color:rgba(0, 0, 0, 0.9);">（如图3热力图显示"抓取"激活手部区域）</font>_
        2. **<font style="color:rgba(0, 0, 0, 0.9);">输出</font>**<font style="color:rgba(0, 0, 0, 0.9);">：  
</font><font style="color:rgba(0, 0, 0, 0.9);">上下文化的表示</font>![](/images/106f37cec528d9d7b31675b40f13c9e5.png)<font style="color:rgba(0, 0, 0, 0.9);">，其中：</font>
            + <font style="color:rgba(0, 0, 0, 0.9);">视觉部分：文本条件化的空间特征  
</font>_<font style="color:rgba(0, 0, 0, 0.9);">（如"球"的视觉特征被文本强化）</font>_
            + <font style="color:rgba(0, 0, 0, 0.9);">文本部分：视觉条件化的语言特征  
</font>_<font style="color:rgba(0, 0, 0, 0.9);">（如"抓"的动作语义被手部姿态调制）	</font>_

##### <font style="color:rgba(0, 0, 0, 0.9);">(2) 快速视觉分支（Fast Branch）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">输入</font>**<font style="color:rgba(0, 0, 0, 0.9);">：所有帧</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">x₀(v) ∈ ℝ^(T×HW×d)</font>`
+ **<font style="color:rgba(0, 0, 0, 0.9);">处理</font>**<font style="color:rgba(0, 0, 0, 0.9);">：单线性层</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">f(v) = Wx₀(v)</font>`<font style="color:rgba(0, 0, 0, 0.9);">  
</font><font style="color:rgba(0, 0, 0, 0.9);">（权重矩阵</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>`<font style="color:rgba(0, 0, 0, 0.9);">W ∈ ℝ^(d×d)</font>`<font style="color:rgba(0, 0, 0, 0.9);">）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">作用</font>**<font style="color:rgba(0, 0, 0, 0.9);">：线性层作为</font>**<font style="color:rgba(0, 0, 0, 0.9);">高通滤波器</font>**<font style="color:rgba(0, 0, 0, 0.9);">，保留高频运动信息（如快速移动的手部）</font>

#### <font style="color:rgba(0, 0, 0, 0.9);">特征聚合</font>
##### <font style="color:rgba(0, 0, 0, 0.9);">慢速分支特征的时间复制</font>
1. **<font style="color:rgba(0, 0, 0, 0.9);">输入特征</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);">慢速分支输出</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">h</font>__<font style="color:rgba(0, 0, 0, 0.9);">p</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">R</font>_<font style="color:rgba(0, 0, 0, 0.9);">M</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font><font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">H</font>__<font style="color:rgba(0, 0, 0, 0.9);">W</font>_<font style="color:rgba(0, 0, 0, 0.9);">+</font>_<font style="color:rgba(0, 0, 0, 0.9);">L</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">  
</font><font style="color:rgba(0, 0, 0, 0.9);">（</font>_<font style="color:rgba(0, 0, 0, 0.9);">M</font>_<font style="color:rgba(0, 0, 0, 0.9);">=</font>_<font style="color:rgba(0, 0, 0, 0.9);">T</font>_<font style="color:rgba(0, 0, 0, 0.9);">/</font>_<font style="color:rgba(0, 0, 0, 0.9);">k</font>_<font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">为片段数，例如200帧→40个片段）</font>
2. **<font style="color:rgba(0, 0, 0, 0.9);">时间复制</font>**<font style="color:rgba(0, 0, 0, 0.9);">：  
</font><font style="color:rgba(0, 0, 0, 0.9);">对每个片段 </font>_<font style="color:rgba(0, 0, 0, 0.9);">m</font>_<font style="color:rgba(0, 0, 0, 0.9);"> 的编码 </font>_<font style="color:rgba(0, 0, 0, 0.9);">hp</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)[</font>_<font style="color:rgba(0, 0, 0, 0.9);">m</font>_<font style="color:rgba(0, 0, 0, 0.9);">] </font>**<font style="color:rgba(0, 0, 0, 0.9);">重复k次</font>**<font style="color:rgba(0, 0, 0, 0.9);">，覆盖对应片段的原始k帧</font>

**<font style="color:rgba(0, 0, 0, 0.9);">意义:语义传播</font>**<font style="color:rgba(0, 0, 0, 0.9);">：将1秒（k帧）内的关键语义（如"抓球动作"）复制到所有子帧，确保时序一致性。</font>

##### <font style="color:rgba(0, 0, 0, 0.9);">双分支特征融合机制</font>
###### <font style="color:rgba(0, 0, 0, 0.9);">(1)分解慢速分支输出</font>
<font style="color:rgba(0, 0, 0, 0.9);">复制后的</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">h</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">包含两部分：</font>

![](/images/33427d6248750ff56bc2daac42c25de5.png)

+ **<font style="color:rgba(0, 0, 0, 0.9);">文本情境化视觉编码</font>**<font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">h</font>__<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">R</font>_<font style="color:rgba(0, 0, 0, 0.9);">T</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">H</font>__<font style="color:rgba(0, 0, 0, 0.9);">W</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">  
</font><font style="color:rgba(0, 0, 0, 0.9);">（已通过跨模态注意力融合文本语义的视觉特征）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">视觉情境化文本编码</font>**<font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">hs</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)∈R</font>_<font style="color:rgba(0, 0, 0, 0.9);">T</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">L</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">  
</font><font style="color:rgba(0, 0, 0, 0.9);">（已通过视觉特征调制的文本特征）</font>

###### <font style="color:rgba(0, 0, 0, 0.9);">(2)聚合模块 g 的操作</font>
1. **<font style="color:rgba(0, 0, 0, 0.9);">输入对齐</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>
    - _<font style="color:rgba(0, 0, 0, 0.9);">h</font>__<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">：稀疏语义特征（复制后）</font>
    - _<font style="color:rgba(0, 0, 0, 0.9);">f</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">：快速分支的细节特征（原始分辨率）</font>
2. **<font style="color:rgba(0, 0, 0, 0.9);">逐元素相加 + 线性投影</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>

![](/images/74dea41e9fd65468ef649bcf01276908.png)

    - **<font style="color:rgba(0, 0, 0, 0.9);">相加</font>**<font style="color:rgba(0, 0, 0, 0.9);">：融合全局语义（慢速）与局部细节（快速）</font>
    - **<font style="color:rgba(0, 0, 0, 0.9);">线性层</font>**<font style="color:rgba(0, 0, 0, 0.9);">：调整特征分布（权重矩阵</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">W</font>_<font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">R</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">）</font>
3. **<font style="color:rgba(0, 0, 0, 0.9);">残差连接</font>**<font style="color:rgba(0, 0, 0, 0.9);">：保留原始慢速分支的强语义</font>

![](/images/3e473ad77bcee9645957658a80fa20ca.png)

###### <font style="color:rgba(0, 0, 0, 0.9);">(3) 最终视频-文本特征拼接</font>
<font style="color:rgba(0, 0, 0, 0.9);">将聚合后的视觉编码与文本编码拼接：</font>

F(v,s)=[Fv(v,s),hs(v,s)]∈RT×(HW+L)×d

+ **<font style="color:rgba(0, 0, 0, 0.9);">视觉部分</font>**<font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">F</font>__<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">：多模态增强的时空特征</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">文本部分</font>**<font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">h</font>__<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">：视觉调制的语言特征</font>



### **<font style="color:rgba(0, 0, 0, 0.9);">解码器架构详解</font>**
#### **<font style="color:rgba(0, 0, 0, 0.9);">(1) 输入构造</font>**
| **<font style="color:rgba(0, 0, 0, 0.9);">输入项</font>** | **<font style="color:rgba(0, 0, 0, 0.9);">维度</font>** | **<font style="color:rgba(0, 0, 0, 0.9);">作用</font>** |
| :---: | :---: | :---: |
| <font style="color:rgba(0, 0, 0, 0.9);">时间查询（Time Queries）</font> | <font style="color:rgba(0, 0, 0, 0.9);">{</font>_<font style="color:rgba(0, 0, 0, 0.9);">q</font>__<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">}</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">=</font><font style="color:rgba(0, 0, 0, 0.9);">1</font>_<font style="color:rgba(0, 0, 0, 0.9);">T</font>_<font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">R</font>_<font style="color:rgba(0, 0, 0, 0.9);">T</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_ | <font style="color:rgba(0, 0, 0, 0.9);">每帧的可学习定位信号，初始化=对象编码+时间编码</font> |
| <font style="color:rgba(0, 0, 0, 0.9);">视频-文本特征</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">F</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font> | <font style="color:rgba(0, 0, 0, 0.9);">R</font>_<font style="color:rgba(0, 0, 0, 0.9);">T</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font><font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">H</font>__<font style="color:rgba(0, 0, 0, 0.9);">W</font>_<font style="color:rgba(0, 0, 0, 0.9);">+</font>_<font style="color:rgba(0, 0, 0, 0.9);">L</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_ | <font style="color:rgba(0, 0, 0, 0.9);">编码器输出的多模态特征（含视觉+文本语义）</font> |


+ **<font style="color:rgba(0, 0, 0, 0.9);">时间查询初始化</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font><font style="color:rgba(0, 0, 0, 0.9);">qt=ObjectEmbed+TimeEnc(t)</font>
+ <font style="color:rgba(0, 0, 0, 0.9);"></font>
    - `<font style="color:rgba(0, 0, 0, 0.9);">ObjectEmbed</font>`<font style="color:rgba(0, 0, 0, 0.9);">：可学习参数（所有帧共享，表示目标共性）维度 </font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">=256</font>
    - `<font style="color:rgba(0, 0, 0, 0.9);">TimeEnc</font>`<font style="color:rgba(0, 0, 0, 0.9);">：冻结的正弦位置编码（注入时序信息）与Transformer相同的正弦函数</font>

#### **<font style="color:rgba(0, 0, 0, 0.9);">(2) 解码器块结构（N层堆叠）</font>**
<font style="color:rgba(0, 0, 0, 0.9);">每层包含三个核心操作（如图4所示）：</font>

1. **<font style="color:rgba(0, 0, 0, 0.9);">时间自注意力（Temporal Self-Attention）</font>**
    - **<font style="color:rgba(0, 0, 0, 0.9);">作用</font>**<font style="color:rgba(0, 0, 0, 0.9);">：建模帧间全局关系（如第5帧"抓取"与第3帧"抬手"的因果性）</font>
    - **<font style="color:rgba(0, 0, 0, 0.9);">实现</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font><font style="color:rgba(0, 0, 0, 0.9);">Attention(Q=qt,K={qi}i=1T,V={qi}i=1T)</font>

<font style="color:rgba(0, 0, 0, 0.6);">*所有时间查询相互交互，复杂度</font><font style="color:rgba(0, 0, 0, 0.6);"> </font>_<font style="color:rgba(0, 0, 0, 0.6);">O</font>_<font style="color:rgba(0, 0, 0, 0.6);">(</font>_<font style="color:rgba(0, 0, 0, 0.6);">T</font>_<font style="color:rgba(0, 0, 0, 0.6);">2</font><font style="color:rgba(0, 0, 0, 0.6);">)</font><font style="color:rgba(0, 0, 0, 0.6);"> </font><font style="color:rgba(0, 0, 0, 0.6);">*</font>

2. **<font style="color:rgba(0, 0, 0, 0.9);">时间对齐交叉注意力（Time-Aligned Cross-Attention）</font>**
    - **<font style="color:rgba(0, 0, 0, 0.9);">作用</font>**<font style="color:rgba(0, 0, 0, 0.9);">：将当前帧的多模态特征注入查询</font>

**<font style="color:rgba(0, 0, 0, 0.9);">输入与输出</font>**

        * **<font style="color:rgba(0, 0, 0, 0.9);">输入</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>
            + **<font style="color:rgba(0, 0, 0, 0.9);">查询（Query）</font>**<font style="color:rgba(0, 0, 0, 0.9);">：解码器的Refined Time Query</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">Q</font>__<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">R</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">（当前帧的时间查询）</font>
            + **<font style="color:rgba(0, 0, 0, 0.9);">键值（Key-Value）</font>**<font style="color:rgba(0, 0, 0, 0.9);">：视频-文本特征</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">F</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">[</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">]</font><font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">R</font><font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">H</font>__<font style="color:rgba(0, 0, 0, 0.9);">W</font>_<font style="color:rgba(0, 0, 0, 0.9);">+</font>_<font style="color:rgba(0, 0, 0, 0.9);">L</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">（</font>**<font style="color:rgba(0, 0, 0, 0.9);">仅当前帧</font>**<font style="color:rgba(0, 0, 0, 0.9);">的多模态特征）</font>
        * **<font style="color:rgba(0, 0, 0, 0.9);">输出</font>**<font style="color:rgba(0, 0, 0, 0.9);">：更新后的时间查询</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">Q</font>__<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">′</font><font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">R</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">，融合当前帧的视觉与文本信息</font>

#### **<font style="color:rgba(0, 0, 0, 0.9);">(2) 计算步骤</font>**
**<font style="color:rgba(0, 0, 0, 0.9);">特征切片</font>**<font style="color:rgba(0, 0, 0, 0.9);">：  
</font><font style="color:rgba(0, 0, 0, 0.9);">从编码器输出</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">F</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">中提取</font>**<font style="color:rgba(0, 0, 0, 0.9);">当前帧t</font>**<font style="color:rgba(0, 0, 0, 0.9);">的特征：</font>

        1. <font style="color:rgba(0, 0, 0, 0.9);">Ft=F(v,s)[t,:,:]∈R(HW+L)×d</font>
            + <font style="color:rgba(0, 0, 0, 0.9);">包含：</font>
                - <font style="color:rgba(0, 0, 0, 0.9);">视觉特征</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">F</font>__<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">[</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">]</font><font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">R</font>_<font style="color:rgba(0, 0, 0, 0.9);">H</font>__<font style="color:rgba(0, 0, 0, 0.9);">W</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">（空间位置特征）</font>
                - <font style="color:rgba(0, 0, 0, 0.9);">文本特征</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">h</font>__<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">v</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">[</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">]</font><font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">R</font>_<font style="color:rgba(0, 0, 0, 0.9);">L</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">（语言token特征）</font>

**<font style="color:rgba(0, 0, 0, 0.9);">注意力权重计算</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>

        2. ![](/images/c4a605a3933f2088e03496992b29bbb5.png)
            + **<font style="color:rgba(0, 0, 0, 0.9);">关键约束</font>**<font style="color:rgba(0, 0, 0, 0.9);">：  
</font>_<font style="color:rgba(0, 0, 0, 0.9);">Q</font>__<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);"> </font>**<font style="color:rgba(0, 0, 0, 0.9);">仅与当前帧的</font>****<font style="color:rgba(0, 0, 0, 0.9);"> </font>**_**<font style="color:rgba(0, 0, 0, 0.9);">F</font>**__**<font style="color:rgba(0, 0, 0, 0.9);">t</font>**_**<font style="color:rgba(0, 0, 0, 0.9);"> </font>****<font style="color:rgba(0, 0, 0, 0.9);">交互</font>**<font style="color:rgba(0, 0, 0, 0.9);">，不跨帧计算（图4中对角白线所示）</font>
            + <font style="color:rgba(0, 0, 0, 0.9);">投影矩阵</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">W</font>__<font style="color:rgba(0, 0, 0, 0.9);">Q</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">W</font>__<font style="color:rgba(0, 0, 0, 0.9);">K</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">W</font>__<font style="color:rgba(0, 0, 0, 0.9);">V</font>_<font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">R</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_<font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">可学习</font>

**<font style="color:rgba(0, 0, 0, 0.9);">残差连接</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>

        1. <font style="color:rgba(0, 0, 0, 0.9);">Qt′=LayerNorm(Qt+Attention(Qt,Ft,Ft))</font>
3. **<font style="color:rgba(0, 0, 0, 0.9);">前馈网络（FFN）</font>**
    - **<font style="color:rgba(0, 0, 0, 0.9);">作用</font>**<font style="color:rgba(0, 0, 0, 0.9);">：特征非线性变换</font>
    - **<font style="color:rgba(0, 0, 0, 0.9);">实现</font>**<font style="color:rgba(0, 0, 0, 0.9);">：2层MLP + ReLU激活</font>

#### **<font style="color:rgba(0, 0, 0, 0.9);">(3) 输出处理</font>**
+ **<font style="color:rgba(0, 0, 0, 0.9);">Refined Queries</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font><font style="color:rgba(0, 0, 0, 0.9);">{</font>_<font style="color:rgba(0, 0, 0, 0.9);">Q</font>__<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">}</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">=</font><font style="color:rgba(0, 0, 0, 0.9);">1</font>_<font style="color:rgba(0, 0, 0, 0.9);">T</font>_<font style="color:rgba(0, 0, 0, 0.9);">（融合全局时序与局部多模态特征）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);">预测头</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>

**<font style="color:rgba(0, 0, 0, 0.9);">输入：精炼时间查询（Refined Time Queries）</font>**

        * **<font style="color:rgba(0, 0, 0, 0.9);">来源</font>**<font style="color:rgba(0, 0, 0, 0.9);">：时空解码器输出的</font><font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">{</font>_<font style="color:rgba(0, 0, 0, 0.9);">Q</font>__<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">}</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">=</font><font style="color:rgba(0, 0, 0, 0.9);">1</font>_<font style="color:rgba(0, 0, 0, 0.9);">T</font>_<font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">R</font>_<font style="color:rgba(0, 0, 0, 0.9);">T</font>_<font style="color:rgba(0, 0, 0, 0.9);">×</font>_<font style="color:rgba(0, 0, 0, 0.9);">d</font>_
            + <font style="color:rgba(0, 0, 0, 0.9);">每个</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">Q</font>__<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">融合了：</font>
                - **<font style="color:rgba(0, 0, 0, 0.9);">全局时序信息</font>**<font style="color:rgba(0, 0, 0, 0.9);">（通过时间自注意力）</font>
                - **<font style="color:rgba(0, 0, 0, 0.9);">当前帧的多模态特征</font>**<font style="color:rgba(0, 0, 0, 0.9);">（通过时间对齐交叉注意力）</font>

**<font style="color:rgba(0, 0, 0, 0.9);">输出：时空管状结构</font>**

| **<font style="color:rgba(0, 0, 0, 0.9);">预测类型</font>** | **<font style="color:rgba(0, 0, 0, 0.9);">输出形式</font>** | **<font style="color:rgba(0, 0, 0, 0.9);">用途</font>** |
| :---: | :---: | :---: |
| <font style="color:rgba(0, 0, 0, 0.9);">空间定位（每帧）</font> | _<font style="color:rgba(0, 0, 0, 0.9);">b</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">[</font><font style="color:rgba(0, 0, 0, 0.9);">0</font><font style="color:rgba(0, 0, 0, 0.9);">,</font><font style="color:rgba(0, 0, 0, 0.9);">1</font><font style="color:rgba(0, 0, 0, 0.9);">]</font><font style="color:rgba(0, 0, 0, 0.9);">4</font> | <font style="color:rgba(0, 0, 0, 0.9);">边界框坐标（中心x,y + 宽高w,h），归一化到视频帧尺寸</font> |
| <font style="color:rgba(0, 0, 0, 0.9);">时间定位（全局）</font> | _<font style="color:rgba(0, 0, 0, 0.9);">τ</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">τ</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">e</font>_<font style="color:rgba(0, 0, 0, 0.9);">∈</font><font style="color:rgba(0, 0, 0, 0.9);">[</font><font style="color:rgba(0, 0, 0, 0.9);">0</font><font style="color:rgba(0, 0, 0, 0.9);">,</font><font style="color:rgba(0, 0, 0, 0.9);">1</font><font style="color:rgba(0, 0, 0, 0.9);">]</font>_<font style="color:rgba(0, 0, 0, 0.9);">T</font>_ | <font style="color:rgba(0, 0, 0, 0.9);">预测每帧作为起止时间的概率分布</font> |
| **<font style="color:rgba(0, 0, 0, 0.9);">最终输出</font>** | <font style="color:rgba(0, 0, 0, 0.9);">{</font>_<font style="color:rgba(0, 0, 0, 0.9);">b</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">}</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">=</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>__<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">e</font>_ | <font style="color:rgba(0, 0, 0, 0.9);">时空管：选定时段内所有帧的边界框序列</font> |


<font style="color:rgba(0, 0, 0, 0.9);">具体实现</font>

    - **<font style="color:rgba(0, 0, 0, 0.9);">边界框序列</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>![](/images/57ea71a1315a409bff2d8ede1825018c.png)
    - **<font style="color:rgba(0, 0, 0, 0.9);">起止概率</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>![](/images/0860e2ce7b3318418e4e8ddd2389358d.png)

**<font style="color:rgba(0, 0, 0, 0.9);">(1) 空间定位头（3层MLP）</font>**

        * **<font style="color:rgba(0, 0, 0, 0.9);">结构</font>**<font style="color:rgba(0, 0, 0, 0.9);">：  
</font>`<font style="color:rgba(0, 0, 0, 0.9);">Linear(d→d) → ReLU → Linear(d→d) → ReLU → Linear(d→4)</font>`
            + <font style="color:rgba(0, 0, 0, 0.9);">最后一层使用Sigmoid激活，确保输出在</font><font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">[</font><font style="color:rgba(0, 0, 0, 0.9);">0</font><font style="color:rgba(0, 0, 0, 0.9);">,</font><font style="color:rgba(0, 0, 0, 0.9);">1</font><font style="color:rgba(0, 0, 0, 0.9);">]</font><font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">范围内。</font>
        * **<font style="color:rgba(0, 0, 0, 0.9);">示例</font>**<font style="color:rgba(0, 0, 0, 0.9);">：  
</font><font style="color:rgba(0, 0, 0, 0.9);">若</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">Q</font>__<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">编码“手抓球”状态，MLP输出可能为：  
</font>_<font style="color:rgba(0, 0, 0, 0.9);">b</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">=</font><font style="color:rgba(0, 0, 0, 0.9);">[</font><font style="color:rgba(0, 0, 0, 0.9);">0.6</font><font style="color:rgba(0, 0, 0, 0.9);">,</font><font style="color:rgba(0, 0, 0, 0.9);">0.3</font><font style="color:rgba(0, 0, 0, 0.9);">,</font><font style="color:rgba(0, 0, 0, 0.9);">0.1</font><font style="color:rgba(0, 0, 0, 0.9);">,</font><font style="color:rgba(0, 0, 0, 0.9);">0.1</font><font style="color:rgba(0, 0, 0, 0.9);">]</font><font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">→ 表示边界框中心在(60%,30%)，宽高各占10%帧尺寸。</font>

**<font style="color:rgba(0, 0, 0, 0.9);">(2) 时间定位头（2层MLP）</font>**

        * **<font style="color:rgba(0, 0, 0, 0.9);">结构</font>**<font style="color:rgba(0, 0, 0, 0.9);">：  
</font>`<font style="color:rgba(0, 0, 0, 0.9);">Linear(d→d) → ReLU → Linear(d→1)</font>`
            + <font style="color:rgba(0, 0, 0, 0.9);">起止头共享结构，但参数独立。</font>
        * **<font style="color:rgba(0, 0, 0, 0.9);">概率生成</font>**<font style="color:rgba(0, 0, 0, 0.9);">：</font>
            + <font style="color:rgba(0, 0, 0, 0.9);">对每个</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">Q</font>__<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">，分别预测标量</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">τ</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">[</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">]</font><font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">和</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">τ</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">e</font>_<font style="color:rgba(0, 0, 0, 0.9);">[</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">]</font>
            + <font style="color:rgba(0, 0, 0, 0.9);">通过Softmax归一化为概率分布：τ^s=Softmax([τ^s[1],...,τ^s[T]]),τ^e同理</font>

**<font style="color:rgba(0, 0, 0, 0.9);">推理时的关键步骤</font>**

**<font style="color:rgba(0, 0, 0, 0.9);">(1) 起止时间选择</font>**

        * **<font style="color:rgba(0, 0, 0, 0.9);">联合概率分布</font>**<font style="color:rgba(0, 0, 0, 0.9);">：计算所有有效起止组合的概率：</font><font style="color:rgba(0, 0, 0, 0.9);">P(s,e)=τ^s[s]⋅τ^e[e],约束 e>s</font>
        * **<font style="color:rgba(0, 0, 0, 0.9);">最优解</font>**<font style="color:rgba(0, 0, 0, 0.9);">：选择最大概率对应的</font><font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">(</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);">,</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">e</font>_<font style="color:rgba(0, 0, 0, 0.9);">)</font><font style="color:rgba(0, 0, 0, 0.9);">：</font><font style="color:rgba(0, 0, 0, 0.9);">(t^s,t^e)=args<emaxP(s,e)</font>

**<font style="color:rgba(0, 0, 0, 0.9);">(2) 时空管生成</font>**

        * **<font style="color:rgba(0, 0, 0, 0.9);">空间连贯性</font>**<font style="color:rgba(0, 0, 0, 0.9);">：从</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">s</font>_<font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">到</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">e</font>_<font style="color:rgba(0, 0, 0, 0.9);"> </font><font style="color:rgba(0, 0, 0, 0.9);">逐帧提取预测框</font><font style="color:rgba(0, 0, 0, 0.9);"> </font>_<font style="color:rgba(0, 0, 0, 0.9);">b</font>_<font style="color:rgba(0, 0, 0, 0.9);">^</font>_<font style="color:rgba(0, 0, 0, 0.9);">t</font>_<font style="color:rgba(0, 0, 0, 0.9);">。</font>
        * **<font style="color:rgba(0, 0, 0, 0.9);">示例</font>**<font style="color:rgba(0, 0, 0, 0.9);">：  
</font><font style="color:rgba(0, 0, 0, 0.9);">若输入查询为“成人抓球”，输出可能为：</font>
            + <font style="color:rgba(0, 0, 0, 0.9);">起止时间：第3帧（抬手）→第8帧（收回）</font>
            + <font style="color:rgba(0, 0, 0, 0.9);">空间轨迹：手部边界框连续移动并包围球。</font>

  
 

### <font style="color:rgb(25, 27, 31);">timm 库</font>
<font style="color:rgb(25, 27, 31);">Py</font>**<font style="color:rgb(25, 27, 31);">T</font>**<font style="color:rgb(25, 27, 31);">orch</font>**<font style="color:rgb(25, 27, 31);">Im</font>**<font style="color:rgb(25, 27, 31);">age</font>**<font style="color:rgb(25, 27, 31);">M</font>**<font style="color:rgb(25, 27, 31);">odels，简称 timm，是一个巨大的 </font>**<font style="color:rgb(25, 27, 31);">PyTorch </font>**<font style="color:rgb(25, 27, 31);">代码集合</font>

<font style="color:rgb(25, 27, 31);">包括了一系列：</font>

+ **<font style="color:rgb(25, 27, 31);">image models</font>**
+ **<font style="color:rgb(25, 27, 31);">layers</font>**
+ **<font style="color:rgb(25, 27, 31);">utilities</font>**
+ **<font style="color:rgb(25, 27, 31);">optimizers</font>**
+ **<font style="color:rgb(25, 27, 31);">schedulers</font>**
+ **<font style="color:rgb(25, 27, 31);">data-loaders / augmentations</font>**
+ **<font style="color:rgb(25, 27, 31);">training / validation scripts</font>**

<font style="color:rgb(25, 27, 31);">旨在将各种 SOTA 模型整合在一起，并具有复现 ImageNet 训练结果的能力。</font>

**<font style="color:rgb(25, 27, 31);">timm 库</font>**<font style="color:rgb(25, 27, 31);">实现了</font>**<font style="color:rgb(25, 27, 31);">最新的</font>**<font style="color:rgb(25, 27, 31);">几乎</font>**<font style="color:rgb(25, 27, 31);">所有的具有影响力</font>**<font style="color:rgb(25, 27, 31);">的</font>**<font style="color:rgb(25, 27, 31);">视觉</font>**<font style="color:rgb(25, 27, 31);">模型，它不仅提供了模型的权重，还提供了一个很棒的</font>**<font style="color:rgb(25, 27, 31);">分布式训练</font>**<font style="color:rgb(25, 27, 31);">和</font>**<font style="color:rgb(25, 27, 31);">评估</font>**<font style="color:rgb(25, 27, 31);">的</font>**<font style="color:rgb(25, 27, 31);">代码框架</font>**<font style="color:rgb(25, 27, 31);">，方便后人开发。</font>

## <font style="color:rgb(25, 27, 31);">代码解读:</font>
### 
##
