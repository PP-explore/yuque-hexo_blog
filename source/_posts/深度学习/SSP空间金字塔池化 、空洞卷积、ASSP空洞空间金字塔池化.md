---
title: SSP空间金字塔池化 、空洞卷积、ASSP空洞空间金字塔池化
date: '2025-10-15 14:08:56'
updated: '2025-10-15 17:01:19'
categories:
  - 人工智能
tags:
  - 深度学习
cover: /images/custom-cover.jpg
recommend: true
---
问题：

<font style="color:rgb(77, 77, 77);">在神经网络的训练过程中，我们都需要保证数据集图片大小的一致性，可这是为什么呢？我们知道一个神经网络通常包含三个部分：卷积、池化、全连接。</font>

<font style="color:rgb(77, 77, 77);">假设给定一个30*30大小的输入图片，通过一个3*3的卷积核得到大小为29*29的输出和给定一个40*40大小的输入图片，得到大小为39*39的输出之间有区别吗？其实是没有区别的，因为在这里我们要训练的是卷积核的参数，与输入的图片大小无关。</font>

<font style="color:rgb(77, 77, 77);">再来看池化层，池化层其实可以理解成一个压缩的过程，无论是AVE还是MAX其实也输入都没啥关系，输出大小直接变为输出一半就完了（参数为2）。</font>

<font style="color:rgb(77, 77, 77);">所以问题出现在全连接层上，假设同一个池化层的输出分别是32*32*1和64*64*1，这就出问题了，因为全连接层的权重矩阵W是一个固定值，池化层的不同尺寸的输出会导致全连接层无法进行训练。</font>

<font style="color:rgb(77, 77, 77);">针对这个问题，原有的解决思路是通过拉伸或者裁剪去统一图片的尺寸，但是会造成信息丢失，失真等等众多问题。</font>

<font style="color:rgb(77, 77, 77);">所以大佬就想了个办法，将原有的神经网络处理流程从图2改变为了图3，提出了SPP结构，也就是池化金字塔，利用多尺度解决这个问题。</font>

![](/images/bd9a562bd41d995094080a1ed3c706fd.png)

### <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">SPP </font><font style="color:rgb(77, 77, 77);">Spatial Pyramid Pooling</font>
利用多个不同尺度的池化层进行特征的提取，融合成一个21维的向量输入至全连接层。

![](/images/bf414ad99a59c67d6ff5f92d92ea1c5c.png)

如图所示，从下往上看，输入图片的大小可以是任意的，经过卷积层卷积之后获取特征图的channels数量为256，将其输入到SPP结构中。图中从左往右看，分别将特征图分成了16个格子，4个格子和1个格子。

假设特征图大小是：width*height，这里蓝色格子的大小就是width/4*height/4，绿色格子的大小就是width/2*height/2，灰色格子的大小就是width*height。对每个格子分别进行池化，一般是采用MAX pooling，这样子我们分别可以得到16*256、4*256、1*256的向量，将其叠加就是21维向量，这样子就保证了无论输入图片尺寸是多少，最终经过SPP输出的尺度都是一致的，也就可以顺利地输入到全连接层。





### <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">空洞卷积:</font>
<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">空洞卷积（Dilated Convolution，也称为膨胀卷积或扩张卷积）是卷积神经网络中一种特殊的卷积操作，它通过在卷积核元素之间插入间隔（空洞）来扩大感受野，同时保持分辨率不变</font>

下图清晰的展示了一维普通卷积和空洞卷积的区别：![](/images/52e4b73cf34ff5f61c64887867c1e454.jpeg)

二维空洞卷积则为：

![](/images/5eb12c1e63f2a14f575850f4b2b77f99.gif)

<font style="color:rgb(51, 51, 51);">kernel = 3; stride = 1; pad = 0; rate = 1</font>

| <font style="color:rgba(0, 0, 0, 0.9);">特性</font> | <font style="color:rgba(0, 0, 0, 0.9);">普通卷积</font> | <font style="color:rgba(0, 0, 0, 0.9);">空洞卷积(dilation=2)</font> |
| --- | --- | --- |
| <font style="color:rgba(0, 0, 0, 0.9);">卷积核物理大小</font> | <font style="color:rgba(0, 0, 0, 0.9);">3x3</font> | <font style="color:rgba(0, 0, 0, 0.9);">3x3</font> |
| <font style="color:rgba(0, 0, 0, 0.9);">实际感受野</font> | <font style="color:rgba(0, 0, 0, 0.9);">3x3</font> | <font style="color:rgba(0, 0, 0, 0.9);">5x5（等效）</font> |
| <font style="color:rgba(0, 0, 0, 0.9);">输出分辨率</font> | <font style="color:rgba(0, 0, 0, 0.9);">可能减小（若stride>1）</font> | <font style="color:rgba(0, 0, 0, 0.9);">保持输入分辨率</font> |
| <font style="color:rgba(0, 0, 0, 0.9);">计算量</font> | <font style="color:rgba(0, 0, 0, 0.9);">标准计算量</font> | <font style="color:rgba(0, 0, 0, 0.9);">与普通卷积相同</font> |


<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">空洞卷积有两种实现方式，第一，卷积核填充0，第二，输入等间隔采样</font><font style="color:rgb(51, 51, 51);">。</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">空洞卷积作用：</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">保持高分辨率：避免过早下采样丢失细节（对密集预测任务如分割/检测关键）</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">扩大感受野</font><font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：在不增加参数量的情况下捕获更大范围上下文</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">替代池化层</font><font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：减少信息损失</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">不一定会提升所有任务效果</font><font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">对小物体检测可能不利（因细节可能被跳过）</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">空洞卷积（dilation=2, stride=1, padding=2）示例：</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">实际感受野</font><font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：5×5（等效）</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">物理卷积核</font><font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：仍为3×3，但元素间有1个间隔</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">输出尺寸</font><font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：5×5（因padding=2保持尺寸）</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">计算过程</font><font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">（以输出(2,2)位置为例）：</font>

```plain
实际参与计算的输入位置（×表示被跳过）：
[1,  ×, 3,  ×, 5]
[×,  ×, ×,  ×, ×]
[11, ×,13, ×,15]
[×,  ×, ×, ×, ×]
[21, ×,23, ×,25]

输出[2,2] = 1×1 + 3×1 + 5×1
          +11×1 +13×1 +15×1
          +21×1 +23×1 +25×1 = 117
```

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">虽然计算了9个点，但这些点实际分布在5×5区域（感受野扩大），关键输出尺寸仍为5×5！</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">空洞卷积保持分辨率的核心在于两个设计：</font>

1. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">Padding的扩展  
</font><font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">空洞卷积需要的padding量计算公式：</font>

```plain
复制
padding = dilation * (kernel_size - 1) // 2
```

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">对于3×3核，dilation=2时：  
</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">padding = 2*(3-1)/2 = 2</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">  
</font><font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">这比普通卷积的padding=1更大，确保边缘信息不被截断</font>

2. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">Stride保持为1  
</font><font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">空洞卷积通常配合</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">stride=1</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">使用，避免主动下采样</font>

| <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">机制</font> | <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">作用</font> |
| --- | --- |
| <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">增大padding</font> | <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">补偿空洞带来的边缘信息损失</font> |
| <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">stride=1</font> | <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">避免主动下采样</font> |
| <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">间隔采样</font> | <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">在更大感受野内计算，但不减少输出点的数量</font> |
| <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">参数量的保持</font> | <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">仍使用3x3卷积核的参数量，只是计算时"跳着看"</font> |


<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">通过这种设计，空洞卷积实现了"看得更广，但看得更稀疏"的效果，完美平衡了感受野和分辨率的需求。这就是它在视频处理和密集预测任务中被广泛使用的原因！</font>

<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);"></font>

### <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">ASPP </font>
<font style="color:rgb(77, 77, 77);">ASPP一开始在DeepLabv2中提出</font>

![](/images/832a96fe643c80891646be32f8f3419a.png)

ASPP模块示意图

对于给定的输入以不同采样率的空洞卷积并行采样，将得到的结果concat到一起，扩大通道数，然后再通过1*1的卷积将通道数降低到预期的数值。相当于以多个比例捕捉图像的上下文。

![](/images/b7dd0551c90a9cc5f83b70d6f2f1667f.png)

添加ASPP模块后的网络如图6所示，将Block4的输出输入到ASPP，经过多尺度的空洞卷积采样后经过池化操作，然后由1*1卷积将通道数降低至预期值。
