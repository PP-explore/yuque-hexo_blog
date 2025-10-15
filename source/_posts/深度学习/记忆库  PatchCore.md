---
title: 记忆库  PatchCore
date: '2025-09-03 14:33:03'
updated: '2025-10-15 14:34:30'
---
![](/images/1fb3087ae48e21ec8dd000b1da24b9a3.png)

PatchCore的核心思想是把一幅图像切分为一系列的小块(patch)，然后根据一定的计算规则，计算出每个patch的异常值(可以理解为偏离正常图像的程度，偏离越大，异常值越大)，然后对每个patch的异常值**进行bilinear插值**，便可得到原图像每个像素点的异常值，再根据对异常值设定的域值，超过该域值的像素点，便认为是异常点，最终便可得到所有异常像素点。

PatchCore需要提前预处理某场景正样本，便可对某场景的异常值进行检测，换句话说，PatchCore只支持对已知场景进行异常检测，如果输入未知场景的图像，检测效果是未知的，一般效果不会太好，因为PatchCore的原理是学习场景的正样本，然后拿负样本和正样本进行比对，从而找出正负样本不一样的地方，所以如果输入未知场景的负样本，由于没有对应场景的正样本，模型自然也无法知道图像的哪些地方是异常部分。



## 代码方法流程参考
初始化与加载（load方法）  
•backbone：预训练的特征提取网络（如WideResNet）。

•layers_to_extract_from：从backbone的哪些层提取特征。

•patchsize和patchstride：定义局部块的大小和步长。

•特征聚合：使用NetworkFeatureAggregator从指定层提取特征。

•预处理（Preprocessing）：对特征进行降维（使用随机投影或其它方法）到pretrain_embed_dimension。

•适配聚合（Aggregator）：进一步将特征降维到target_embed_dimension。

•异常打分器（NearestNeighbourScorer）：基于最近邻距离计算异常分数。

•异常分割器（RescaleSegmentor）：将块级别的异常分数上采样到原图大小，生成异常分割掩码。



特征提取（_embed方法）  
•输入图像通过backbone提取多尺度特征。

•使用PatchMaker将每个特征图分割成局部块。

•对不同层的特征进行尺寸调整（插值）以对齐空间尺寸（统一到第一层的块数量）。

•将块特征进行预处理（降维）和适配聚合（进一步降维），得到每个块的特征向量。

训练阶段（_fill_memory_bank）  
•遍历训练数据（正常样本），提取特征（调用_embed）。

•将所有正常样本的特征块合并成一个大的特征矩阵（内存库）。

•使用featuresampler（如K-Center采样）对特征矩阵进行采样，减少内存库的大小。

•将采样后的特征存入异常打分器（最近邻搜索模块）中，用于后续的最近邻搜索。



测试阶段（_predict）  
•提取测试图像的特征（包括块的空间信息）。

•使用异常打分器计算每个块的异常分数：计算测试特征与内存库中特征的最近邻距离。

•将块级别的异常分数转换为图像级别的异常分数（取最大值）和分割掩码：

•图像级别分数：所有块分数的最大值。

•分割掩码：将块级别的分数图通过插值恢复到原图大小。



关键点  
•局部块特征：将特征图分割成小块，每个块的特征表示局部区域的上下文。

•多尺度特征融合：从不同层提取特征，并调整到相同空间尺寸后进行拼接。

•内存库：存储正常样本的特征块，用于与测试特征进行最近邻比较。

•采样：使用采样方法（如K-Center）减少内存库的大小，同时保持其代表性。

## PatchCore包含训练和推理两个阶段
### 训练阶段
![](/images/3e9141d7e681c6b553a075d40666bb95.png)

patchcore训练框架,先经由Pretrained Encoder 特征提取, 而且只提取中间几层特征(其目的是排除浅层通用特征和深层不适合工业检测的特诊)

然后将中间层特征进行聚合操作得到一个28*28的patch图像,共计28*28=784个patch向量,然后提取若干Patch向量构造coreset代表整个patch图像,coreset存入MemoryBank(内存向量数据库)

#### Pretrained Encoder特征提取
可以变更backbone来提取特征.

特征融合部分

![](/images/e4732869d31e032a1157f31c8df62d16.png)

先对**<font style="color:#DF2A3F;">特征图内</font>**的特征进行融合

layer2 层的28*28*512的特征图，会经由kernel size为3*3的unfold核逐一抽取[3, 3, 512]区域内的特征，然后flatten为3*3*512的向量，3*3*512的向量进一步做AdaptiveAvgPool的均值化操作，从而将[3, 3, 512]区域内的特征融合为一个3*3*output_dim的向量，最后对3*3*output_dim向量进行fold操作，还原回28*28中的一个特征点，该特征点的深度为output_dim。

layer3的特征层(1024, 14, 14)要进一步进行bilinear插值,得到(1024, 3, 3, 28, 28)的形状,



![](/images/8d84bfc8c1801ab943d73627ea9cb4fb.jpeg)

**特征层间的特征融合**

更深层的中间层，先双向线性插值对齐第一个中间层的尺寸，然后各特征层经concat结合为一个特征层，该特征层再经由AdaptiveAvgPool对所有特征层进行融合，即得到patch特征的最终表达。

所以就是对特征进行自适应均匀池化,拉平之后都缩放到1024,得到 (784,1024)

遍历完所有的测试图片，得到的features为(193648, 1024)即(784×247, 1024<font style="color:rgb(221, 212, 202);background-color:rgb(29, 31, 32);">)</font>

![](/images/3a382f58360432adb5d8a70f25a9d15c.png)

### <font style="background-color:rgba(255, 255, 255, 0);">CoreSet降采样</font>
从原始的patch特征向量集中，挑选出一定比例(作为超参数)的patch特征向量，用这些挑选出来的部分patch特征向量，代表原来的patch特征向量全集，从而提高了计算效率。

经过一个全连接层将通道数减少至128，然后就是贪心策略的采样

贪心算法产生coreset

![](/images/a68429bff2a718248007205a37551021.jpeg)

初始阶段，随机选取10个index，通过如下代码计算所有特征点关于这十个点的欧式距离，得到集合D(193648, 10)，接着取均值得到平均距离集合d1(193648, 1)，选取值最大的那个值作为Mc的一个点x，接着计算x关于所有点的欧式距离d2(193648, 1)，将d1和d2进行对应位运算，保留较小的值，从而得到d3，最后取d3中最大的值加入到Mc，并成为新的x，一直循环193648*0.1次。

```java
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

```

计算:![](/images/0b5d4618c5d5a3f3af02ba6d3f418b50.png)

### 检测和定位
一幅测试图像经由Encoder提取出其patch特征后，经由faiss的KNN搜索功能后，找到每个patch与其距离最近的Memory Bank中的点，该距离作为每个patch的异常值，这些patch异常值中的最大者，即Image Level的异常值，如果异常值超过一定范围，说明图像中真的有异常，然后可通过segment操作，基于patch异常值，做bilinear插值，得到一张和原图尺寸一样的异常分布图像，对该异常分布图像进行域值处理，便可得到分割异常的mask图像。

![](/images/490cf5cdeb4368017a3600fa16e91f14.jpeg)

patches与Memory Bank中的向量进行KNN比对，以patch最近的邻居的Euclidean距离作为patch的异常值，便得到每个patch的异常值，所有patch异常值中的最大者，即为Image Level的异常值，图右侧Patches异常值矩阵，其Image Level的异常值为9.3。

![](/images/0d245d3119d4e506f3d929c2aa873df2.png)

![](/images/a4bb75052477b5de06a8bacaca8f79d0.gif)

