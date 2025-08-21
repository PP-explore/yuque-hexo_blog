---
title: LeNet-5
date: '2024-12-30 15:09:40'
updated: '2025-08-21 16:18:38'
---
<font style="color:rgb(0, 0, 0);background-color:rgb(238, 240, 244);">LRN全称为Local Response Normalization，即局部响应归一化</font>  
<font style="color:rgb(0, 0, 0);background-color:rgb(238, 240, 244);">LRN是一种深度学习中的一种归一化方法，由AlexNet卷积神经网络的原文提出并使用</font>

<font style="color:rgb(0, 0, 0);background-color:rgb(238, 240, 244);">但由于后续的多张论文，使用LRN后都没什么效果，且计算复杂，现一般都已被BN归一化方法替代</font>



conv1：输入→卷积→ReLU→局部响应归一化→重叠最大池化层

conv2：卷积→ReLU→局部响应归一化→重叠最大池化层

conv3：卷积→ReLU

conv4：卷积→ReLU

conv5：卷积→ReLU→重叠最大池化层(经过这层之后还要进行flatten展平操作)

FC1：全连接→ReLU→Dropout

FC2：全连接→ReLU→Dropout

FC3(可看作softmax层)：全连接→ReLU→Softmax

:::info
第一层：卷积层

卷积核大小11*11，输入通道数根据输入图像而定，输出通道数为96，步长为4。

池化层窗口大小为3*3，步长为2。

第二层：卷积层

卷积核大小5*5，输入通道数为96，输出通道数为256，步长为2。

池化层窗口大小为3*3，步长为2。

第三层：卷积层

卷积核大小3*3，输入通道数为256，输出通道数为384，步长为1。

第四层：卷积层

卷积核大小3*3，输入通道数为384，输出通道数为384，步长为1。

第五层：卷积层

卷积核大小3*3，输入通道数为384，输出通道数为256，步长为1。

池化层窗口大小为3*3，步长为2。

第六层：全连接层

输入大小为上一层的输出，输出大小为4096。

Dropout概率为0.5。

第七层：全连接层

输入大小为4096，输出大小为4096。

Dropout概率为0.5。

第八层：全连接层

输入大小为4096，输出大小为分类数。

注意：需要注意一点，5个卷积层中前2个卷积层后面都会紧跟一个池化层，而第3、4层卷积层后面没有池化层，而是连续3、4、5层三个卷积层后才加入一个池化层。

:::

![](/images/5c248e32c3aeb726fc455c6b168bd5de.png)![](/images/261e997e20eaf9c5e64b42eaf1bc62cb.png)![](/images/1ef973c142967859af13a5af19630eb6.png)![](/images/537e3829282885773d380e69f34343cd.jpeg)

MaxPooling没有可学习的参数![](/images/d99731866f9e7f2b26f383c44477b30f.png)![](/images/afa17b9bc5fb540a9f2a273d34986394.png)![](/images/4e937b705f982700cf0c70ed34214a07.png)

