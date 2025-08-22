---
title: 知乎Transformer文章阅读记录
date: '2025-01-02 20:27:37'
updated: '2025-08-22 16:20:09'
---
<font style="color:#000000;">三万字最全解析！从零实现Transformer: </font>[https://zhuanlan.zhihu.com/p/648127076](https://zhuanlan.zhihu.com/p/648127076)



#  关于"计算attention时容易出现NaN"：![](/images/32cd00dbe2f9ac7767faa4fc26e94a8c.png)
**Attention机制中Softmax的工作原理**： 在计算Attention时，通常会使用Softmax函数来将注意力分布标准化，使得每一行的注意力权重和为1。但是，Softmax计算的输入是加权得分（score），如果输入的一整行都被mask（即所有值都被设置为无穷小或负无穷大），那么Softmax的分母（归一化项）会变成0。这会导致在计算注意力权重时产生NaN（Not a Number），因为Softmax的输出是数值除以分母，分母为0时会出现非法操作。  ![](/images/f7bb0f75747f6e24851c5375d8a888a6.png)![](/images/c8e115c4ed3901c03f9be8d478e54045.png)





![](/images/728b8150021bd54506ba8b9264d30c58.png)![](/images/34864c91d5ea8689ff37c03ea9d4d12a.png)![](/images/818551c4380efb13d4af26d14df412fc.png)



![](/images/77fc258fbcc673800536d22b7b703fb7.png)

![](/images/45f6835c238a56e354f4846afe4086e9.png)

![](/images/0c57384d53dd717501829146c35e7255.png)

这里causal mask就是对self attention中QK内积之后必定输出为n*n方阵， ，Causal Mask 是在计算注意力分数矩阵 QK^T 后，执行 Softmax 之前引入的。它的作用是通过屏蔽未来位置的信息，确保当前时间步的预测只能依赖于已知的过去和当前的信息。  ![](/images/8bb07aa29a9fc32fd8b440069bdcc56b.png)

