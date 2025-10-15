---
title: MyBatisPlus
date: '2025-10-10 20:01:46'
updated: '2025-10-14 14:22:55'
---
![](/images/9ba880b1e677672a68567ec84e659573.png)![](/images/38920f2ecf3474bb032f2fff3de8c0a5.png)

![](/images/fd833d2c258302efbb6b8022add87d99.png)![](/images/bbc6377eb4ec4fcd4a62d286da091ebe.png)

布尔类型的字段由于变量名开头为is mybatis内部会忽略is而把变量名变为married，所有要手动添加注解指定表中的字段名字

![](/images/f259cf921b4b6a6a1cb6af1d8d945a2c.png)

![](/images/6555ba9a8f58a3a1b8dda71d363e620b.png)![](/images/272874d0c7f7ef76e9e8511a1c7ba6de.png)



## 功能
### 条件构造器
主要就是封装了mysql中的语句和条件判断

![](/images/b41df5941c0065e559d5264465733029.png)

![](/images/655ac1fa377151420794e1e2b33e9ff0.png)![](/images/772863790d9b54a6b89cce43720dd9d5.png)![](/images/156c7af997818195ecf604dbadae8885.png)![](/images/a5c65b5392b5ef97efc1c373eb5626e7.png)

使用示例：

![](/images/96cc31be00fa7621d18e9dc1062c3e42.png)

![](/images/8b5e0ccb77dd1771bb5a5912caf8887d.png)

使用LambdaQueryWrapper避免硬编码：

![](/images/ee1f4ad2d37235ba11371fe767bf94c0.png)

![](/images/2cb9dee029a9e668e2891b30b4cbc776.png)

### 自定义SQL
mp目的是用来简化操作的，但是完全使用mp拼接sql违背企业开发规范，导致sql硬编码到业务逻辑中。

![](/images/34c297c0a6e5240476b571c027c9ed2d.png)

### service接口
接口中实现了更丰富的增删改查的方法：

![](/images/cd8ed802010d8946453b4b119007e152.png)

![](/images/26770b26d4c48e447786ec8ddc27668e.png)

![](/images/b106b05852df8812d24c72c1f64bcb68.png)

