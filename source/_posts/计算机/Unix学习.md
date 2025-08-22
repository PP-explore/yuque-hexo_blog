---
title: Unix学习
date: '2025-08-21 16:27:02'
updated: '2025-08-22 20:35:43'
categories:
  - 计算机
tags:
  - 计算机基础
cover: /images/custom-cover.jpg
recommend: true
---
```plain
1. tar 命令
语法： tar [主选项 + 辅选项] 文件或目录

-z : 使用 gzip 来压缩和解压文件

-v : --verbose 详细的列出处理的文件

-f : --file=ARCHIVE 使用档案文件或设备，这个选项通常是必选的

-c : --create 创建一个新的归档（压缩包）

-x : 从压缩包中解出文件
-j: 解压 bzip2 格式
-J: 解压 xz 格式
解压缩指令
解压.tgz文件：
（1）解压到当前文件夹
      命令：tar zxvf  文件名.tgz -C ./     
      eg:tar zxvf demo.tgz -C ./
（2）解压到指定文件夹下
     命令：tar  zxvf  文件名.tgz  -C /指定路径
     eg:解压到家目录：tar zxvf simple-examples.tgz -C /Home

tar -xzvf archive.tar.gz # 解压 .tar.gz 格式的文件
 
tar -xjvf archive.tar.bz2 # 解压 .tar.bz2 格式的文件
 
tar -xJvf archive.tar.xz # 解压 .tar.xz 格式的文件
 
tar -xvf archive.tar # 解压 .tar 格式的文件
压缩指令：

tar -czvf archive.tar.gz /path/to/directory # 压缩目录为 .tar.gz 格式的文件
 
tar -czvf archive.tar.gz file1 file2 file3 # 压缩多个文件为 .tar.gz 格式的文件
 
tar -cjvf archive.tar.bz2 /path/to/directory # 压缩目录为 .tar.bz2 格式的文件
 
tar -cjvf archive.tar.bz2 file1 file2 file3 # 压缩多个文件为 .tar.bz2 格式的文件
 
tar -cJvf archive.tar.xz /path/to/directory # 压缩目录为 .tar.xz 格式的文件
 
tar -cJvf archive.tar.xz file1 file2 file3 # 压缩多个文件为 .tar.xz 格式的文件

```

```python
curl指令
最常见的curl指令
1.发送POST请求
curl -X POST -d 'a=1&b=nihao' URL
2.发送json格式请求：
curl -H "Content-Type: application/json" -X POST -d '{"abc":123,"bcd":"nihao"}' URL
curl -H "Content-Type: application/json" -X POST -d @test.json URL
'其中，-H代表header头，-X是指定什么类型请求(POST/GET/HEAD/DELETE/PUT/PATCH)'
'-d代表传输什么数据'
3.下载
curl -O http://www.linux.com/dodo1.JPG
curl -O http://www.linux.com/dodo[1-5].JPG #循环下载
'-O:将返回内容输出到当前目录下，和url中文件名相同的文件中（不含目录）'
-L：跟随重定向（因为 某些链接可能会有重定向）。

详细参数和教程：
https://blog.csdn.net/angle_chen123/article/details/120675472

```

## bash操作


[Linux下shell脚本：bash的介绍和使用（详细）-CSDN博客](https://blog.csdn.net/weixin_42432281/article/details/88392219)

[shell中的case语句详解_shell case-CSDN博客](https://blog.csdn.net/qq_36417677/article/details/104395344)

[Bash Shell - shell 编程 | shift 命令用法笔记 - 《技术私房菜》 - 极客文档](https://geekdaxue.co/read/shenweiyan@cookbook/shell-shift-note)



![](/images/1edb393582eeed55812e6ec92f9b89ef.png)

![](/images/beafc5b65a5381f3e4fd67a5b1605d05.png)

![](/images/dc0b58110378472d30cc0edf9a06277a.png)

![](/images/d304517a44a4ecea0ff7521ae6d96ce0.png)

![](/images/30607917ede54341644f88ccb62f77ed.png)![](/images/aa104b9a9f816c29fb80f9784a9cf21a.png)

![](/images/52e5c9b006cca7abebba4d871e64b666.png)

![](/images/43b3a1059a54d1cd890a9dad979b83e3.png)

![](/images/1aa2429e0082b86b8596c2be0d27a263.png)

![](/images/6db28764d5f7685e87dee3228995a5a5.png)![](/images/320e02e5034bf20c12790d79fcda4935.png)

![](/images/e3c03969d8d513b91dd83a28de117340.png)

## 文件操作
![](/images/94ae360f6e288768faa8ee91020ca78d.png)

![](/images/85e34f5c6f21139c6e6bd4e2614ab7f1.png)

![](/images/e91cc0d1bd8598448d6fbeca716ea5d7.png)

![](/images/a7295cd47a8461c2cdc27d2f42306488.png)

![](/images/173778a8c20a9974e1878933765001d1.png)

![](/images/568be40c41de98f025e02ce238541873.png)

![](/images/6cd71d6941ceec42fb485d0664c202e6.png)

### <font style="color:rgb(77, 77, 77);">ls    目录列表</font>
![](/images/5779f6db105f0b02715ad384256e68c0.png)

![](/images/ce123ae35ee9e648b2e9542a4af188c2.png)

![](/images/9a47559075b6e82503e94531278a0759.png)![](/images/14df6042310642cc1b055c59e72af44b.png)

![](/images/c7d95493372a18e420abbe111fe21b31.png)

![](/images/56f3da3c67e07f6bcb89a8ff40c70bbf.png)![](/images/d3b880d336d004329a2bbd8f8c19e65a.png)

![](/images/fd316eb540b2e9e64808b6017372e843.png)

<font style="color:rgb(77, 77, 77);">ls最常用的参数有四个： </font>**<font style="color:#DF2A3F;">-a -l -F -s</font>**<font style="color:rgb(77, 77, 77);">。</font><font style="color:rgb(77, 77, 77);">  
</font><font style="color:rgb(77, 77, 77);">ls -a     ls  --all</font>

<font style="color:rgb(77, 77, 77);">说明：Linux上的文件以“.”开头的文件被系统视为隐藏文件，仅用ls命令是看不到他们的，而用ls -a除了显示 一般文件名外，连隐藏文件也会显示出来。</font>

<font style="color:rgb(51, 51, 51);">ls -l（这个参数是字母L的小写，不是数字1）</font>  
<font style="color:rgb(51, 51, 51);">说明：这个命令可以</font><font style="color:#DF2A3F;">使用长格式显示文件内容</font><font style="color:rgb(51, 51, 51);">，如果需要察看更详细的文件资料，就要用到ls -l这个指令。</font>![](/images/ab4828926000c41aefe9e450be92d90f.png)

### cat 显示文件内容
### rm      删除文件
![](/images/000390ea3e4956c3b33cdf0a7bdc75cc.png)

![](/images/df8273d4cbdc2f4a09b7444794ea5879.png)

![](/images/0fc4d1f3d574dffdd4f0896de460a787.png)

### 复制文件：cp    
移动文件：mv
![](/images/6ca9cb0510a2461ac2c5f6e61888655c.png)

![](/images/0c71c0f644e2a2efa0908869431b9dc6.png)

![](/images/bcbc6f95ebd363d8268aeeae9c29b720.png)













### shell重定向
![](/images/ed4de2684ab28ca5995df3333050704a.png)

![](/images/f20ca2c6d83202dfd883197041a61746.png)

![](/images/7c2abe316d0fce56a7b96b3b6505c0c9.png)

![](/images/9f0bd17c978627decee5da819fc02b94.png)

![](/images/b0668f819f35c5aa04839e7fde9257be.png)

![](/images/898b8531a8604d768afe5cf7aca9e2e5.png)![](/images/fa8f3610d08f9d0197b4734d1c0af88c.png)![](/images/1b172a419755f72d5fd600aa8e1e7bef.png)

### 文件操作
![](/images/1b4b3824a86eecf3be4c5c6fde189d02.png)

![](/images/481626687d340f001bace67f639ae95a.png)![](/images/8b8cc26a3960badfbe6e38612e6f9285.png)![](/images/620b5bf97737ed877d651d21119d1222.png)![](/images/0ddae9ec361690c18c4ceef5a95ecf52.png)![](/images/d7f82aa693602d90fc5b47f86e241fb8.png)![](/images/2fdd2f7528a72a82d46f0f98555fb295.png)![](/images/9c512dc980c46779cfd865d48fc2bed8.png)



![](/images/0f8a1f43961e95b9768c126b47faf562.png)

![](/images/8e0693ec415f1078c148ac2260408f0f.png)



![](/images/fcf7cf3bf5c55cfaab83c718063059ab.png)

![](/images/e0d025415c329e0c3f563d95fa8e3576.png)



### 文件系统
![](/images/72e8c9eaed4e57ea3d34ba7a2d76f69d.png)

![](/images/ac7d298b8c8b555a983dd79487353d6e.png)

[超级块，i节点，数据块，目录块，间接块_超级块是什么-CSDN博客](https://blog.csdn.net/ds1130071727/article/details/89409426#:~:text=%E5%9C%A8i-%E8%8A%82%E7%82%B9%E7%9A%84%E7%AC%AC11%E9%A1%B9)

![](/images/0c13bf7a18ac53e861f09b68b27c1b21.gif)![](/images/02a5b6b2776ac7c168e5572eeb23a70b.png)![](/images/c40e35603f693179438e852915d1d44e.png)

![](/images/48fe603c12db0f4fced621c63d9090d8.png)



也就是说一个目录里面存着目录下的文件名列表，查找先在当前目录的文件列表里找文件名，文件名可得对应的i节点号，然后从i节点列表得到相应i节点记录，i节点记录文件信息和数据块号。<font style="color:rgb(243, 59, 69);">内核将文件的入口(47,filename)添加到目录文件里</font>

<font style="color:rgb(243, 59, 69);">目录其实也是文件，只是它的内容比较特殊：包含文件名字列表，列表一般包含两个部分：i-节点号和文件名</font><font style="color:rgb(77, 77, 77);">。所以它的创建过程和文件创建过程一样，只是第二步写的内容不同。</font><font style="color:rgb(243, 59, 69);">一个目录创建时至少包括两个链接：“.”，“..”</font>

![](/images/d4c1e9f5c94b8d6287d96735195ad65c.png)

### UNIX shell
![](/images/0ae940785eeabf97184fdb1b206fbc6a.png)



![](/images/87d181f6de4978236db68c96dcf6f673.png)![](/images/fa47efb725860d186dba9101a9b3e79d.png)

![](/images/792090b1ec3d59ba8755e66c4df058db.png)

#### echo
![](/images/b4bbf864961ffd48b8a13aeced2f15c3.png)

![](/images/6cae45c75671bead5a535a731d0a32a7.png)

### shell变量
![](/images/c8e1c7f8e966e39443a841df988047dd.png)

![](/images/ceefee4821ec70a8d638fdb86ef8113e.png)

![](/images/dec01356161857b4f2fc9cb6a5ce1c08.png)

![](/images/24e3704e1064e0a96041c09b547c38b1.png)

![](/images/6f82d10a9b7e63414c5a594064910fec.png)

![](/images/52cb709c5170d299518b96f262c20bc3.png)

![](/images/7800e8f1dc5a0f06dbc6fa2eb409e4a6.png)![](/images/be1b82c7934ee9b7e05a63980c923fa7.png)

![](/images/85ce3dab68fcee83ba9535585e357735.png)

![](/images/bb53a1392336b7bba233c48c3254deba.png)

## 元字符
![](/images/faa10f6a955f34fdbf705d2d234f66d2.png)

![](/images/1f70e6ea6d219880d4ade2d779ea4b41.png)

![](/images/154c85dbd68f211a200daa9ec2ce43ea.png)

![](/images/b819a3fd2f22154380eb4fefffb557ee.png)

![](/images/206b9a1e7fff230fcc25efb6eba77445.png)

![](/images/62711d1b771f8b1653862cc2a56df22c.png)

#### 后台计算：&符号
![](/images/851d62069f8683f145af9f04d05232c1.png)

![](/images/2de2086b4acc1d4d02177d63ce3934a6.png)

#### 链接命令：管道操作符 |
![](/images/64442a94b3cabb6f6d7d3ee83df4fa58.png)

![](/images/061bd1630f30268de055c628905c32dc.png)

##  UNIX系统工具
![](/images/6161bf00ae5a7b521f61991539a7620b.png)

#### 显示PID
![](/images/dd58a143b8afda795bc7cdbf5cb9a2fe.png)

![](/images/c653d736c28218bb3eb32c17357373b9.png)

#### 终止进程：kill
![](/images/908066afb25d9deecca6f057fceb2bf7.png)

![](/images/edb01c4ea91681d17e5bf7e6ce3768b4.png)

#### 文件搜索：grep
![](/images/326daa792133942f114163003941ac6b.png)

![](/images/b4646593c2ce6e1136f5c07aa1d063e1.png)

![](/images/95b31b90755a5af1a42de700492e6de3.png)

![](/images/db93f26bea200c87cdf83a557b52ebcc.png)

#### 文本文件排序：sort
![](/images/8be696c9ac4abc2fb3a1f3f302ba9d5b.png)![](/images/0ce6bb750dfb68404c6eff471acd3296.png)

## 启动文件：用户策略文件
![](/images/5aed356766db50b139070a44c9af35cb.png)

![](/images/bf703c4313686f6a47c1b127380ec734.png)

![](/images/7eb293c1669fd3bf4222dfba9d41f04b.png)

#### export命令
![](/images/8ea33f8eaf6d5d1a393fac4056eff7ec.png)

## UNIX进程管理
![](/images/04c8b038ce005f5dd4b96bdc5a0a8600.png)

![](/images/154e2579c61165a4dcc8258a6befdaf7.png)![](/images/1f95400d33cf6a0aa89d2a4c5a8b9340.png)

![](/images/58e109913b7f0eb144e6882cbde69d62.png)![](/images/b9c6db379ae9c486a163675de307ae22.png)![](/images/037d2602017638feeb3bd6f82e5e316b.png)![](/images/de382ea06626833f4dc03abcc36b587e.png)![](/images/bbbed829361754dc09508bc763627977.png)

##  Shell编程
![](/images/4b715e2900b859fb6024344d1c1e2357.png)

### chmod修改文件权限
[Linux权限详解（chmod、600、644、700、711、755、777、4755、6755、7755）「建议收藏」-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2069886)

![](/images/272a0f377d7d0ec4477b69ca0bb3f500.png)

![](/images/63da681e4cdb51f3a116ed2878f5b709.png)![](/images/18338f7e43f15fb7db1ee025dc03eb3e.png) ![](/images/bd2cc42a8a7304521336c46bd76657ad.png)

### dot命令
![](/images/5b96b32a538d102ea0c15c481fff6f4d.png)

### 读取输入：read
![](/images/cb0afe52bfb4c18a44428e86ea6e1d8a.png)![](/images/138c1874e65302e22469fab35cc4f00c.png)

### 创建变量
![](/images/eeba8cbe590f2dafc65a665b82d5d99a.png)![](/images/398e99932dda735b9c6f6e89f83b8dbd.png)![](/images/3aabf427dadcf6ba87e59c4be6ff15ed.png)

### 命令行参数
![](/images/10711042947b6e66be415fc4f20f761e.png)![](/images/9c05803a239222c1d70568da8e7f42d1.png)![](/images/7fda9d114a3ae3077ca1840cdcd579aa.png)

![](/images/5b3fcd7b98782774f3b95bcc76c6c9e4.png)

### 终止程序：exit命令
![](/images/096c87fa6fa56fa00e4e6831633fde75.png)

### 条件和试验（判断语句）
![](/images/03f16df26fb1241d536dbbf68c64a665.png)![](/images/61f59fee41496b707708e30dcb839d23.png)![](/images/81d754d8a8b5dea3b3506d1e37ee492e.png)

#### 测试命令：test
![](/images/06e71dea869d3c31d12565b6738e0a85.png)![](/images/b576e88a235a4454d92e24d505e68754.png)

数值：

![](/images/3e9c2caae3db2096f4e738f4451140de.png)	串值：

![](/images/9265dbba42ec1e0c4b739800da008133.png)![](/images/3b91462c4596c34cd59847515b58e0f4.png)

文件

![](/images/9d5bd5662506f4e2f181f987c8355376.png)

### 参数替换
![](/images/a87eb09f66fc1e86dd50578fd47a03b3.png)![](/images/2d81ee5b85be06a849f71820fef6e3f4.png)![](/images/23b138284c06d1a724fc24ec7932c4d8.png)![](/images/d0c852f146029798ae96256ed79a5bd6.png)

### 算术运算命令
![](/images/8271d4f6397a94314a882b695346a5a6.png)



## 
##
