---
title: Docker
date: '2024-12-16 19:53:33'
updated: '2025-08-22 20:50:01'
categories:
  - 技术杂文
tags: null
cover: /images/custom-cover.jpg
recommend: true
---
| **<font style="color:rgb(212, 208, 201);">概念</font>** | **<font style="color:rgb(212, 208, 201);">说明</font>** |
| --- | --- |
| <font style="color:rgb(212, 208, 201);">Docker 镜像(Images)</font> | <font style="color:rgb(212, 208, 201);">Docker 镜像是用于创建 Docker 容器的模板，比如 Ubuntu 系统。</font> |
| <font style="color:rgb(212, 208, 201);">Docker 容器(Container)</font> | <font style="color:rgb(212, 208, 201);">容器是独立运行的一个或一组应用，是镜像运行时的实体。</font> |
| <font style="color:rgb(212, 208, 201);">Docker Registry</font> | <font style="color:rgb(212, 208, 201);">Docker 仓库用来保存镜像，可以理解为代码控制中的代码仓库。Docker Hub(</font>[<font style="color:rgb(212, 208, 201);">https://hub.docker.com</font>](https://hub.docker.com/)<br/><font style="color:rgb(212, 208, 201);">) 提供了庞大的镜像集合供使用。一个 Docker Registry 中可以包含多个仓库（Repository）；每个仓库可以包含多个标签（Tag）；每个标签对应一个镜像。通常，一个仓库会包含同一个软件不同版本的镜像，而标签就常用于对应该软件的各个版本。我们可以通过 <仓库名>:<标签> 的格式来指定具体是这个软件哪个版本的镜像。如果不给出标签，将以 latest 作为默认标签。</font> |
| <font style="color:rgb(212, 208, 201);">Docker 主机(Host)</font> | <font style="color:rgb(212, 208, 201);">一个物理或者虚拟的机器用于执行 Docker 守护进程和容器。</font> |
| <font style="color:rgb(212, 208, 201);">Docker 客户端(Client)</font> | <font style="color:rgb(212, 208, 201);">Docker 客户端通过命令行或者其他工具使用 Docker SDK (</font>[<font style="color:rgb(212, 208, 201);">https://docs.docker.com/develop/sdk/</font>](https://docs.docker.com/develop/sdk/)<br/><font style="color:rgb(212, 208, 201);">) 与 Docker 的守护进程通信</font> |
| <font style="color:rgb(212, 208, 201);">Docker Machine</font> | <font style="color:rgb(212, 208, 201);">Docker Machine是一个简化Docker安装的命令行工具，通过一个简单的命令行即可在相应的平台上安装Docker，比如VirtualBox、 Digital Ocean、Microsoft Azure。</font> |


![](/images/cfec25a5301bfac580817c27326898c3.webp)

`docker images`列出本地主机上的镜像

![](/images/02dca1d5bcbc3cb987f026749d10e859.webp)![](/images/cf93a689a1e0c21cd5597bd509d4ebd5.png)

`docker search 某个XXX镜像名字`查询某个镜像

`docker pull 某个XXX镜像名字`下载某个镜像

```python
//默认下载最新的redis
docker pull redis
//下载指定的redis版本
docker pull redis:4.0.1

```

`docker rmi 某个XXX镜像名字ID`删除某个下载的镜像

## 容器命令
`docker run [OPTIONS]  IMAGE  [COMMAND]  [ARG...]`

:::info
OPTIONS说明（常用）：

 --name=“容器新名字”: 为容器指定一个名称

-d: 后台运行容器，并返回容器ID，也即启动守护式容器；

-i：以交互模式运行容器，通常与 -t 同时使用；

-t：为容器重新分配一个伪输入终端，通常与 -i 同时使用；

-P: 随机端口映射；

-p: 指定端口映射，有以下四种格式 ip:hostPort:containerPort ip::containerPort hostPort:containerPort containerPort

`docker stop 容器ID或者容器名`<font style="color:rgb(24, 24, 24) !important;">停止容器</font>

:::

`<font style="color:rgb(56, 58, 66);background-color:rgb(246, 248, 250);">docker </font><font style="color:rgb(193, 132, 1);background-color:rgb(246, 248, 250);">rm</font>`<font style="color:rgb(56, 58, 66);background-color:rgb(246, 248, 250);"> 容器ID</font><font style="color:rgb(24, 24, 24) !important;">删除停止容器</font>

`<font style="color:rgb(56, 58, 66);background-color:rgb(246, 248, 250);">docker stop</font>`<font style="color:rgb(56, 58, 66);background-color:rgb(246, 248, 250);"> 容器ID或者容器名</font>

`<font style="color:rgb(56, 58, 66);background-color:rgb(246, 248, 250);">docker </font><font style="color:rgb(1, 132, 187);background-color:rgb(246, 248, 250);">start</font><font style="color:rgb(56, 58, 66);background-color:rgb(246, 248, 250);"> 容器ID或者容器名</font>`<font style="color:rgb(56, 58, 66);background-color:rgb(246, 248, 250);">  </font><font style="color:rgb(36, 41, 46);">前提是已经根据镜像创建过容器，只不过创建的容器暂时未启动</font>

<font style="color:rgb(24, 24, 24) !important;">进入已运行的容器：</font>

**<font style="color:rgb(36, 41, 46);">方式一：</font>**<font style="color:rgb(24, 24, 24) !important;">  
</font><font style="color:rgb(36, 41, 46);">exec 是在容器中打开新的终端，并且可以启动新的进程，因为是新终端，用exit退出，不会导致容器的停止。</font>

`<font style="color:rgb(56, 58, 66);background-color:rgb(246, 248, 250);">docker </font><font style="color:rgb(193, 132, 1);background-color:rgb(246, 248, 250);">exec</font><font style="color:rgb(56, 58, 66);background-color:rgb(246, 248, 250);"> -it 容器ID /bin/bash</font>`

**<font style="color:rgb(36, 41, 46);">方式二：</font>**

<font style="color:rgb(36, 41, 46);">attach 直接进入容器启动命令的终端，不会启动新的进程。如果进入前台交互式启动容器,，</font>**<font style="color:rgb(36, 41, 46);">用exit退出，会导致容器的停止</font>**<font style="color:rgb(36, 41, 46);">。</font>**<font style="color:rgb(36, 41, 46);">用ctrl+p+q，不会导致容器停止</font>**<font style="color:rgb(36, 41, 46);">。如果进入后台守护式启动容器，不仅无法进行交互，并且ctrl+c会导致守护式进程停止。</font>

`<font style="color:rgb(56, 58, 66);background-color:rgb(246, 248, 250);"> docker attach 容器ID</font>`

:::info
<font style="color:rgb(106, 115, 125);">提示：工作中我们一般使用 docker run -d image:tag /bin/bash启动容器，再通过docker exec -it 容器ID /bin/bash，最为安全可靠。</font>

:::

<font style="color:rgb(24, 24, 24) !important;">退出容器的方法 </font>

1. <font style="color:rgb(36, 41, 46);">exit：run进去容器，exit退出，容器停止</font>
2. <font style="color:rgb(36, 41, 46);">ctrl+p+q：run进去容器，ctrl+p+q退出，容器不停止</font>

**保存配置后的容器为新镜像**： 一旦容器的配置完成，您可以将其保存为新的镜像，以便在断网的服务器上加载和运行。

```plain

docker commit your_container_name your_new_image_name:tag
```

**保存新的镜像为 **`**.tar**`** 文件**： 将已配置的容器保存为 `.tar` 文件，以便将其迁移到断网的服务器上。

```plain
docker save -o your_image_name.tar your_image_name:tag

```

**加载 Docker 镜像**： 首先，使用 `docker load` 命令在有网络的服务器上加载 Docker 镜像。假设您已经从其他服务器或 Docker Hub 拉取了镜像，并将其保存为 `.tar` 文件。

```plain
docker load -i /path/to/your_image_name.tar
```
