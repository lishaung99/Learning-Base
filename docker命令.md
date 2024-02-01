# Docker命令

## 完整使用

```
步骤
1 创建 
docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
2. 退出 exit
3.再次进入docker exec -it <容器名称或容器ID> bash
注意：首次退出后需要启动docker
命令 docker restart 容器名
4.删除 docker rm 容器名
注意：删除前需要停止容器
docker stop 容器名
查看所有容器  docker ps -a
```

## 配置

配置docker开发环境

```
#如果是首次使用Docker, 可执行下述命令进行安装和配置(仅首次执行):
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
#从 DockerHub https://hub.docker.com/r/sophgo/tpuc_dev 下载所需的镜像
docker pull sophgo/tpuc_dev:latest
```

## 创建容器

确保安装包在当前目录, 然后在当前目录创建容器如下

**简单创建**

```
docker run -it --name llama-gpu llama-gpu /bin/bash
```

**加入 -m   3000    为创建多少内存的docker**

```
docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
# myname只是举个名字的例子, 请指定成自己想要的容器的名字
```

```
docker run -v $PWD/..:/workspace -p 8001:8001 -it sophgo/tpuc_dev:latest
```

两种都可以

## 查看

查看docker内存大小

```
ubuntu@sc7-fp300:~$ docker exec -it new bash
root@9dd72b08e055:/workspace# cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
100000
root@9dd72b08e055:/workspace# cat /sys/fs/cgroup/memory/memory.limit_in_bytes
9223372036854771712
root@9dd72b08e055:/workspace# exit
exit
ubuntu@sc7-fp300:~$ cat /sys/fs/cgroup/memory/memory.limit_in_bytes
9223372036854771712
```

查看已创建docker数量

```markdown
docker ps -aq
docker ps 命令用于列出当前正在运行的容器
docker ps -a: 所有容器
 -a ：列出所有容器，
 -q ：仅显示容器的 ID。这个命令会输出所有容器的 ID。
 |：管道符号，将第一个命令的输出传递给下一个命令。
 wc -l：wc 命令用于计算输出中的行数，选项 -l 表示只计算行数。这个命令会计算输入的行数，并输出结果。
```

## 删除

删除docker容器

```
docker rm <容器名称或容器ID> 
可以以空格的方式 删除多个容器
注:
要删除一个容器，需要提供容器的名称或容器的 ID。
在删除容器之前，请确保容器已经停止运行，如果容器正在运行，先停止它。
可以使用下面的命令停止容器：
docker stop <容器名称或容器>
删除后无法恢复
```

## 重启

重新启动容器

```
docker start <容器名称或容器ID>
```

## 进入

再次进入同一个docker

```
docker exec -it <容器名称或容器ID> bash

上述命令会进入指定的 Docker 容器，并提供一个交互式的 Bash 终端；
只有已经在运行状态的容器才能进入。如果容器未运行，你需要先启动它，然后再执行进入容器的命令。
```



![image-20231228101420152](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231228101420152.png)

![image-20231228101430129](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231228101430129.png)
