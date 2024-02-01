# SC7 FP300安装及环境配置（Linux）

## 注意

- 在pcie模式下不需要安装交叉编译工具器

- 大语言模型暂不需要安装tpu-nntc

- 在一个docker容器中即可，不需要多个容器
- 如果在Libsophon中无法ls /dev/bm*查看到

## 加速器

- 采用标准 PCIE X16 Gen4.0 物理接口，宿主机侧需提供标准的 X16 slot 或至少 X8 in X16 的 PCIE slot（要求尾部开放）.

- 实际应用中建议安装到标准服务器中使用且对卡做相应转速调节保证散热.，不建议安装在普通 PC 机/工控机中使用，如需使用，需要外加导风结构并在卡入风口安装高风量、风压风扇。
- 人工智能加速卡 SC7 FP300（全高全长双宽卡）上搭载 8 颗 BM1684X 芯片，可以支持8 x 32 路高清视频硬解码。标准 PCIE 4.0 接口，采用无风扇设计.

- 配置如下图：

<img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230828162841157.png" alt="image-20230828162841157" style="zoom: 67%;" />

## 安装与部署

人工智能加速卡 SC7 FP300 作为标准的 PCIe 卡，请按照如下步骤安装使用。

注意：在安装板卡进入服务器前，请拔掉服务器的主电源和任何网络，少数服务器产品在连接电源时，PCIE 插槽有漏电现象。

1. 使服务器脱离有效运行状态。
2. 关闭服务器电源。
3. 从服务器断开所有电源线，请参阅服务器服务手册。
4. 从机箱卸下箱盖
5. 将板卡插入可用的 PCIe 系统插槽。请轻轻摇晃设备使设备插入到位，不要用力过猛。请勿将 PCIe 板卡安装在 PCI 插槽，反之亦然。PCIe 板卡支持上插，即低位宽的设备可插入高位宽的 PCIe 插槽。
6. 将设备安装支架固定在服务器机箱，根据需要安装托架螺丝，或者啮合服务器固定装置。将卡固定到服务器机箱中
7.  人工智能加速卡 SC7 FP300 需要连接服务器内的 ATX 开关电源，提供 300W 电源
8. 使服务器恢复正常工作，装回箱盖，重新连接电源线以及任何网络电缆，打开系统电源。

### 设备识别确认

安装完毕后，可进入 Linux 操作系统，查看置于该计算机内的 BM1684X 设备，即对应的加速卡。

```
lspci|grep 168
```

## 操作系统环境

### Ubuntu

如果需要安装 Ubuntu 操作系统，推荐安装以下版本：Ubuntu 16.04、Ubuntu 16.04.5、Ubuntu 16.04.7、Ubuntu 18.04、Ubuntu 18.04.5、Ubuntu 18.04.6、Ubuntu 20.04.4



本机版本

cat /proc/version

Linux version 5.4.0-164-generic (buildd@bos03-amd64-056) (gcc version 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.2)) #181-Ubuntu SMP Fri Sep 1 13:41:22 UTC 2023

### SDK包

具体文件信息参考：[3.1. SDK 简介 — SophonSDKUserGuide v23.07.01 文档 (sophgo.com)](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/sdk_intro/1_intro.html)

- 下载SDK包，链接：[https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/12/01/22/Release_v2312-LTS.zip](https://developer.sophgo.com/site/index/material/all/all.html)

```
sudo apt-get install p7zip
sudo apt-get install p7zip-fullcd 
7z x Release_2370701-public.zip
cd Release_2370701-public
```

<img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230828150617143.png" alt="image-20230828150617143" style="zoom:80%;" />

#### **Docker安装(前提)**

- ```python
  # 安装docker
  sudo apt-get install docker.io
  # docker命令免root权限执行
  # 创建docker用户组，若已有docker组会报错，没关系可忽略
  sudo groupadd docker
  # 将当前用户加入docker组
  sudo gpasswd -a ${USER} docker
  # 重启docker服务
  sudo service docker restart
  # 切换当前会话到新group或重新登录重启X会话
  newgrp docker
  
  
  #添加镜像
  docker pull sophgo/tpuc_dev:latest
    
  
  提示：需要logout系统然后重新登录，再使用docker就不需要sudo了。
  ```

docker环境创建目录

```

```



#### tpu-mlir环境初始化

- 将压缩包解压到tpu-mlir

- ```python
  cd tpu-mlir_20230802_054500
  mkdir tpu-mlir
  tar zxvf tpu-mlir_v1.2.8-g32d7b3ec-20230802.tar.gz --strip-components=1 -C tpu-mlir
  ```

- 创建docker容器并进入Docker

  ```python
  cd tpu-mlir
  # 如果当前系统没有对应的镜像，会自动从docker hub上下载；此处将tpu-mlir的上一级目录映射到docker内的/workspace目  录；这里用了8001到8001端口的映射（使用ufw可视化工具会用到端口号）。
  # 如果端口已被占用，请根据实际情况更换为其他未占用的端口。
  docker run -v $PWD/..:/workspace -p 8001:8001 -it sophgo/tpuc_dev:latest#成功后自动进入到root模式
  docker run --privileged myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
        
        
  docker run --privileged --name newcode -v $PWD:/workspac -p 8001:8001 -it sophgo/tpuc_dev:latest
  '''
  	在Docker容器中运行名为sophgo/tpuc_dev的镜像。同时将主机的当前目录的上一级目录挂载到容器的/workspace目录，并将主机的8001端口映射到容器的8001端口，以便通过主机的8001端口访问容器内的服务。运行时可以进行交互式操作
  参数解释
  	docker run : 运行一个容器。
  	-v $PWD/..:/workspace : 卷挂载参数，用于将当前目录的上一级目录挂载到容器内的/workspace目录。$PWD表示当前工作目录的路径。
  	-p 8001:8001 : 端口映射参数，将主机的8001端口映射到容器的8001端口:可以通过主机的8001端口访问容器内的服务。
  	-it: 交互式运行的参数，以便在容器中进行交互式操作。
  	sophgo/tpuc_dev:latest : 这是要运行的Docker镜像的名称和标签。sophgo/tpuc_dev是镜像的名称，:latest是标签。 
  '''
  ```

  ```python
  '''
  注：如果编译错误如下
  docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?.
  See 'docker run --help'.
  '''
  sudo service docker status#检查docker状态
  #可以重启docker服务
  sudo service docker restart
  newgrp docker
  #可以使用 docker version检查，出现Server：选项正确/使用sudo service docker status检查（如下图）
  ```

  <img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230829100752992.png" alt="image-20230829100752992" style="zoom:80%;" />

- 初始化软件环境

  ```python
  cd /workspace/tpu-mlir
  #官方： source scripts/envsetup.sh  ：找不到路径
  source tpu-mlir_v1.2.8-g32d7b3ec-20230802/envsetup.sh
  #官方： ./build.sh
  ./tpu-mlir_v1.2.8-g32d7b3ec-20230802/customlayer/build.sh
   
  ```
  
  ```python
  请注意，如果docker stop后重新进入，则需要重新source环境变量。
  
  在该文件夹下 使用find -name envsetup.sh查找文件位置，根据自身地址修改命令
  build.sh文件位置如上行一样查找
  # 使用示例项目代码中自带的模型转换脚本，可以将pytorch模型转为onnx模型.
  ```

<img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230829103049046.png" alt="image-20230829103049046" style="zoom:80%;" />

```
如遇到上图问题，请进入到build.sh的文件夹下编译
即 cd /workspace/tpu-mlir/tpu-mlir_v1.2.8-g32d7b3ec-20230802/customlayer
build
#出现信息
Install the project...
-- Install configuration: "Debug"
-- Installing: /workspace/tpu-mlir/tpu-mlir_v1.2.8-g32d7b3ec-20230802/lib/libbackend_custom.so
```

#### tpu-nntc 环境初始化（没用到就不需要）

- 将压缩包解压到tpu-nntc

  ```
  cd tpu-nntc_20230802_054100
  mkdir tpu-nntc
  tar zxvf tpu-nntc_v3.1.9-62a7181e-230802.tar.gz --strip-components=1 -C tpu-nntc
  ```

- 创建docker容器并进入Docker

  ```python
  cd tpu-nntc
  # 如果当前系统没有对应的镜像，会自动从docker hub上下载；此处将tpu-nntc的上一级目录映射到docker内的/workspace目录；这里用了8001到8001端口的映射（使用ufw可视化工具会用到端口号）。
  # 如果端口已被占用，请根据实际情况更换为其他未占用的端口。
  docker run -v $PWD/..:/workspace -p 8001:8001 -it sophgo/tpuc_dev:latest
  ```

- 初始化软件环境

  ```python
  cd /workspace/tpu-nntc
  source scripts/envsetup.sh
  
  # 请注意，如果docker stop后重新进入，则需要重新source环境变量。
  ```

#### Libsophon环境搭建

具体驱动安装方法，可参考《LIBSOPHON 使用手册》**CHAPTER 3 安装   LIBSOPHON** 中的内容

参考网址：[3. 安装libsophon — LIBSOPHON-GUIDE v23.07.01 文档 (sophgo.com)](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/libsophon/guide/html/1_install.html)

准备（不要在一台机器上混用多种安装方式）

- 如果安装了V3.0.0版本及之前版本的SDK驱动，请先卸载旧的驱动

  ```python
  #进入SDK的安装目录，执行：
  sudo ./remove_driver_pcie.sh
  #或者：
  sudo rmmod bmsophon
  sudo rm -f /lib/modules/$(uname -r)/kernel/drivers/pci/bmsophon.ko
  ```

- 查看Linux架构，libsophon包主要文件及功能如下

  ```python
  uname -m
  x86_64
  #如果显示中含有X86_64 :amd64
  #如果显示中含有aarch64 :arm64
  ```

![image-20230828172012506](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230828172012506.png)

- 安装libsophon

  ```python
  cd libsophon_20230806_182258
  #安装依赖库，只需要执行一次：
  sudo apt install dkms libncurses5
  #安装libsophon：
  sudo dpkg -i sophon-*_amd64.deb
  #在终端执行如下命令，或者登出再登入当前用户后即可使用bm-smi等命令：
  source /etc/profile
  ```

  ```python
  #注意：
  #检查驱动是否安装成功：执行 ls /dev/bm* 看看是否有 /dev/bm-sohponX （X表示0-N），如果有表示安装成功。 正常情况下#输出如下信息：/dev/bmdev-ctl /dev/bm-sophon0
  输出：/dev/bmdev-ctl   /dev/bm-sophon1  /dev/bm-sophon3  /dev/bm-sophon5  
  /dev/bm-sophon7 /dev/bm-sophon0  /dev/bm-sophon2  /dev/bm-sophon4  /dev/bm-sophon6
  
  #安装位置
  /opt/sophon/
  ├── driver-0.4.9
  ├── libsophon-0.4.9
  │    ├── bin
  │    ├── data
  │    ├── include
  │    └── lib
  └── libsophon-current -> /opt/sophon/libsophon-0.4.9
  #注：若使用不同版本，必须在/opt/sophon下放置，在使用deb包安装时，/opt/sophon/libsophon-current会指向最后安装的那个版本。在卸载后，它会指向余下的最新版本（如果有的话）
  
  #错误提示：
  modprobe: FATAL: Module bmsophon is in use.
  #解决：手动停止正在使用驱动的程序后，手动执行下面的命令来安装新的deb包里的驱动
  sudo modprobe -r bmsophon
  sudo modprobe bmsophon
  ```

- 如果安装了sophon-mw及sophon-rpc，因为它们对libsophon有依赖关系，请先卸载它们

  ```python
  sudo apt remove sophon-driver sophon-libsophon
  #或者:
  sudo dpkg -r sophon-driver
  sudo dpkg -r sophon-libsophon-dev
  sudo dpkg -r sophon-libsophon
  ```

  ```python
  #错误解决：
  #手工卸载驱动：
  dkms status
  #检查输出结果，通常如下：
  bmsophon, 0.4.9, 5.15.0-41-generic, x86_64: installed
  #记下前两个字段，套用到如下命令中：
  sudo dkms remove -m bmsophon -v 0.4.9 --all
  #然后再次卸载驱动：
  sudo apt remove sophon-driver
  sudo dpkg --purge sophon-driver
  #彻底清除libsophon：
  sudo apt purge sophon-libsophon
  ```

#### sophon-mw环境搭建

如果使用Debian/Ubuntu系统（arm为例）：

![image-20230828163611046](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230828163611046.png)

- 安装

  ```python
  cd sophon-mw_20230807_032400/
  
  sudo dpkg -i sophon-mw-sophon-ffmpeg_0.7.0_amd64.deb sophon-mw-sophon-ffmpeg-dev_0.7.0_amd64.deb
  
  sudo dpkg -i sophon-mw-sophon-opencv_0.7.0_amd64.deb sophon-mw-sophon-opencv-dev_0.7.0_amd64.deb
  
  #在终端执行如下命令，或者logout再login当前用户后即可使用安装的工具：
  source /etc/profile
  ```

  ```python
  #注意：位于SOC模式时，系统已经预装了：
    sophon-mw-soc-sophon-ffmpeg
    sophon-mw-soc-sophon-opencv
  
  #只需要按照上述步骤安装：
    sophon-mw-soc-sophon-ffmpeg-dev_v23.07.01_arm64.deb
    sophon-mw-soc-sophon-opencv-dev_v23.07.01_arm64.deb
    
  #安装位置：
    /opt/sophon/
  ├── libsophon-v23.07.01
  ├── libsophon-current -> /opt/sophon/libsophon_|ver|
  ├── sophon-ffmpeg_|ver|
  │   ├── bin
  │   ├── data
  │   ├── include
  │   ├── lib
  │   │   ├── cmake
  │   │   └── pkgconfig
  │   └── share
  ├── sophon-ffmpeg-latest -> /opt/sophon/sophon-ffmpeg_|ver|
  ├── sophon-opencv_|ver|
  │   ├── bin
  │   ├── data
  │   ├── include
  │   ├── lib
  │   │   ├── cmake
  │   │   │   └── opencv4
  │   │   └── pkgconfig
  │   ├── opencv-python
  │   ├── share
  │   └── test
  └── sophon-opencv-latest -> /opt/sophon/sophon-opencv_|ver|
  
  ```

- 卸载方式

  ```python
  sudo apt remove sophon-mw-sophon-opencv-dev sophon-mw-sophon-opencv
  sudo apt remove sophon-mw-sophon-ffmpeg-dev sophon-mw-sophon-ffmpeg
  #或者:
  sudo dpkg -r sophon-mw-sophon-opencv-dev
  sudo dpkg -r sophon-mw-sophon-opencv
  sudo dpkg -r sophon-mw-sophon-ffmpeg-dev
  sudo dpkg -r sophon-mw-sophon-ffmpeg
  ```

#### 交叉编译工具链

- 安装gcc-aarch64-linux-gnu工具链

  ```python
  sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnuls
  ```

- 解压sophon-img包里的libsophon_soc_0.4.9_aarch64.tar.gz，将lib和include的所有内容拷贝到soc-sdk文件夹。

  ```python
  cd sophon-img_20230810_221353
  # 创建依赖文件的根目录
  mkdir -p soc-sdk
  # 解压sophon-img release包里的libsophon_soc_0.4.9_aarch64.tar.gz
  tar -zxf libsophon_soc_0.4.9_aarch64.tar.gz
  # 将相关的库目录和头文件目录拷贝到依赖文件根目录下
  cp -rf libsophon_soc_0.4.9_aarch64/opt/sophon/libsophon-0.4.9/lib ${soc-sdk}
  cp -rf libsophon_soc_0.4.9_aarch64/opt/sophon/libsophon-0.4.9/include ${soc-sdk}
  ```

- 解压sophon-mw包里的sophon-mw-soc_<x.y.z>_aarch64.tar.gz，将sophon-mw下lib和include的所有内容拷贝到soc-sdk文件夹。

  ```python
  cd sophon-mw_20230807_032400
  # 解压sophon-mw包里的sophon-mw-soc_0.7.0_aarch64.tar.gz ，其中x.y.z为版本号
  tar -zxf sophon-mw-soc_0.7.0_aarch64.tar.gz 
  # 将ffmpeg和opencv的库目录和头文件目录拷贝到依赖文件根目录下
  cp -rf sophon-mw-soc_0.7.0_aarch64/opt/sophon/sophon-ffmpeg_0.7.0/lib ${soc-sdk}
  cp -rf sophon-mw-soc_0.7.0_aarch64/opt/sophon/sophon-ffmpeg_0.7.0/include ${soc-sdk}
  cp -rf sophon-mw-soc_0.7.0_aarch64/opt/sophon/sophon-opencv_0.7.0/lib ${soc-sdk}
  cp -rf sophon-mw-soc_0.7.0_aarch64/opt/sophon/sophon-opencv_0.7.0/include ${soc-sdk}
  ```

- 如果需要使用第三方库，可以使用qemu在x86上构建虚拟环境安装，再将头文件和库文件拷贝到soc-sdk目录中，具体可参考《 [sophon-mw使用手册](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/sophon-mw/manual/html/index.html) 》

- 验证

  ```python
  which aarch64-linux-gnu-g++
  # 终端输出内容
  # /usr/bin/aarch64-linux-gnu-g++
  ```

  ```
  ./model_transform.py --model_name Yolact --model_def ./yolact_base_54_800000.onnx --input_shapes [[1,3,550,550]] --mean 0.0,0.0,0.0 --scale 0.0039216,0.0039216,0.0039216 --keep_aspect_ratio--pixel_format rgb --test_input ../COCO2017/000000000632.jpg --test_result yolact_top_outputs.npz --mlir yolact.mlir
  ```
  

错误：

![image-20230920094732267](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230920094732267.png)

执行

```
sudo apt-get update
sudo apt-get upgrade
```

![image-20230927143236207](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230927143236207.png)

![image-20231219174512470](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231219174512470.png)
