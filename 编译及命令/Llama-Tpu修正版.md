

# Llama2-TPU（pcie模式下7b模型）

----------------------------

**如果体验直接执行阶段三**

## 阶段一：模型编译

### 准备

链接：https://pan.baidu.com/s/1x3FsXZ1UUhFH2XQK9PhJcg?pwd=sc7l 
提取码：sc7l 

下载文件tpu-mlir，libsophon（Release版本也可以），以及语言模型并解压

git clone https://github.com/sophgo/Llama2-TPU.git 

请使用更新后的文件（文档路径不一致请使用更新后的Llama2-TPU的包）

**首次使用docker：**

安装准备

```
#如果是首次使用Docker, 可执行下述命令进行安装和配置(仅首次执行):
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
# 拉取镜像
docker pull sophgo/tpuc_dev:latest
```

### 步骤一：docker

启动容器，如下：（在使用包的上一层启动）

``` shell
#启动容器
docker run --privileged --name code -v $PWD:/workspace -it sophgo/tpuc_dev:latest
如果跑web前端界面，需要指定端口
```

需要包含文件：

![image-20231219182314445](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231219182314445.png)

**下面的阶段二和三都在Docker中进行**



### 步骤二：TPU-MLIR

下载TPU-MLIR代码并编译，这里直接下载

``` shell
# 注意根据实际文件路径编译
tar -zxvf tpu-mlir_v1.1.0_RC1.77-g339a77f9-20231110.tar.gz
cd tpu-mlir
source ./envsetup.sh
```

### 步骤三：依赖包

**下载transfomers、sentencepiece、Llama2-TPU以及百度网盘里的.bin模型，并替换transformers里面的modeling_llama.py**

``` shell
pip install sentencepiece transformers==4.31.0

# 下载sophgo例程

cd Llama2-TPU/compile
# 下载模型并解压
unzip llama-2-7b-chat-hf.zip
pip show transformers
cp modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
```

* PS：不一定是/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py这个路径，建议替换前先pip show transformers查看一下

### 步骤四：生成onnx文件

``` shell
python export_onnx_fast.py
```

![image-20231219185524888](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231219185524888.png)

* PS1：如果想要生成13b的onnx模型，需要将export_onnx_fast.py的24行改为"llama-2-13b-chat-hf"
* PS2：如果你想要debug，而不是一下子生成完成全部的onnx模型，可以将234行的num_layers改成1

### 步骤五：生成bmodel文件

生成单芯模型

``` shell
./compile.sh --mode int8
```

生成双芯模型

``` shell
./compile.sh --mode int8 --num_device 2
```

- PS1：最终会在Llama2-TPU/compile路径下生成combine后的文件，名为llama2-7b_int8.bmodel

- PS2：生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left（磁盘空间不足会报的错）

**问题: 该阶段报错，一般是转换文件有问题。**

1. 找不着文件：检查文件名以及路径

2. 编译有问题

   - 如果运行过程中提示有关键词mlir，代表tpu_mlir包安装不正确，可以重新安装

   - 如果报错：bmruntime_interface.h: No such file or directory，或者error：bmrt_creat_ex was not……，代表libsophgo安装不正确（与bmrt……相关）请重新安装或拉去该包替换原来的

----------------------------

## 阶段二：可执行文件生成

### 第一步：编译可执行文件

``` python
'''
此步骤前提
需要提前提前下载好libsophon.zip并解压，同时需要将Llama2-TPU/pcie/demo/CMakeLists.txt第三行改为解压后的路径，否则`编译会报错`
'''
cd Llama2-TPU/pcie/demo
mkdir build && cd build
cmake .. && make -j
```

执行完成后

```
ldd llama2
```

正常情况：

![image-20231219184312812](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231219184312812.png)

错误请删除

![image-20231219184442451](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231219184442451.png)

## 阶段三：正式模型推理

### 准备

7b模型使用文：件链接：https://pan.baidu.com/s/1-1QsOlI192V3VGZM-gLD-w?pwd=sc7l 
提取码：sc7l 

放到同一个文件夹下面

### PCIE上执行

```shell
# 安装依赖
unzip libsophon.zip
apt install ./sophon-libsophon_0.4.9_amd64.deb ./sophon-libsophon-dev_0.4.9_amd64.deb
source /etc/profile.d/libsophon-bin-path.sh

cd Llama2-TPU/bmodel
./llama2 --dev_id 0 --model llama2-7b_int8_dev.bmodel
```

![image-20231219184618732](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231219184618732.png)

问题：如果没有找到这2个sophon*.deb文件

```
cd build
cmake ..
make package
apt install ./sophon-libsophon_0.4.9_amd64.deb ./sophon-libsophon-dev_0.4.9_amd64.deb
source /etc/profile.d/libsophon-bin-path.sh
```

### Web界面 (请看下面的更新)

**其他：使用gradio跑前端页面**

**## 准备**

\* 准备一些pip库

```shell
shell

pip install gradio==3.50.0

pip install mdtex2html
```

**## 执行**

```shell
shell

cd pcie/web_demo

python web_demo.py
```

 PS0：`gradio必须装3.50.0`

\* PS1：请根据实际路径调整web_demo/chat.py的16行到18行中的device_id，bmodel_path，token_path，默认使用第一颗芯片

\* PS2：请根据实际需要调整web_demo/chat.py的78行到82行中的prompt，默认是使用prompt

\* PS3：请根据实际路径调整web_demo/CMakeLists.txt中的set(LIBSOPHON /workspace/soc0701)

\* PS4：请根据实际block数目调整web_demo/chat.cpp中的NUM_LAYERS，默认是使用Llama2-7B(NUM_LAYERS=32)

**报错**

![image-20231228100510289](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231228100510289.png)

### web更新

该界面可以在new这个docker环境中实现运行

[sophgo/Llama2-TPU: run Llama2 in BM1684X (github.com)](https://github.com/sophgo/Llama2-TPU)

```
pip install gradio==3.39.0
cd Llama2-TPU/web_demo
mkdir build
cd build
cmake ..
make -j
```

编译成功会在`build`文件夹下生成`libtpuchat.so*`, 此时可以在web_demo.py中指定bmodel_path token_path device_id, lib_path(编译生产的`libtpuchat.so*`文件, 默认路径是`./build`下), 以及dev_id。

运行

```
python web_demo.py --dev 0 --bmodel_path ../../Llama2-TPU2/compile/llama2-7b_int8.bmodel
```

![image-20231228153112190](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231228153112190.png)

该web界面可以看到历史信息

![image-20240110152047483](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20240110152047483.png)

若提供的网址打不开界面，解决方法如下：

**报错信息**

![image-20231228152228812](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231228152228812.png)

解决方法：

1. 下载缺失文件: https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64
2. 并修改名称为`frpc_linux_amd64_v0.2`. 
3. 将文件移动到目录 /usr/local/lib/python3.10/dist-packages/gradio/ 下
3. ![image-20240110102746015](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20240110102746015.png)

![cpd](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231228152110152.png)

