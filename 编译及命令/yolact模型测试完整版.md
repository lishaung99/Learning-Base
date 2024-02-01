# 基于sc7 1684x的yolact模型测试完整版

# 1.环境部署

## 1.1 sdk开发包

```shell
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/06/15/16/Release_230701-public.zip
unzip Release_230701-public.zip
```

## 1.2 安装libsophon

uname -m：查看系统架构

```shell
cd Release_230701-public/libsophon_20230806_182258
sudo dpkg -i sophon-*.deb(会有提示 *arm*.deb的文件无法安装：因为该系统不是arm的，忽略uname -m)
source /etc/profile
```

## 1.3 安装sophon-mw

依赖libsophon

```shell
cd Release_230701-public/sophon-mw_20230807_032400
sudo dpkg -i sophon-mw-sophon-ffmpeg_0.7.0_amd64.deb sophon-mw-sophon-ffmpeg-dev_0.7.0_amd64.deb
sudo dpkg -i sophon-mw-sophon-opencv_0.7.0_amd64.deb sophon-mw-sophon-opencv-dev_0.7.0_amd64.deb
source /etc/profile
```

## 1.4 安装sophon-sail

编译可被Python3接口调用的Wheel文件

### sail编译与安装

使用**典型编译方式一**  （PCIE MODE）

使用默认安装路径,编译包含bmcv,sophon-ffmpeg,sophon-opencv的SAIL

```shell
使用默认安装路径,编译包含bmcv,sophon-ffmpeg,sophon-opencv的SAIL
下载sophon-sail源码,解压后进入其源码目录
cd Release_230701-public/sophon-sail_20230802_085400
cd sophon-sail/
mkdir build && cd build
cmake .. #如果没有cmake，按照提示安装sudo apt  install cmake
make pysail
```

打包生成python wheel,生成的wheel包的路径为‘python/pcie/dist’,文件名为‘sophon-3.6.0-py3-none-any.whl’

```shell
cd ../python/pcie
chmod +x sophon_pcie_whl.sh
./sophon_pcie_whl.sh
```

安装python wheel

```shell
pip install ./dist/sophon-3.6.0-py3-none-any.whl --force-reinstall
#pip install --upgrade pip
```



## 1.5 交叉编译环境搭建

### 安装工具链

```shell
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

### sophon-img

解压sophon-img包里的libsophon_soc_<x.y.z>_aarch64.tar.gz，将lib和include的所有内容拷贝到soc-sdk文件夹

```shell
cd sophon-img_20230810_221353
# 创建依赖文件的根目录
mkdir -p soc-sdk
# 解压sophon-img release包里的libsophon_soc_0.4.9_aarch64.tar.gz
tar -zxf libsophon_soc_0.4.9_aarch64.tar.gz
# 将相关的库目录和头文件目录拷贝到依赖文件根目录下
cp -rf libsophon_soc_0.4.9_aarch64/opt/sophon/libsophon-0.4.9/lib ${soc-sdk}
cp -rf libsophon_soc_0.4.9_aarch64/opt/sophon/libsophon-0.4.9/include ${soc-sdk}
```

### sophon-mw

```shell
cd sophon-mw_20230807_032400
# 解压sophon-mw包里的sophon-mw-soc_0.7.0_aarch64.tar.gz ，其中x.y.z为版本号
tar -zxf sophon-mw-soc_0.7.0_aarch64.tar.gz 
# 将ffmpeg和opencv的库目录和头文件目录拷贝到依赖文件根目录下
cp -rf sophon-mw-soc_0.7.0_aarch64/opt/sophon/sophon-ffmpeg_0.7.0/lib ${soc-sdk}
cp -rf sophon-mw-soc_0.7.0_aarch64/opt/sophon/sophon-ffmpeg_0.7.0/include ${soc-sdk}
cp -rf sophon-mw-soc_0.7.0_aarch64/opt/sophon/sophon-opencv_0.7.0/lib ${soc-sdk}
cp -rf sophon-mw-soc_0.7.0_aarch64/opt/sophon/sophon-opencv_0.7.0/include ${soc-sdk}
```

### 验证

```shell
which aarch64-linux-gnu-g++
# 终端输出内容
# /usr/bin/aarch64-linux-gnu-g++
```

## 1.6 tpu-mlir

主要量化工具

### 安装docker

```shell
以下为首次安装时需要的配置：
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
docker pull sophgo/tpuc_dev:v2.2
docker pull sophgo/tpuc_dev:latest
```

# 2.部署 yolact

使用tpu-mlir

## 2.1. 前期准备

### 创建docker

注意TPU-MLIR工程在docker中的路径应该是/workspace/tpu-mlir

```
在Release_230701-public/tpu-mlir_20230802_054500下运行docker命令

docker run --privileged --name name -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

### 配置路径

```shell
tar -zxf tpu-mlir_v1.2.8-g32d7b3ec-20230802.tar.gz
source tpu-mlir_v1.2.8-g32d7b3ec-20230802/envsetup.sh
可以输入model_transform.py测试环境

```

### 下载算法

```shell
git clone https://github.com/sophon-ai-algo/examples.git
# Yolact示例项目代码位置 /examples/simple/yolact

# 通过脚本下载需要的数据和模型
cd examples/simple/yolact/
# 执行脚本下载数据和模型
./scripts/download.sh

注：如果遇到断网，即timeout ，可以在github上面搜索yolact_base_54_800000.pth手动下载
```

### 拷贝数据集

```shell
cp -rf $TPUC_ROOT/regression/image .
cp -rf $TPUC_ROOT/regression/dataset/COCO2017/ .
mkdir workspace && cd workspace
```

## 2.2 模型转换

###  pytorch模型转onnx模型

```shell
#使用示例项目代码中自带的模型转换脚本，可以将pytorch模型转为onnx模型：
python3 ../scripts/converter/convert.py --input ../data/models/yolact_base_54_800000.pth --mode onnx --cfg yolact_base

#移动onnx模型到当前目录
mv ../scripts/converter/yolact_base_54_800000.onnx .
```

### ONNX转MLIR

```shell
# 创建模型转换命令脚本并执行
vi onnx2mlir.sh
sh onnx2mlir.sh

# onnx2mlir.sh中的内容
model_transform.py \
    --model_name Yolact \
    --model_def ./yolact_base_54_800000.onnx \
    --input_shapes [[1,3,550,550]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --test_input ../COCO2017/000000000632.jpg \
    --test_result yolact_top_outputs.npz \
    --mlir yolact.mlir
```

转成mlir文件后, 会生成一个 `${model_name}_in_f32.npz` 文件, 该文件是模型的输入文件。

### MLIR转F32模型

```shell
# 创建模型转换命令脚本并执行
vi mlir2bmodel_f32.sh
sh mlir2bmodel_f32.sh

# mlir2bmodel_f32.sh中的内容
model_deploy.py \
    --mlir yolact.mlir \
    --quantize F32 \
    --chip bm1684x \
    --tolerance 0.99,0.99 \
    --test_input Yolact_in_f32.npz \
    --test_reference yolact_top_outputs.npz \
    --model Yolact_1684x_f32.bmodel
```

编译完成后, 会生成名为 `${model_name}_1684x_f32.bmodel` 的文件。

### MLIR转INT8模型

#### 生成校准表

转INT8模型前需要跑calibration, 得到校准表，然后用校准表, 生成对称或非对称bmodel；如果对称符合需求, 一般不建议用非对称, 因为 非对称的性能会略差于对称模型。

用现有的100张来自COCO2017的图片举例, 执行calibration，这一步可能等待的时间较长:

```shell
vi run_cali.sh
sh run_cali.sh

#run_cali.sh的内容如下
run_calibration.py yolact.mlir \
    --dataset ../COCO2017 \
    --input_num 100 \
    -o yolact_cali_table
```

运行完成后会生成名为 `${model_name}_cali_table` 的文件, 该文件用于后续编译INT8 模型的输入文件。

#### 编译为INT8对称量化模型

```shell
# 创建模型转换命令脚本并执行
vi mlir2bmodel_int8_sym.sh
sh mlir2bmodel_int8_sym.sh

# mlir2bmodel_int8_sym.sh内容如下
model_deploy.py \
    --mlir yolact.mlir \
    --quantize INT8 \
    --calibration_table yolact_cali_table \
    --chip bm1684x \
    --test_input Yolact_in_f32.npz \
    --test_reference yolact_top_outputs.npz \
    --tolerance 0.85,0.45 \
    --model Yolact_1684x_int8_sym.bmodel
```

编译完成后, 会生成名为 `${model_name}_1684x_int8_sym.bmodel` 的文件。

#### 编译为INT8非对称量化模型

```shell
# 创建模型转换命令脚本并执行
vi mlir2bmodel_int8_asym.sh
sh mlir2bmodel_int8_asym.sh

# mlir2bmodel_int8_asym.sh内容如下
model_deploy.py \
    --mlir yolact.mlir \
    --quantize INT8 \
    --asymmetric \
    --calibration_table yolact_cali_table \
    --chip bm1684x \
    --test_input Yolact_in_f32.npz \
    --test_reference yolact_top_outputs.npz \
    --tolerance 0.90,0.55 \
    --model Yolact_1684x_int8_asym.bmodel
```

编译完成后, 会生成名为 `${model_name}_1684x_int8_asym.bmodel` 的文件。

## 2.3 Yolact模型推理测试

### 准备示例程序

将Yolact_1684x_f32.bmodel、Yolact_1684x_int8_asym.bmodel、Yolact_1684x_int8_sym.bmodel三个文件移动到/data/models下

```shell
mv Yolact_1684x_f32.bmodel Yolact_1684x_int8_asym.bmodel Yolact_1684x_int8_sym.bmodel ../data/models
# 退回至yolact/python目录
cd ../python
```

```shell
export LD_LIBSORARY_PATH=/opt/sophon/sophon-opencv-latest/lib/:$LD_LIBRARY_PATH
export LD_LIBSORARY_PATH=/opt/sophon/sophon-ffmpeg-latest/lib/:$LD_LIBRARY_PATH
export LD_LIBSORARY_PATH=/opt/sophon/sophon-bmcpu-latest/lib/:$LD_LIBRARY_PATH
export LD_LIBSORARY_PATH=/opt/sophon/sophon-libsophon-current/lib/:$LD_LIBRARY_PATH
```



### 部署测试

```shell
# sail解码 + bmcv预处理 + sail推理 + opencv后处理
# 下面的model参数Yolact_1684x_int8_asym.bmodel、Yolact_1684x_int8_sym.bmodel
python3 yolact_bmcv.py --cfgfile configs/yolact_base.cfg --model ../data/models/Yolact_1684x_f32.bmodel --is_video 0 --input_path ../image/
# 执行完毕后，在当前目录生成result bmcv文件夹，检测结果保存在该文件夹下。

# opencv解码 + opencv预处理 + sail推理 + opencv后处理
# 下面的model参数Yolact_1684x_int8_asym.bmodel、Yolact_1684x_int8_sym.bmodel
python3 yolact_sail.py --cfgfile configs/yolact_base.cfg --model ../data/models/Yolact_1684x_f32.bmodel --is_video 0 --input_path ../image/
# 执行完毕后，在当前目录生成result_cv文件夹，检测结果保存在该文件夹下。
```

```shell
python3 yolov5_opencv_3output.py --bmodel ../workspace/yolov5s_1684x_f32.bmodel --input ../image/dog.jpg
```

![image-20230927143035770](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230927143035770.png)
