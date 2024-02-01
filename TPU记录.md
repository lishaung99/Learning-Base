# **TPU命令大全**

## 设备识别

lspci|grep 1684x

## 查看驱动

 ls /dev/bm*

## 查看TPU

bm-smi

## Tpu驱动sysfs文件系统介绍

sysfs文件系统接口用来获取TPU的利用率等信息

- npu_usage，Tpu（npu）在一段时间内（窗口宽度）处于工作状态的百分比。
- npu_usage_enable，是否使能统计npu利用率，默认使能。1：开启   0  关闭
- npu_usage_interval，统计npu利用率的时间窗口宽度，单位ms，默认500ms。取值范围[200,2000]。

```
root@bitmain:/sys/class/bm-sophon/bm-sophon0/device# cat npu_usage_interval

"interval": 600

root@bitmain:/sys/class/bm-sophon/bm-sophon0/device# echo 500 > npu_usage_interval

root@bitmain:/sys/class/bm-sophon/bm-sophon0/device# cat npu_usage_interval

"interval": 500
```

关闭利用率的统计

```
root@bitmain:/sys/class/bm-sophon/bm-sophon0/device# cat npu_usage_enable

"enable": 1

root@bitmain:/sys/class/bm-sophon/bm-sophon0/device# echo 0 > npu_usage_enable

root@bitmain:/sys/class/bm-sophon/bm-sophon0/device# cat npu_usage_enable

"enable": 0

root@bitmain:/sys/class/bm-sophon/bm-sophon0/device# cat npu_usage

Please, set [Usage enable] to 1

root@bitmain:/sys/class/bm-sophon/bm-sophon0/device# echo 1 > npu_usage_enable

root@bitmain:/sys/class/bm-sophon/bm-sophon0/device# cat npu_usage_enable

"enable": 1

root@bitmain:/sys/class/bm-sophon/bm-sophon0/device# cat npu_usage

"usage": 0, "avusage": 0

root@bitmain:/sys/class/bm-sophon/bm-sophon0/device#
```

查看利用率

```
root@bitmain:/sys/class/bm-sophon/bm-sophon0/device# cat npu_usage

"usage": 0, "avusage": 0
#过去一个时间窗口内的npu利用率
#自安装驱动以来的npu利用率
```

## 文件功能

### model_transform.py

模型转换，将其他模型转换为**mlir**模型

#### 支持图片输入

```
model_transform.py \
    --model_name resnet \
    --model_def resnet.onnx \
    --input_shapes [[1,3,224,224]] \  
    --mean 103.939,116.779,123.68 \
    --scale 1.0,1.0,1.0 \
    --pixel_format bgr \
    --test_input cat.jpg \
    --test_result resnet_top_outputs.npz \
    --mlir resnet.mlir
```

--input_shapes ：指定输入的shape, 例如[[1,3,640,640]]; 二维数组, 可以支持多输入情况

#### 支持多输入

当模型有多输入的时候, 可以传入1个npz文件, 或者按顺序传入多个npz文件, 用逗号隔开

```
model_transform.py \
    --model_name somenet \
    --model_def  somenet.onnx \
    --test_input somenet_in.npz \ # a.npy,b.npy,c.npy
    --test_result somenet_top_outputs.npz \
    --mlir somenet.mlir
```

### **model_deploy.py**

将模型从中间状态mlir 转换为bmodel

```
model_deploy.py \
   --mlir resnet.mlir \  
   --quantize F32 \ # F16/BF16
   --chip bm1684x \
   --test_input resnet_in_f32.npz \
   --test_reference resnet_top_outputs.npz \
   --model resnet50_f32.bmodel
```

--mlir ：模型文件

--quantize：量化类型 ，支持F32/F16/BF16/INT8

--chip：模型将要用到的平台

--test_input ：验证文件

 --test_reference ：用于验证模型正确性的参考数据(使用npz格式)

### model_runner

```
model_runner.py \
   --input sample_in_f32.npz \
   --model sample.bmodel \
   --output sample_output.npz
```

