# llama.cpp LLM模型Windows CPU 部署（运行LLama2测试）

cpu：AMD Ryzen 7 5800 8-Core Processor 

LLAMA2-7B的参数量化过后大约13GB，模型运行后cpu利用率75%左右。

## 准备

### 安装MinGW

llama.cpp使用c++语言写的，因此安装编译器MinGW（Windows版g++编译工具）

安装之后需要把安装路径添加到系统环境变量，运用gcc -v查看是否安装成功

### 安装Cmake

下载地址：https://cmake.org/download/

下载安装，环境变量可以安装软件步骤中选择自动加入（如下图），也可以自己手动加入到系统变量中

<img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230817095555363.png" alt="image-20230817095555363" style="zoom:67%;" />

安装完成之后如下图

<img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230817095625743.png" alt="image-20230817095625743" style="zoom:67%;" />

## llama.cpp下载编译

### 下载

创建项目文件夹

在该项目文夹下使用git下载

```markdown
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
ls
```

### cmake编译

```markdown
mkdir build
cd build
cmake .. 
cmake --build . --config Release
```

注意：

1.如果`cmake ..` 编译出现not set 问题，请使用`cmake .. -G "MinGW Makefiles"`重新编译

2.若该命令编译失败：在CMakelists文件中设置了以下参数重新编译

![image-20230821163152294](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230821163152294.png)

3.若出现下图错误，请将编译记录删除：可以直接删除项目，重新下载

![image-20230821163329675](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230821163329675.png)

### 测试运行

```
在bin目录下
./main -h
```

## 运行LLaMA-7B模型测试

### 模型下载

https://huggingface.co/nyanko7/LLaMA-7B/tree/main

请求链接[Llama access request form - Meta AI](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

具体申请步骤可以参考：【Llama2模型申请与本地部署详细教程】 https://www.bilibili.com/video/BV1H14y1X7kD/?share_source=copy_web&vd_source=c52fdad139a3f6e2bf4086a79c9575ad

文件down后如图所示有三个，下载下来的源码放到\llama.cpp\models\7B

![image-20230817172446966](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230817172446966.png)

注意：tokenizer.model本身下载参数里面没有，需要从源码手动拉一个

链接：https://github.com/facebookresearch/llama.git

### 转换为ggml  F32 格式

```
PS D:\Work\llama2\llama.cpp> python convert.py .\llama.cpp\7B\

注：convert.py文件就在llama.cpp下
```

![image-20230821163123214](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230821163123214.png)

![image-20230817172316466](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230817172316466.png)

### 量化

```markdown
build\bin\quantize.exe D:\Work\llama2\llama.cpp\models\7B\ggml-model-f32.bin D:\Work\llama2\llama.cpp\models\7B\ggml-model-q4_0.bin  q4_0
```

注：ggml-model-q4_0.bin已有资源。链接如下：
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML

![image-20230821163754662](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230821163754662.png)

### **命令行交互**

```
.\build\bin\main.exe -m D:\Work\llama2\llama.cpp\models\7B\ggml-model-q4_0.bin  -n 128  --repeat_penalty 1.0 --color -i -r "User:" -f D:\Work\llama2\llama.cpp\prompts\chat-with-bob.txt


ggml-model-f32.bin
```

![image-20230821164338230](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230821164338230.png)

结果：

![image-20230821170907796](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230821170907796.png)

![image-20231227174149391](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231227174149391.png)

注：7b模型目前无法支持中文

附链接：Chinese-Llama-2中文第二代

https://huggingface.co/soulteary/Chinese-Llama-2-7b-ggml-q4

## Tips

**量化**

量化是一种减少用于表示数字或值的比特数的技术。由于量化减少了模型大小，因此它有利于在cpu或[嵌入式系统](https://so.csdn.net/so/search?q=嵌入式系统&spm=1001.2101.3001.7020)等资源受限的设备上部署模型。

llama量化手段：将权重存储在低精度数据类型中来降低模型参数的精度。

**GGML**

GGML库是一个为机器学习设计的张量库，它的目标是使大型模型能够在高性能的消费级硬件上运行。这是通过整数量化支持和内置优化算法实现的。

###  遇到的问题：

**1.cmake编译不通过**

具体参考”cmake编译章节“

**2.生成bin文件无法执行**

尝试1：修改生成的bin文件，无法执行

测试：down已经量化好的模型，可以运行，排除cpu架构问题。

解决办法：直接重新下载了一个llama.cpp编译。

### **延伸任务**——算子处理：算子怎么处理的

## 实现参考链接

MinGW安装参考

https://blog.csdn.net/weixin_42357472/article/details/131314105

部署参考链接

http://t.csdn.cn/Lgxar