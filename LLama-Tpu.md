

# Llama2-Tpu

本项目实现BM1684X部署语言大模型[Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境

下文中默认是PCIE环境

## 开发环境

### 下载docker

```
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

**后文假定都在docker的workplace环境中**

### 下载Llama-7b模型

（如果服务器可以链接huggingface也可以不下载）

下载路径为: http://disk-sophgo-vip.quickconnect.to/sharing/RAcn5E1zU 密码：123456

或者huggingface官方下载

### 修改源代码

修改的目的是为了保证model_tool --combine的时候block和block_cache权重能对齐

```
pip show transformers
vi /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
```

修改316行左右的代码，修改前为

```python
if past_key_value is not None:
  kv_seq_len += past_key_value[0].shape[-2]
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
```

修改后

```python
if past_key_value is not None:
  kv_seq_len += past_key_value[0].shape[-2]
if past_key_value is not None:
  cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len-1)
else:
  cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
```

### 下载`TPU-MLIR`代码并编译

也可以直接下载编译好的release包解压

```
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```

### 下载[sentencepiece](https://github.com/google/sentencepiece)

```
git clone git@github.com:google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j
```

注意：此处编译之后后期使用到该包还是无法找到，使用`pip install `下载安装即可

### 下载libsophon库并安装

如果是跑分布式模型，libsophon请联系[yi.chu@sophgo.com](mailto:yi.chu@sophgo.com)获取

在算能官网https://developer.sophgo.com/site/index/material/all/all.html可以找到SDK最新版本，如下：

```
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/06/15/16/Release_230501-public.zip
```

注意：git不下来可以在官网上直接下载

```
前提：安装过驱动
安装依赖库
sudo apt install dkms libncurses5
sudo dpkg -i  sophon-libsophon_0.4.9_amd64.deb
安装libsophon
sudo dpkg -i sophon-libsophon-dev_0.4.9_amd64.deb
0.4.9为版本号，注意检查改为自己所下载的版本号

遇到红色E
sudo apt-get update


对于安装包
mkdir build
cd build
cmake ..
make driver
make package
```

下载Llama-Tpu

```
git clone git@github.com:sophgo/Llama2-TPU.git
```

## 编译模型（分布式）

1. 导出所有onnx模型，如果过程中提示缺少某些组件，直接`pip install 组件`即可

   ```
   cd Llama2-TPU/compile
   python3 export_onnx.py
   
   文件保存在complie/tmp中
   ```

2. 对onnx模型进行编译，生成bmodel，这个过程会花一些时间，最终生成`llama2-7b.bmodel`文件

   ```
   ./compile.sh --num_device 2
   ```

   F32/F16/BF16/INT8

## 参考

[sophgo/Llama2-TPU (github.com)](https://github.com/sophgo/Llama2-TPU)

