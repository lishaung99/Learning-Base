# SC7 TPU及bmodel命令

## **TPU命令**

### **查看TPU**

```
bm-smi
```

![image-20231227104847862](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231227104847862.png)

### 清理TPU显存

```
bm-smi --dev=id号 --recovery
```



## bmodel命令

### 查看编译的bmodel信息

```
tpu_model --info xxx.bmodel
```

```
bmodel version: B.2.2                         # bmodel的格式版本号
chip: BM1684                                  # 支持的芯片类型
create time: Mon Apr 11 13:37:45 2022         # 创建时间

==========================================    # 网络分割线，如果有多个net，会有多条分割线
net 0: [informer_frozen_graph]  static        # 网络名称为informer_frozen_graph， 为static类型网络（即静态网络），如果是dynamic，为动态编译网络
------------                                  # stage分割线，如果每个网络有多个stage，会有多个分割线
stage 0:                                      # 第一个stage信息
subnet number: 41                             # 该stage中子网个数，这个是编译时切分的，以支持在不同设备切换运行。通常子网个数
                                              # 越少越好
input: x_1, [1, 600, 9], float32, scale: 1    # 输入输出信息：名称、形状、量化的scale值
input: x_3, [1, 600, 9], float32, scale: 1
input: x, [1, 500, 9], float32, scale: 1
input: x_2, [1, 500, 9], float32, scale: 1
output: Identity, [1, 400, 7], float32, scale: 1

device mem size: 942757216 (coeff: 141710112, instruct: 12291552, runtime: 788755552)  # 该模型在TPU上内存占用情况（以byte为单位)，格式为： 总占用内存大小（常量内存大小，指令内存大小, 运行时数据内存占用大小)
host mem size: 8492192 (coeff: 32, runtime: 8492160)   # 宿主机上内存占用情况（以byte为单位），格式为： 总占用内存大小（常量内存大小，运行时数据内存大小）
```

### 查看详细参数信息

```
tpu_model --print xxx.bmodel
```

### 分解bmodel

将一个包含多个网络多种stage的bmodel分解成只包含一个网络的一个stage的各个bmodel，分解出来的bmodel按照net序号和stage序号，命名为bm_net0_stage0.bmodel、bm_net1_stage0.bmodel等等。

```
tpu_model --extract xxx.bmodel
```

### 合并

#### 合并bmodel

```
tpu_model --combine a.bmodel b.bmodel c.bmodel -o abc.bmodel
```

将多个bmodel合并成一个bmodel，-o用于指定输出文件名，如果没有指定，则默认命名为compilation.bmodel。

多个bmodel合并后：

- 不同net_name的bmodel合并，接口会根据net_name选择对应的网络进行推理
- 相同net_name的bmodel合并，会使该net_name的网络可以支持多种stage(也就是支持不同的input shape)。接口会根据用户输入的shape，在该网络的多个stage中选择。对于静态网络，它会选择shape完全匹配的stage；对于动态网络，它会选择最靠近的stage。

限制：同一个网络net_name，使用combine时，要求都是静态编译，或者都是动态编译。暂时不支持相同net_name的静态编译和动态编译的combine。

#### 合并文件夹

```
tpu_model --combine_dir a_dir b_dir c_dir -o abc_dir
```

同combine功能类似，不同的是，该功能除了合并bmodel外，还会合并用于测试的输入输出文件。它以文件夹为单位合并，文件夹中必须包含经过编译器生成的三个文件：input_ref_data.dat, output_ref_data.dat, compilation.bmodel。

### 导出二进制数据

```
tpu_model --dump xxx.bmodel start_offset byte_size out_file
```

将bmodel中的二进制数据保存到一个文件中。通过print功能可以查看所有二进制数据的[start, size]，对应此处的start_offset和byte_size。

### 输出测试信息

```
bmrt_test --bmodel 文件.bmodel
```

