### **linspace()**

返回一个一维的tensor（张量），这个张量包含了从start到end，分成steps个线段得到的向量。常用的几个变量

用法 a.linspace(start，end，step，dtype)

​	start：开始值

​	end：结束值

​	steps：分割的点数，默认是100

​	dtype：返回值（张量）的数据类型

```
import torch
print(torch.linspace(3,10,5))
#tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000])
 
 
type=torch.float
print(torch.linspace(-10,10,steps=6,dtype=type))
#tensor([-10.,  -6.,  -2.,   2.,   6.,  10.])
```

### np.ceil（a）

对a的数值向上取整

### np.floor(a)

对a的数据向下取整

### axis

axis翻译过来就是轴的意思。

[numpy](https://so.csdn.net/so/search?q=numpy&spm=1001.2101.3001.7020)数组中：

- 一维数组拥有一个轴：axis=0；
- 二维数组拥有两个轴：axis=0，axis=1；
- 三维数组拥有三个轴：axis=0，axis=1，axis=2。
- 四维数组拥有三个轴：axis=0，axis=1，axis=2，axis=3。

numpy数组都有`[]`标记，其对应关系：axis=0对应最外层的`[]`，axis=1对应第二外层的`[]`，…，axis=n对应第n外层的`[]`

axis=-1，表示在当前数组最后一维度操作

例如：

```python
import numpy as np
arr = np.array([1, 2, 3])
arr.sum(axis = 0)

6

#过程：
#第一步：axis=0对应最外层[]，其内最大单位块为：1，2，3，并去掉[]
#第二步：单位块是数值，直接计算：1+2+3=6
```

### np.repeat（）

参数：

 [axis](https://so.csdn.net/so/search?q=axis&spm=1001.2101.3001.7020)=None，时候就会flatten当前矩阵，实际上就是变成了一个行向量

 axis=0,增加行数，列数不变

 axis=1,增加列数，行数不变

 **repeats 复制次数或者按照特定方式复制**

```python
import numpy as np
c = np.array(([1,2],[3,4],[4,5]))
print (c.shape)
f = np.repeat(c,3)
print(f)
print('c形状：',c.shape)

'''
(3, 2)
[1 1 1 2 2 2 3 3 3 4 4 4 4 4 4 5 5 5]
c形状： (3, 2)
'''
```

```python
#对行操作，增加的是列，一行2个元素，[2,3]就是2个元素(行不变，列变：第一列复制2次，第二列复制3次)
d=np.repeat(c,[2,3],axis=1)
print(d)
print('d形状：',d.shape)

'''
[[1 1 2 2 2]
 [3 3 4 4 4]
 [4 4 5 5 5]]
d形状： (3, 5)
'''
```

### getattr（）

函数用于返回一个对象属性值。

**getattr(object, name[, default])**

- object -- 对象。

- name -- 字符串，对象属性。

- default -- 默认返回值，如果不提供该参数，在没有对应属性时，将触发 AttributeError。

- ```python
  >>>class A(object):
  ...     bar = 1
  ... 
  >>> a = A()
  >>> getattr(a, 'bar')        # 获取属性 bar 值
  1
  >>> getattr(a, 'bar2')       # 属性 bar2 不存在，触发异常
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  AttributeError: 'A' object has no attribute 'bar2'
  >>> getattr(a, 'bar2', 3)    # 属性 bar2 不存在，但设置了默认值
  3
  >>>
  ```


### 线性层

$$
\text{output}_i = \sum{j=1}^{2} \text{input}_j \times \text{weight}_{ij} + \text{bias}_i
$$



​	(i)表示输出特征的索引，(j)表示输入特征的索引。

​	对于线性层（2，10），就有10个输出特征。这个过程对每个输出特征都会执行，最终得到一个具有10个元素的输出向量

### einsum()

爱因斯坦求和约定，其实作用就是把求和符号省略

### view()

对 Tensor 的尺寸修改，可以采用 `torch.view()`

```
x = torch.randn(4, 4)
y = x.view(16)
# -1 表示除给定维度外的其余维度的乘积
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

output：
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

类似于reshape，将tensor转换为指定的shape，原始的data不改变。返回的tensor与原始的tensor共享存储区。返回的tensor的size和stride必须与原始的tensor兼容

### torch.onnx.export()

```python
torch.onnx.export(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, aten=False, export_raw_ir=False, operator_export_type=None, opset_version=None, _retain_param_name=True, do_constant_folding=False, example_outputs=None, strip_doc_string=True, dynamic_axes=None, keep_initializers_as_inputs=None)
```

参数说明：

**model** (torch.nn.Module)  要导出的**模型**.
**args** (tuple of arguments) – 模型的**输入**, 任何非Tensor参数都将硬编码到导出的模型中；任何Tensor参数都将成为导出的模型的输入，并按照他们在args中出现的顺序输入。因为export运行模型，所以我们需要提供一个输入张量x。只要是正确的类型和大小，其中的值就可以是随机的。请注意，除非指定为动态轴，否则输入尺寸将在导出的ONNX图形中固定为所有输入尺寸。在此示例中，我们使用输入batch_size 1导出模型，但随后dynamic_axes 在torch.onnx.export()。因此，导出的模型将接受大小为[batch_size，3、100、100]的输入，其中batch_size可以是可变的。
**export_params** (bool, default True) – 如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
**verbose** (bool, default False) - 如果指定，我们将打印出一个导出轨迹的调试描述。
**training** (bool, default False) - 在训**练模式下导出模型**。目前，ONNX导出的模型只是为了做推断，所以你通常不需要将其设置为True。
**input_names** (list of strings, default empty list) – 按顺序分配名称到图中的输入节点
**output_names** (list of strings, default empty list) –按顺序分配名称到图中的输出节点
**dynamic_axes** – {‘input’ : {0 : ‘batch_size’}, ‘output’ : {0 : ‘batch_size’}}) # variable lenght axes

```python
import torch

class MLPModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.fc0 = nn.Linear(8, 8, bias=True)
      self.fc1 = nn.Linear(8, 4, bias=True)
      self.fc2 = nn.Linear(4, 2, bias=True)
      self.fc3 = nn.Linear(2, 2, bias=True)

  def forward(self, tensor_x: torch.Tensor):
      tensor_x = self.fc0(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      tensor_x = self.fc1(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      tensor_x = self.fc2(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      output = self.fc3(tensor_x)
      return output

model = MLPModel()
tensor_x = torch.rand((97, 8), dtype=torch.float32)
export_output = torch.onnx.dynamo_export(model, tensor_x)
# 说明
# 只需要模型以及输入即可
```

### masked_fill()

**masked_fill_(mask, value)**
掩码操作
用value填充tensor中与mask中值为1位置相对应的元素。mask的形状必须与要填充的tensor形状一致（可广播的张量也可以）。

```
a = torch.randn(5,6)

x = [5,4,3,2,1]
mask = torch.zeros(5,6,dtype=torch.float)
for e_id, src_len in enumerate(x):
    mask[e_id, src_len:] = 1
mask = mask.to(device = 'cpu')
print(mask)
a.data.masked_fill_(mask.byte(),-float('inf'))
print(a)
----------------------------输出
tensor([[0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1.],
        [0., 1., 1., 1., 1., 1.]])
tensor([[-0.1053, -0.0352,  1.4759,  0.8849, -0.7233,    -inf],
        [-0.0529,  0.6663, -0.1082, -0.7243,    -inf,    -inf],
        [-0.0364, -1.0657,  0.8359,    -inf,    -inf,    -inf],
        [ 1.4160,  1.1594,    -inf,    -inf,    -inf,    -inf],
        [ 0.4163,    -inf,    -inf,    -inf,    -inf,    -inf]])


```

### enumerate()

enumerate()是python的内置函数
enumerate在字典上是枚举、列举的意思
对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
enumerate多用于在for循环中得到计数
例如对于一个seq，得到：

`(0, seq[0]), (1, seq[1]), (2, seq[2])`

如果对一个列表，既要遍历索引又要遍历元素时，首先可以这样写：

```python
list1 = ["这", "是", "一个", "测试"]
for i in range (len(list1)):
    print i ,list1[i]
    
# 上述方法有些累赘，利用enumerate()会更加直接和优美：
list1 = ["这", "是", "一个", "测试"]
for index, item in enumerate(list1):
    print index, item
>>>
0 这
1 是
2 一个
3 测试
```

**enumerate还可以接收第二个参数**，用于指定索引**起始值**，如

```python
list1 = ["这", "是", "一个", "测试"]
for index, item in enumerate(list1, 1):
    print index, item
>>>
1 这
2 是
3 一个
4 测试
```

如果要统计文件的行数，可以这样写

```python
count = len(open(filepath, 'r').readlines())

#这种方法简单，但是可能比较慢，当文件比较大时甚至不能工作。
#可以利用enumerate()：

count = 0
for index, line in enumerate(open(filepath,'r'))： 
    count += 1
```

### np.triu（）

**np.triu(目标tensor, 赋值数)**

对角线元素赋值

```python
import numpy as np

arr = np.ones((3,3))
print(arr)

[[1 1 1
  1 1 1
  1 1 1]]
 
>>>print(np.triu(arr,0))
 
 [[1 1 1
 	 0 1 1
 	 0 0 1]]
 	 
>>> print(np.triu(arr,1))

 [[0 1 1
 	 0 0 1
 	 0 0 0]]
 	 
>>> print(np.triu(arr,-1))	 
   [[1 1 1
 	 	 1 1 1
 	   0 1 1]]
 	 
```

### torch.max(input, dim) 函数

索引dim上的最大值

```python
import torch
a = torch.tensor([[1,5,62,54], [2,6,2,6], [2,65,2,6]])
print(a)

tensor([[ 1,  5, 62, 54],
        [ 2,  6,  2,  6],
        [ 2, 65,  2,  6]])

torch.max(a, 1)

torch.return_types.max(
values=tensor([62,  6, 65]),
indices=tensor([2, 3, 1]))
```

### torch.topk() 函数

函数作用：用来获取张量或者[数组](https://link.zhihu.com/?target=https%3A//so.csdn.net/so/search%3Fq%3D%E6%95%B0%E7%BB%84%26spm%3D1001.2101.3001.7020)中最大或者最小的元素以及索引位置，是一个经常用到的基本函数。



## TIPs

pytorch中的操作

可以改变 tensor 变量的操作都带有一个后缀 `_`, 例如 `x.copy_(y), x.t_()` 都可以改变 x 变量

