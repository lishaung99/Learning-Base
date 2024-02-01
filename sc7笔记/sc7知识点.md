# BM1684X开发

## 整体感知

<img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20240109111328908.png" alt="image-20240109111328908" style="zoom:80%;" />

<img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20240109111448446.png" alt="image-20240109111448446" style="zoom: 80%;" />

### SDK开发包

<img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231227104126428.png" alt="image-20231227104126428" style="zoom:67%;" />

### 环境配置

参考sc7fp300环境搭建

https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/index.html

## 开发原理

### kernel原理





<img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231227103119586.png" alt="image-20231227103119586" style="zoom:67%;" />

okkernel：提供底层原子接口 ，封装通讯机制

全局DDR核本地内存的关系（bm1684）单指令 多数据，

![image-20231227103628793](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231227103628793.png)





![image-20231227103609096](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231227103609096.png)

原子操作在arm9上执行

### BMRuntime

**总结**：读取bmodel，将bmodel放到tpu上执行，提供接口移植算法。

BMRuntime用于**读取BMCompiler的编译输出**(.bmodel)，驱动其在SOPHON TPU芯片中执行。BMRuntime向用户提供了丰富的接口，便于用户移植算法，其软件架构如下:

<img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231227111415747.png" alt="image-20231227111415747" style="zoom: 50%;" />

提供两种接口 c / c++ ，接口默认都是同步接口，有个别是异步接口(由NPU执行功能，CPU可以继续往下执行)，会特别说明。

BMRuntime分为四个部分：

<img src="C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231227111337039.png" alt="image-20231227111337039" style="zoom:67%;" />

#### BMLIB

对应的头文件是bmlib_runtime.h，对应的lib库为libbmlib.so，用于**设备管理，包括设备内存的管理**

##### 常用

1.设备

###### bm_dev_request

```c
/* [out] handle
 * [in]  devid
 */
bm_status_t bm_dev_request(bm_handle_t *handle, int devid);
```

用于请求一个设备，得到设备句柄handle。其他设备接口(bm_xxxx类接口)，都需要指定这个设备句柄。

| 参数名 | 输入/输出 | 说明                   |
| ------ | --------- | ---------------------- |
| handle | 输出      | 保存创建的handle的指针 |
| devid  | 输入      | 指定具体设备           |

其中devid表示设备号，在PCIE模式下，存在多个设备时可以用于选择对应的设备；在SoC模式下，请指定为0。

当请求成功时，返回BM_SUCCESS；否则返回其他错误码。

###### bm_dev_free

```c
 /* [out] handle
 */
void bm_dev_free(bm_handle_t handle);

//使用参考
// start program
bm_handle_t bm_handle;
bm_dev_request(&bm_handle, 0);
// do things here
......
// end of program
bm_dev_free(bm_handle);
```

用于释放一个设备。通常应用程序开始需要请求一个设备，退出前释放这个设备。

###### bmrt_load_bmodel

```
/*
Parameters: [in] p_bmrt      - Bmruntime that had been created.
            [in] bmodel_path - Bmodel file directory.
Returns:    bool             - true: success; false: failed.
*/
bool bmrt_load_bmodel(void* p_bmrt, const char *bmodel_path);
```

加载bmodel文件，加载后bmruntime中就会存在若干网络的数据，后续可以对网络进行推理。

###### bmrt_load_bmodel_data

```
/*
Parameters: [in] p_bmrt      - Bmruntime that had been created.
            [in] bmodel_data - Bmodel data pointer to buffer.
            [in] size        - Bmodel data size.
Returns:    bool             - true: success; false: failed.
*/
bool bmrt_load_bmodel_data(void* p_bmrt, const void * bmodel_data, size_t size);
```

加载bmodel，不同于bmrt_load_bmodel，它的bmodel数据存在内存中。

2.内存

###### bm_malloc_device_byte

```c
/* [in]  handle
 * [out] pmem
 * [in]  size
 */
bm_status_t bm_malloc_device_byte(bm_handle_t handle, bm_device_mem_t *pmem,
                                  unsigned int size);
```

申请指定大小的device mem，size为device mem的字节大小。当申请成功时，返回BM_SUCCESS；否则返回其他错误码。

###### bm_free_device

```c
/* [in]  handle
 * [out] mem
 */
void bm_free_device(bm_handle_t handle, bm_device_mem_t mem);

//使用参考
// alloc 4096 bytes device mem
bm_device_mem_t mem;
bm_status_t status = bm_malloc_device_byte(bm_handle, &mem, 4096);
assert(status == BM_SUCCESS);
// do things here
......
// if mem will not use any more, free it
bm_free_device(bm_handle, mem);
```

释放device mem。任何申请的device mem，不再使用的时候都需要释放。

###### bm_mem_get_device_size

```c
// [in] mem
unsigned int bm_mem_get_device_size(struct bm_mem_desc mem);
```

得到device mem的大小，以字节为单位。

###### bmrt_tensor_t

**bm_tensor_t**结构体用来表示一个tensor：

```
/*
bm_tensor_t holds a multi-dimensional array of elements of a single data type
and tensor are in device memory */
typedef struct bm_tensor_s {
  bm_data_type_t dtype;
  bm_shape_t shape;
  bm_device_mem_t device_mem;
  bm_store_mode_t st_mode; /* user can set 0 as default store mode */
} bm_tensor_t;
```

###### bmrt_tensor

bmrt_tensor接口可以配置一个tensor

```
void bmrt_tensor(bm_tensor_t* tensor, void* p_bmrt, bm_data_type_t dtype, bm_shape_t shape);
```

此API用于初始化张量。 它将把**deviceMem**分配给 **Tensor**

因此用户应将 `bm_free_device(p_bmrt, tensor->device_mem)`以将其释放。

初始化后，tensor->dtype = dtype, tensor->shape = shape, and tensor->st_mode = 0

| 参数名 | 输入/输出 | 说明                        |
| ------ | --------- | --------------------------- |
| tensor | 输出      | 保存bm_tensor_t指针，不为空 |
| p_bmrt | 输入      | bmruntime的指针，不为空     |
| dtype  | 输入      | 数据类型                    |
| shape  | 输入      | 形状                        |

###### bm_memcpy_s2d

将在**系统内存**上的数据拷贝到**device mem**。系统内存由void指针指定，device mem由bm_device_mem_t类型指定。拷贝成功，返回BM_SUCCESS；否则返回其他错误码。

三种分类（根据拷贝的大小和偏移）

```c
// 拷贝的大小是device mem的大小，从src开始拷贝
/* [in]  handle
 * [out] dst
 * [in]  src
 */
bm_status_t bm_memcpy_s2d(bm_handle_t handle, bm_device_mem_t dst, void *src);
```

```c
// size指定拷贝的字节大小，从src的offset偏移开始拷贝
/* [in]  handle
 * [out] dst
 * [in]  src
 * [in]  size
 * [in]  offset
 */
bm_status_t bm_memcpy_s2d_partial_offset(bm_handle_t handle, bm_device_mem_t dst,
                                         void *src, unsigned int size,
                                         unsigned int offset);
```

```c
// size指定拷贝的字节大小，从src开始拷贝
/* [in]  handle
 * [out] dst
 * [in]  src
 * [in]  size
 */
bm_status_t bm_memcpy_s2d_partial(bm_handle_t handle, bm_device_mem_t dst,
                                  void *src, unsigned int size);
```

###### bm_memcpy_d2s

将device mem中的数据拷贝到系统内存；拷贝成功，返回BM_SUCCESS；否则返回其他错误码。

系统内存由void指针指定，device mem由bm_device_mem_t类型指定。

三种分类（根据拷贝的大小和偏移）

```c
// 拷贝的大小是device mem的大小，从device mem的0偏移开始拷贝
/* [in]  handle
 * [out] dst
 * [in]  src
 */
bm_status_t bm_memcpy_d2s(bm_handle_t handle, void *dst, bm_device_mem_t src);


// size指定拷贝的字节大小，从device mem的offset偏移开始拷贝
/* [in]  handle
 * [out] dst
 * [in]  src
 * [in]  size
 * [in]  offset
 */
bm_status_t bm_memcpy_d2s_partial_offset(bm_handle_t handle, void *dst,
                                         bm_device_mem_t src, unsigned int size,
                                         unsigned int offset);

// size指定拷贝的字节大小，从device mem的0偏移位置开始拷贝
/* [in]  handle
 * [out] dst
 * [in]  src
 * [in]  size
 */
bm_status_t bm_memcpy_d2s_partial(bm_handle_t handle, void *dst,
                                  bm_device_mem_t src, unsigned int size);
```

###### bm_memcpy_d2d

```c
/* [in]  handle
 * [out] dst
 * [in]  dst_offset
 * [in]  src
 * [in]  src_offset
 * [in]  len
 */
bm_status_t bm_memcpy_d2d(bm_handle_t handle, bm_device_mem_t dst, int dst_offset,
                          bm_device_mem_t src, int src_offset, int len);
```

将数据从一个device mem拷贝到另一个device mem。

dst_offset指定目标的偏移，src_offset指定源的偏移，len指定拷贝的大小。

**特别注意**: len是以dword为单位，比如要拷贝1024个字节，则len需要指定为1024/4=256。

###### SOC模式下接口

参考链接

https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/tpu-runtime/reference/html/bmruntime/runtime.html#device-memory-mmap

###### Program synchronize

bm_thread_sync()

同步接口。通常npu推理是异步进行的，用户的cpu程序可以继续执行。本接口用于cpu程序中确保npu推理完成。本章介绍的接口没有特别说明，都是同步接口。只有个别异步接口，需要调用bm_thread_sync进行同步。

```
// [in] handle
bm_status_t bm_thread_sync(bm_handle_t handle);
```

###### bmrt_launch_tensor_ex

对指定的网络，进行npu推理

```
bool bmrt_launch_tensor_ex(void* p_bmrt, const char * net_name,
                           const bm_tensor_t input_tensors[], int input_num,
                           bm_tensor_t output_tensors[], int output_num,
                           bool user_mem, bool user_stmode);
```

