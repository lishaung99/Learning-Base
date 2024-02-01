## 部署示例模型

1.部署mlir时无法顺利执行model_transform.py

![image-20230918104643483](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230918104643483.png)

**问题1**. 基于YOLACT的目标跟踪算法移植与测试

模型转化阶段：ONNX转MLIR

执行命令：sh onnx2mlir.sh

提示：ImportError:cannotimport name'ir'from 'mlir. mlirlibs.mlir'(unknown location）

尝试解决：问题文件为model_transforme.py，在python中可以import mlir，无法import mlir._mlir_libs，该文件下ir只有ir.pyi。（提示为修改过文件地址之后的，未修改显示no module）

**解决**：docker的镜像错误

## 官方部署Llama-tpu

**问题2**：无法source文件

```
$’\r’: command not found
原因：windows中的换行符是 \r，linux中的是\n。
```

```
通过安装dos2unix解决
sudo apt-get install dos2unix
dos2unix envsetup.sh
```

**问题3：**./build.sh失败 


```
-bash: /home/xxx.sh: /bin/bash^M: bad interpreter: No such file or directory
```

**解决**：格式问题

```
windows格式的文件行尾为^M$，unix格式的文件行尾为$
1.查看文件格式
cat -A xxx.sh
：dos格式的文件行尾为^M$，unix格式的文件行尾为$
2.使用下述命令直接替换结尾符为unix格式
sed -i "s/\r//" build.sh
：再次使用1查看文件行尾
```

**问题4**: ./build.sh显示权限错误

```
bash: ./build.sh: Permission denied

解决： chmod 777 build.sh
```

结果

![image-20231007155027806](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231007155027806.png)

Llama2-TPU
编译模型阶段：导出所有onnx模型
执行命令：python3 export_onnx.py
提示：killed
尝试：执行时修改了源文件的地址，执行时同时查看动态内存，到达内存满的临界状态显示killed。服务器内存15g。
**问题**：  **错误**：内存不足

查看sc7使用：bm-smi

**问题5*：红色E

需要重新更新

```
sudo apt-get update
```

![image-20230927095621368](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20230927095621368.png)



**问题**：无法加载权重文件

![image-20231007163825867](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231007163825867.png)

权重文件下载时出现错误，需要重新下载

**问题**：/usr/bin/env: ‘python3\r’: No such file or directory

```
dos2unix model_transform.py
dos2unix model_deploy.py
```

![image-20231008101029697](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231008101029697.png)

问题：无法加载mlir

![image-20231008140614595](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231008140614595.png)

**尝试**  ：重新build加载mlir包 

![image-20231008140911463](C:\Users\carbi\AppData\Roaming\Typora\typora-user-images\image-20231008140911463.png)