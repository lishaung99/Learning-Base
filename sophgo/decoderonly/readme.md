## 文件目录

```shell
.
│  datasets.py        # 数据集脚本
│  decoder_model.py		# 模型脚本
│  onnx_block.py 			# pytorch转onnx脚本
│  print_strcut.py		# 打印模型结构脚本
│  train.py						# 训练脚本
│
└─tmp1	#	onnx保存文件夹
        decoder_layer_0.onnx
        decoder_layer_1.onnx
        embedding.onnx
        projection.onnx
```

## **步骤**

1. ``python train.py`` 得到模型权重文件 " decoder_model.pth"

2. ``python onnx_block`` 得到onnx文件

3. 进入一个算能得docker即可

   首先``source ensource.sh `` 配置环境，防止 ``model_transform.py ``和 ``model_deploy.py`` 找不到文件

   `` vi bmodel.sh ``  并保存

   ```shell
   folder="tmp1"
   name="decoder_only"
   out_model=$name.bmodel
   outdir=${folder}/embedding
   mkdir -p $outdir
   pushd $outdir
   
   
   model_transform.py \
       --model_name embedding \
       --model_def ../embedding.onnx \
       --input_shapes [[1,100]] \
       --input_types "int32" \
       --mlir embedding.mlir
   
   model_deploy.py \
       --mlir embedding.mlir \
       --quantize F16 \
       --quant_input \
       --quant_output \
       --chip bm1684x \
       --model embedding.bmodel \
       --debug
   
   model_transform.py \
       --model_name embedding_cache \
       --model_def ../embedding.onnx \
       --input_shapes [[1,1]] \
       --input_types "int32" \
       --mlir embedding_cache.mlir
   
   model_deploy.py \
       --mlir embedding_cache.mlir \
       --quantize F16 \
       --quant_input \
       --quant_output \
       --chip bm1684x \
       --model embedding_cache.bmodel \
       --debug
   
   rm *.npz
   
   models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '
   
   popd
   
   echo $models
   
   outdir=${folder}/projection
   mkdir -p $outdir
   pushd $outdir
   
   model_transform.py \
       --model_name projection \
       --model_def ../projection.onnx \
       --input_shapes [[1,100,128]] \
       --mlir projection.mlir
   
   model_deploy.py \
       --mlir projection.mlir \
       --quantize W8F16 \
       --quant_input \
       --chip bm1684x \
       --num_device 1 \
       --model projection.bmodel \
       --debug
   
   rm *.npz
   models=${models}${outdir}'/projection.bmodel '
   popd
   echo $models
   
   outdir=${folder}/block
   mkdir -p $outdir
   pushd $outdir
   
   
   for ((i=0; i<=1; i++)); do
       model_transform.py \
           --model_name decoder_layer$i \
           --model_def ../decoder_layer_$i.onnx \
           --mlir decoder_layer_$i.mlir
   
       model_deploy.py \
           --mlir decoder_layer_$i.mlir \
           --quantize W8F16 \
           --quant_input \
           --quant_output \
           --chip bm1684x \
           --num_device 1 \
           --model decoder_layer_$i.bmodel \
           --debug
       rm *.npz
       models=${models}${outdir}'/decoder_layer_'$i'.bmodel ' 
       
   
   done
   popd
   echo $models
   
   model_tool --combine $models -o $out_model
   ```

4. `` ./bmodel.sh `` 得到bmodel文件    decoder_model.bmodel

5. 新开一个终端，在bmodel的文件目录下执行下面的命令(docker外部)

   ```shell
   BMRUNTIME_ENABLE_PROFILE=1 bmrt_test --bmodel decoder_model.bmodel
   ```

   同时在当前目录生成`bmprofile_data-1`文件夹, 为全部的Profile数据。

6. 在docker内部执行

   ```shell
   tpu_profile.py bmprofile_data-1 bmprofile_out
   ```

7. 执行 bmprofile_out 文件下的 result.html 即可
https://github.com/lishaung99/Learning-Base/blob/master/sophgo/decoderonly/image/Snipaste_2024-07-01_17-55-30.png

![sophgo/decoderonly/image/Snipaste_2024-07-01_17-55-30.png](sophgo/decoderonly/image/Snipaste_2024-07-01_17-55-30.png)
