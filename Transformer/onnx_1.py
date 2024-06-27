# from transformer import *
import torch
import os
import torch.onnx
from transformer import *


model = origin_model = torch.load('model.pth')
origin_model.eval()

def convert_to_onnx(model):
    # 创建示例输入
    # nc_inputs_example = torch.randn(batch_size, nc_inputs_size)
    nc_inputs_example = torch.tensor([[1, 2, 3, 4, 0],[1, 5, 6, 3, 7],[1, 2, 8, 4, 0]])
    dec_inputs_example = torch.tensor([[1, 3, 4, 5, 6],[1, 3, 7, 8, 0],[1, 3, 4, 5, 9]])
    # dec_inputs_example = torch.randn(batch_size, dec_inputs_size)
    # 导出为ONNX格式
    torch.onnx.export(
        model=origin_model,
        args=(nc_inputs_example, dec_inputs_example),
        f="model.onnx",
        verbose=True,
        input_names=["nc_inputs", "dec_inputs"],
        output_names=["output"],
        dynamic_axes={
            "nc_inputs": {0: "batch_size"},
            "dec_inputs": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=12  # 指定ONNX的版本
    )

result = convert_to_onnx(model)
print("Done !")

