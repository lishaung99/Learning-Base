'''
Description: 
Author: lishuang
Date: 2024-06-27 15:33:24
FilePath: \transformer\print_onnx.py
LastEditTime: 2024-06-27 15:36:58
LastEditors: lishuang
'''
import onnx
import logging
import os


# Load the ONNX model
file = 'model.onnx'
onnx_model = onnx.load(file)

# Check the model's validity
onnx.checker.check_model(onnx_model)

# Print a human-readable representation of the model
# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_file = os.path.splitext(file)[0] + "_onnx_log.txt"  # 根据bmodel参数生成日志文件
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

logging.info(onnx.helper.printable_graph(onnx_model.graph))
