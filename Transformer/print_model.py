'''
Description: 
Author: lishuang
Date: 2024-06-27 14:01:53
FilePath: \transformer\print_model.py
LastEditTime: 2024-06-27 16:00:24
LastEditors: lishuang
'''
import torch
from transformer import Transformer
import logging
import os

file = 'model.pth'

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_file = os.path.splitext(file)[0] + "_struct_log.txt"  # 根据bmodel参数生成日志文件
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)


# Load the saved model
model = torch.load(file)

# Print the model structure
logging.info(model)
