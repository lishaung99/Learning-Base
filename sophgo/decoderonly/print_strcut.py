import torch
from decoder_model import Decoder
import logging
import os

file = 'decoder_model1.pth'

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