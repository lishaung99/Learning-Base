'''
Description: 
Author: lishuang
Date: 2024-06-27 14:01:53
FilePath: \\transformer\\print_model.py
LastEditTime: 2024-06-27 14:10:53
LastEditors: lishuang
'''
import torch
from transformer import Transformer

# Load the saved model
model = torch.load('model.pth')

# Print the model structure
print(model)
