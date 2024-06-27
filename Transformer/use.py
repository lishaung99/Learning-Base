'''
Description: 
Author: lishuang
Date: 2024-06-27 14:19:51
FilePath: \\transformer\\use.py
LastEditTime: 2024-06-27 14:19:57
LastEditors: lishuang
'''
import torch
from transformer import Transformer  # Make sure this imports the Transformer class correctly

# Load the trained model
model = torch.load('model.pth')
model.eval()  # Set the model to evaluation mode

# Example input data preparation
# Assuming src_vocab_size and tgt_vocab_size are defined, and the same as used during training
src_vocab_size = 10000  # Example value, replace with actual
tgt_vocab_size = 10000  # Example value, replace with actual
enc_inputs = torch.tensor([[1, 2, 3, 4, 5]])  # Example input sequence, replace with actual data
dec_inputs = torch.tensor([[1, 2, 3, 4]])  # Example input sequence for the decoder, replace with actual data

# Move data to the same device as the model (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc_inputs = enc_inputs.to(device)
dec_inputs = dec_inputs.to(device)
model.to(device)

# Make predictions
with torch.no_grad():  # Disable gradient calculation for inference
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

# Process the output
predicted_tokens = torch.argmax(outputs, dim=1)  # Get the index of the highest probability token
predicted_tokens = predicted_tokens.view(enc_inputs.size(0), -1)  # Reshape to batch_size x tgt_len

# Print the predicted tokens
print("Predicted Tokens:", predicted_tokens)
