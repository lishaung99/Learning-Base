import torch
import torch.nn as nn
import numpy as np
from decoder_model import Decoder
import os


if not os.path.exists('tmp1'):
    os.makedirs('tmp1')

# 模型参数
d_model = 128
n_heads = 4
d_k = d_v = 32
d_ff = 512
n_layers = 2
max_len = 100
dropout = 0.1
tgt_vocab_size = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Decoder().to(device)
model = torch.load('decoder_model1.pth', map_location=device)
model.eval()

class Embedding(nn.Module):
    def __init__(self, embedding, pos_encoding):
        super(Embedding, self).__init__()
        self.embedding = embedding
        self.pos_encoding = pos_encoding

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        return self.pos_encoding(embeddings)

class DecoderLayerWrapper(nn.Module):
    def __init__(self, layer):
        super(DecoderLayerWrapper, self).__init__()
        self.layer = layer

    def forward(self, dec_inputs, dec_self_attn_mask):
        return self.layer(dec_inputs, dec_self_attn_mask)

# 创建模型实例
decoder = Decoder()

embedding = Embedding(decoder.tgt_emb, decoder.pos_emb)
input_ids = torch.randint(0, tgt_vocab_size, (1, max_len), dtype=torch.long)
torch.onnx.export(
    embedding,
    input_ids,
    './tmp1/embedding.onnx',
    verbose=True,
    input_names=['input_ids'],
    output_names=['embeddings'],
    opset_version=15
)
# 导出解码器层
for i, layer in enumerate(decoder.layers):
    decoder_layer_wrapper = DecoderLayerWrapper(layer)
    dec_inputs = torch.randn(1, max_len, d_model)
    dec_self_attn_mask = torch.zeros(1, max_len, max_len, dtype=torch.bool)
    torch.onnx.export(
        decoder_layer_wrapper,
        (dec_inputs, dec_self_attn_mask),
        f'./tmp1/decoder_layer_{i}.onnx',
        verbose=True,
        input_names=['dec_inputs', 'dec_self_attn_mask'],
        output_names=['dec_outputs', 'dec_self_attn'],
        opset_version=15
    )

# 导出投影层
projection_layer = decoder.projection
dummy_input = torch.randn(1, max_len, d_model)
torch.onnx.export(
    projection_layer,
    dummy_input,
    './tmp1/projection.onnx',
    verbose=True,
    input_names=['dec_outputs'],
    output_names=['logits'],
    opset_version=15
)