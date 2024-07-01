import torch
import torch.nn as nn
import numpy as np
# from datasets import tgt_vocab_size

d_model = 128
n_heads = 4
d_k = d_v = 32
d_ff = 512
n_layers = 2
max_len = 100
dropout = 0.1
tgt_vocab_size = 10

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False) # 128, 32 * 4 =128
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(d_k)
        
        if mask is not None:
            scores.masked_fill_(mask, -1e9)
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v_s).transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model)(output + Q), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, x):
        return nn.LayerNorm(d_model)(self.fc(x) + x)

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, dec_inputs, dec_self_attn_mask):
        dec_inputs = self.pad_to_max_len(dec_inputs)
        dec_self_attn_mask = self.pad_to_max_len(dec_self_attn_mask, mask=True)

        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns

    def pad_to_max_len(self, x, mask=False):
        # 固定输入长度
        max_len = 100
        if mask:
            return torch.nn.functional.pad(x, (0, max_len - x.size(1)), "constant", 1)
        else:
            return torch.nn.functional.pad(x, (0, max_len - x.size(1)), "constant", 0)

# batch_size = 1
# seq_len = 17
# dec_inputs = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))
# dec_self_attn_mask = torch.ones(batch_size, seq_len).bool()

# decoder = Decoder()
# logits, attns = decoder(dec_inputs, dec_self_attn_mask)
# print(logits.shape)  # 输出形状应为 (batch_size * max_len, tgt_vocab_size)
# print(attns[0].shape)  # 输出注意力形状
