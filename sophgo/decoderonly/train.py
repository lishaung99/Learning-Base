import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from datasets import make_data, MyDataSet
from decoder_model import Decoder

def create_padding_mask(seq, max_len=100):
    seq_len = seq.size(1)
    padding_mask = torch.ones(seq.size(0), max_len, dtype=torch.bool)
    padding_mask[:, :seq_len] = False
    return padding_mask

def pad_sequence(seq, max_len=100):
    return torch.nn.functional.pad(seq, (0, max_len - seq.size(1)), "constant", 0)

if __name__ == "__main__":
    # 数据准备
    enc_inputs, dec_inputs, dec_outputs = make_data()
    print(f"Encoder inputs shape: {enc_inputs.shape}")
    print(f"Decoder inputs: {dec_inputs}")
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Decoder().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略占位符
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    # 训练模型
    for epoch in range(50):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)

            # 填充输入序列和目标序列到固定长度
            dec_inputs = pad_sequence(dec_inputs)
            dec_outputs = pad_sequence(dec_outputs)

            # 创建填充掩码
            dec_self_attn_mask = create_padding_mask(dec_inputs)

            # 模型前向传递
            outputs, dec_self_attns = model(dec_inputs, dec_self_attn_mask)

            # 调整输出形状以匹配目标形状
            loss = criterion(outputs, dec_outputs.view(-1))
            print(f'Epoch: {epoch + 1:04d}, Loss: {loss:.6f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 保存模型
    torch.save(model, 'decoder_model1.pth')
    print("模型已保存")
