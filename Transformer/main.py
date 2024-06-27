import torch.nn as nn
import torch.optim as optim
from datasets import *
from transformer import Transformer
import logging
import os
def log(mode_name):
    # 日志配置
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 使用os.path.join安全地构造日志文件路径
    log_file = os.path.join(os.getcwd(), mode_name + "_train_log.txt")  # 根据mode_name参数生成日志文件于当前工作目录下
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

if __name__ == "__main__":
    model_name = "Transformers"
    log(model_name)
    enc_inputs, dec_inputs, dec_outputs = make_data()
    logging.info(enc_inputs.shape)
    logging.info(dec_inputs)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)         # 忽略 占位符 索引为0.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(50):
        for enc_inputs, dec_inputs, dec_outputs in loader: 
            # enc_inputs : [batch_size, src_len]
            # dec_inputs : [batch_size, tgt_len]
            # dec_outputs: [batch_size, tgt_len]
            enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            logging.info(f'Epoch: {epoch + 1:04d}, loss = {loss:.6f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model, 'model.pth')
    # 获取保存的模型文件的完整路径
    model_path = os.path.abspath('model.pth')
    logging.info(f"Path = {model_path}")
