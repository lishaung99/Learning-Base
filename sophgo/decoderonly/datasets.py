import torch
import torch.utils.data as Data

# 数据预处理
sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],        
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  
             ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']]                

src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}
src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)
tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9}
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}
tgt_vocab_size = len(tgt_vocab) # 10
print(f'tgt_vocab_size: {tgt_vocab_size}\n') 
src_len = len(sentences[0][0].split(" ")) 
tgt_len = len(sentences[0][1].split(" ")) 

def make_data():
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

# 自定义数据集
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
