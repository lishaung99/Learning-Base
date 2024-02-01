# LLM转ONNX思路与步骤（TPU）

## 加载模型

1. 通过huggingface接口加载模型
2. 定义变量从config.json接受模型参数，包括层数，注意力头数，每层的隐藏层数等

```python
folder = "./tmp"
model_path = "llama-2-7b-chat-hf"

origin_model = LlamaForCausalLM.from_pretrained(model_path)
origin_model.eval()
transformer = origin_model.model
config = origin_model.config

MAX_LEN = 512
for param in origin_model.parameters():
    param.requires_grad = False

num_layers = config.num_hidden_layers
hidden_size = config.hidden_size
num_attention_heads = config.num_attention_heads
head_dim = hidden_size // num_attention_heads
layers = transformer.layers
tokenizer = LlamaTokenizer.from_pretrained(model_path)
```

注意：如果缺失部分依赖包，缺什么就安装什么

例如，转换phi-2缺失包 einops ： pip install einops

## 层初始定义处理

### 嵌入层

```python
class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return transformer.embed_tokens(input_ids)
```

### 注意力头

```python
class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits, 1)
        return token
```

注：**topk:取数组的前k个元素进行排序**

### 块

```python
class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            use_cache=True)
        past_k, past_v = past_kv
        return hidden_states, past_k, past_v
```

### 缓存

```python
class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True)
        past_k, past_v = past_kv
        return hidden_states, past_k, past_v

```

### 

## 分层次转换

### 嵌入层转换

```python
def convert_embedding():
    model = Embedding()
    torch.onnx.export(model, (torch.tensor([0, 1, 2, 3])),
                      f'./tmp/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      dynamic_axes={"input_ids": {
                          0: "length"
                      }},
                      do_constant_folding=True,
                      opset_version=15)
```

### 注意力头的转换

```python
def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, hidden_size)
    torch.onnx.export(model, (input),
                      f'./tmp/lm_head.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['token'],
                      do_constant_folding=True,
                      opset_version=15)
```

### 块的转换

```python
def convert_block(layer_id):
    # input
    # MAX_LEN + 1 for model combine
    hidden_states = torch.randn((1, MAX_LEN, hidden_size))
    position_ids = torch.tensor([range(MAX_LEN)], dtype=torch.long)
    attention_mask = -1000 * torch.ones((1, 1, MAX_LEN, MAX_LEN), dtype=torch.float32).triu(diagonal=1)
    model = Block(layer_id)
    # hiddeng_states = model(input_ids, position_ids)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'./tmp/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)
```

### 块缓存的转换

```python
def convert_block_cache(layer_id):
    # input
    hidden_states = torch.randn((1, 1, hidden_size))
    position_ids = torch.tensor([range(1)], dtype=torch.long)
    attention_mask = -1000 * torch.ones((1, 1, 1, MAX_LEN + 1), dtype=torch.float32).triu(diagonal=0)
    past_k = torch.randn((1, MAX_LEN, num_attention_heads, head_dim))
    past_v = torch.randn((1, MAX_LEN, num_attention_heads, head_dim))
    model = BlockCache(layer_id)
    # hiddeng_states = model(input_ids, position_ids)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'./tmp/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'position_ids', 'attention_mask', 'history_k',
            'history_v'
        ],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)
```

## 整体转换架构

```python
# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
for i in range(num_layers):
    print("convert_block_{}".format(i))
    convert_block_cache(i)
    convert_block(i)
convert_embedding()
convert_lm_head()
```



## 总结

转换只涉及到模型参数的加载与使用，没有改写模型的架构关系