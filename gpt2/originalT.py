import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time
import tiktoken
import math
import xformers

@dataclass
class Config:
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 6
    block_size: int = 768
    vocab_size: int = 50304 # 50257 -> 50304 somehow takes less memory (50257 leads to cuda out of memory)

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x
    

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.drop = nn.Dropout(0.1)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        

    def forward(self, x):
            
            B, T, C = x.size()

            qkv = self.c_attn(x)

            q, k, v = qkv.split(self.n_embd, dim=2)

            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            #flash attention not avalible
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.c_proj(y)
            y = self.drop(y)
            return y
    

class Block(nn.Module):
     
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ff = FeedForward(config)
        
    def forward(self, x):
        x =  self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.ff(x))
        return x
    

class Original(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    #By GPT2 initial loss from 150 -> 10
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx)
        x = x + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        loss = None
        logits = self.head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device):

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nondecay_params, "weight_decay": 0.0},
        ]


        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-9, fused=True)
        return optimizer


class DataLoaderLite:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding('gpt2')
        #ds = load_dataset("breadlicker45/youtube-comments-v2")

        # data = ds['train']['text']
        # text = ''.join(' ' + x for x in data)
        # f = open('input.txt', 'w', encoding='utf-8')
        # f.write(text)
        # f.close()
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded: {len(self.tokens)}")
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T

        if self.current_position + B * T + 1 >= len(self.tokens):
            self.current_position = 0
        return x, y    
    

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')

config = Config()
model = Original(config).to(device)

# model = torch.compile(model)
B = 4
T = 768

loader = DataLoaderLite(B, T) # Batch size, sequence length
max_length = 50
num_return_sequece = 5
#per gpt2, as bathc size is not given in original paper
# Should optimize things, but unobservable on NVDIA 1650
torch.set_float32_matmul_precision('high')

#No visible improvment
def get_lr(step):
    if step == 0:
        return 3e-4
    return config.n_embd ** -0.5 * min(step ** -0.5, step * 4000 ** -1.5)

#gpt2 uses AdamW original paper uses Adam
#swaped for weight decay per gpt2
optimizer = model.configure_optimizers(0.1, 3e-4, device)


for i in range(100):
    t0 = time.time()
    # loss_accum = 0
    # for micro_step in range(grad_accum_steps):
    x, y = loader.next_batch()    
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x, y)
    # loss = loss / grad_accum_steps
    # loss_accum += loss.detach()
    loss.backward()
    # per gpt2
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # lr = get_lr(i)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    print(f'step: {i}, loss: {loss.item(),} dt: {dt:.2f}ms')


model.eval()

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode('Music is the way')
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequece, 1)
x = tokens.to(device)

while x.size(1) < max_length:

    with torch.no_grad():

        logits, loss = model(x)

        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)

        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x, xcol), dim=1)


for i in range(num_return_sequece):
    tokens = x[i, :max_length].tolist()
    decode = enc.decode(tokens)
    print('>', decode)