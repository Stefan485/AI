import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torchtune
import math
import time
import llama.tokenizer as tokenizer


class SwiGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

@dataclass
class Config:
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 6
    block_size: int = 4096
    vocab_size: int = 128000 # 

class FeedForward(nn.Module):
    
        def __init__(self, config):
            super().__init__()
            self.norm = torchtune.modules.RMSNorm(config.n_embd)
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.swiglu = SwiGLU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
            self.drop = nn.Dropout(0.1)
    
        def forward(self, x):
            x = self.norm(x)
            x = self.c_fc(x)
            x = self.swiglu(x)
            x = self.c_proj(x)
            x = self.drop(x)
            return x
        

class CausalSelfAttention(nn.Module):
    
        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0

            self.norm = torchtune.modules.RMSNorm(config.n_embd)
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd)
            self.drop = nn.Dropout(0.1)
    
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                         .view(1, 1, config.block_size, config.block_size))
            
    
        def forward(self, x):
                x = self.norm(x)
                B, T, C = x.size()
    
                qkv = self.c_attn(x)
    
                q, k, v = qkv.split(self.n_embd, dim=2)
    
                k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    
                #flash attention not avalible
                # xformers.components.attention.ScaledDotProduct() used in llama for efficency
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
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.ln2 = nn.LayerNorm(config.n_embd)
            self.mlp = FeedForward(config)
    
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x
        

class Llama(nn.Module):
      
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = torchtune.modules.RotaryPositionalEmbeddings(config.n_embd)
        self.drop = nn.Dropout(0.1)
        self.blocks  = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = torchtune.modules.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight

    def forward(self, x, targets=None):

        x = self.tok_emb(x)
        x += self.pos_emb(x)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return x
    

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

        #llama 3 tokenizer
        enc = tokenizer.Tokenizer()
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
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Llama(Config()).to(device)

max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 10 #lama uses 2000, lowered for experimentation
max_steps = 50

def get_lr(step):
     
     if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
     
     if step > max_steps:
         return min_lr
     
     decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
     assert 0 <= decay_ratio <= 1
     coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
     return min_lr + coeff * (max_lr - min_lr)

B = 1
T = 4096
loader = DataLoaderLite(B, T) # Batch size, sequence length
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.95), eps=1e-9, weight_decay=0.1)

for step in range(max_steps) :
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
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    print(f'step: {step}, loss: {loss.item(),} dt: {dt:.2f}ms')


model.eval()

num_return_sequence = 5
max_length = 50
enc = tokenizer.Tokenizer()
tokens = enc.encode('LLama test')
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1)
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


for i in range(num_return_sequence):
    tokens = x[i, :max_length].tolist()
    decode = enc.decode(tokens)
    print('>', decode)
