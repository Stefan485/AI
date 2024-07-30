import torch
import tiktoken
from torch.nn import functional as F
from gpt2 import GPT, GPTconfig
import time
import math
import wandb

max_lr = 6e-4
min_lr = max_lr * 0.1
num_return_sequece = 5
max_new_tokens = 50
max_steps = 100
warmup_steps = 10
enc = tiktoken.get_encoding('gpt2')
config = GPTconfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')
B = 2 # batch size
T = 1024 # sequence length
torch.manual_seed(42)
torch.cuda.manual_seed(42)


class DataLoaderLite:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('../input2.txt', 'r', encoding='utf-8') as f:
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


train_loader = DataLoaderLite(B, T)
torch.set_float32_matmul_precision('high')

model = GPT(config)
# model = torch.nn.Module.compile(model) possible optim?
model = model.to(device)

wandb.login()

wandb.init(project='gpt2',
           config={
                'n_embd': 288,
                'n_head': 12,
                'n_layer': 12,
                'block_size': 512,
                'vocab_size': 50304,
                'dropout': 0.1
           })

def get_lr(step):
    if step < warmup_steps:
        return 3e-4 * (step + 1) / warmup_steps
    
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr) * coeff

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)


for step in range(max_steps):
    
    # wandb.watch(model, log='parameters', log_freq=2) slowdown goes brrrr 300ms -> 8000ms
    
    t0 = time.time()

    x, y = train_loader.next_batch()    
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = {'loss': loss.item(), 'lr': lr}
    
    if step + 1 < max_steps:
        wandb.log(metrics)
    print(f'step: {step}, loss: {loss.item(),} dt: {dt:.2f}ms')


# Generate text
tokens = enc.encode("Hello I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequece, 1)
x = tokens.to(device)

model.eval()
while x.size(1) < max_new_tokens:
    with torch.no_grad():

        logits, loss = model(x)

        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)

        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x, xcol), dim=1)


for i in range(num_return_sequece):
    tokens = x[i, :max_new_tokens].tolist()
    decode = enc.decode(tokens)
    print("> ", decode)

wandb.finish()