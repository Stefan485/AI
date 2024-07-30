import torch
import tiktoken
from torch.nn import functional as F
from gpt2 import GPT, GPTconfig
import time
import wandb

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

enc = tiktoken.get_encoding('gpt2')
config = GPTconfig()

torch.manual_seed(42)
torch.cuda.manual_seed(42)

class DataLoaderLite:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('../input.txt', 'r', encoding='utf-8') as f:
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

num_return_sequece = 5
max_length = 50

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(f'using device: {device}')
B = 4
T = 512
train_loader = DataLoaderLite(B, T)
torch.set_float32_matmul_precision('high')

# model = GPTCustom.from_pretrained('gpt2')
model = GPT(GPTconfig())
model = model.to(device)
# model = torch.compile(model)
def get_lr(step):
    if step == 0:
        return 3e-4
    return config.n_embd ** -0.5 * min(step ** -0.5, step * 4000 ** -1.5)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)

for step in range(max_length):
    t0 = time.time()

    x, y = train_loader.next_batch()    
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    metrics = {'loss': loss.item(), 'lr': get_lr(step)}
    if step + 1 < max_length:
        wandb.log(metrics)
    print(f'step: {step}, loss: {loss.item(),} dt: {dt:.2f}ms')

tokens = enc.encode("Hello I'm a language model,")
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
    print("> ", decode)

wandb.finish()