import torch
import tiktoken
from torch.nn import functional as F
import gpt2
import model_karpathy
import math


max_lr = 6e-4
min_lr = max_lr * 0.1
num_return_sequece = 5
max_new_tokens = 50
max_steps = 50
warmup_steps = 10
enc = tiktoken.get_encoding('gpt2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')
B = 2 # batch size
T = 1024 # sequence length
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision('high')

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
# my_gpt = gpt2.GPT(gpt2.GPTconfig())
my_gpt = gpt2.GPT.from_pretrained('gpt2')
# karpathy_gpt = model_karpathy.GPT(model_karpathy.GPTConfig())
karpathy_gpt = model_karpathy.GPT.from_pretrained('gpt2')
my_gpt = my_gpt.to(device)
karpathy_gpt = karpathy_gpt.to(device)

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr) * coeff

optimizer_my_gpt = my_gpt.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device_type=device)
optimizer_karpathy_gpt = karpathy_gpt.configure_optimizers(weight_decay=0.1, learning_rate=6e-4,
                                                            betas=(0.9, 0.95), device_type=device, zero_stage=0)

def compare_weights(weights1, weights2):
    assert weights1.keys() == weights2.keys(), 'Models have different parameters'
    differences = {}
    # tensors_of_intrest = ['mlp.c_proj', 'mlp.c_fc']
    # keys = []
    # for x in weights1.keys():
        # if any(y in x for y in tensors_of_intrest):
            # keys.append(x)
   
    # print(keys)
    # s = set(keys)
    print(weights1.keys())
    s = set(weights1.keys())
    assert all(torch.allclose(t1, t2) and k1 == k2 for (k1, t1), (k2, t2) #Doesn't pass the test
                in zip(weights1.items(), weights2.items())), 'Models have different weights'
    print("passed")
    for k in s:
        w1 = weights1[k]
        w2 = weights2[k]
        abs_error = torch.abs(w1 - w2)
        rel_error = abs_error / (torch.abs(w1) + 1e-5)
        abs_mean = abs_error.mean()
        rel_mean = rel_error.mean()
        if abs_mean and rel_mean != 0.0:
            differences[k] = f'abs_mean: {abs_mean:.4f}, rel_mean: {rel_mean:.4f}'

    differences = dict(sorted(differences.items()))
    print(differences)


for step in range(1):
    print(f'step: {step}')
    weights1, weights2 = my_gpt.state_dict(), karpathy_gpt.state_dict()
    # compare_weights(weights1, weights2)
    x, y = train_loader.next_batch()    
    x, y = x.to(device), y.to(device)

    optimizer_my_gpt.zero_grad()
    logits, loss = my_gpt(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(my_gpt.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer_my_gpt.param_groups:
        param_group['lr'] = lr
    optimizer_my_gpt.step()

    torch.cuda.synchronize()

    optimizer_karpathy_gpt.zero_grad()
    logits, loss = karpathy_gpt(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(karpathy_gpt.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer_karpathy_gpt.param_groups:
        param_group['lr'] = lr
    optimizer_karpathy_gpt.step()

    torch.cuda.synchronize()

#transformer.h.0.ln_1
#transformer.h.0.attn
#transformer.h.0.mlp.c_fc
#transformer.h.0.mlp.c_proj
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

my_gpt.transformer.h[0].ln_1.register_forward_hook(get_activation('my_gpt.transformer.h.0.ln_1'))
my_gpt.transformer.h[0].attn.register_forward_hook(get_activation('my_gpt.transformer.h.0.attn'))
my_gpt.transformer.h[0].mlp.c_fc.register_forward_hook(get_activation('my_gpt.transformer.h.0.mlp.c_fc'))
my_gpt.transformer.h[0].mlp.c_proj.register_forward_hook(get_activation('my_gpt.transformer.h.0.mlp.c_proj'))
karpathy_gpt.transformer.h[0].ln_1.register_forward_hook(get_activation('karpathy_gpt.transformer.h.0.ln_1'))
karpathy_gpt.transformer.h[0].attn.register_forward_hook(get_activation('karpathy_gpt.transformer.h.0.attn'))
karpathy_gpt.transformer.h[0].mlp.c_fc.register_forward_hook(get_activation('karpathy_gpt.transformer.h.0.mlp.c_fc'))
karpathy_gpt.transformer.h[0].mlp.c_proj.register_forward_hook(get_activation('karpathy_gpt.transformer.h.0.mlp.c_proj'))

my_gpt.eval()
karpathy_gpt.eval()
tokens = enc.encode("Hello I'm a ")
tokens = torch.tensor(tokens, dtype=torch.long)

output = my_gpt(tokens.unsqueeze(0).to(device))
print(activation['my_gpt.transformer.h.0.ln_1'])
output = karpathy_gpt(tokens.unsqueeze(0).to(device))
print(activation['karpathy_gpt.transformer.h.0.ln_1'])

print('----------------------------SECOND TEST---------------------------------')

tokens = enc.encode("Hello I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)

output = my_gpt(tokens.unsqueeze(0).to(device))
print(activation['my_gpt.transformer.h.0.ln_1'])
output = karpathy_gpt(tokens.unsqueeze(0).to(device))
print(activation['karpathy_gpt.transformer.h.0.ln_1'])
