import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from transformers import AutoTokenizer
import time
import constants
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


parser = argparse.ArgumentParser(description='This is a demonstration program')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = constants.BATCH_SIZE
block_size = constants.BLOCK_SIZE
max_iters = constants.MAX_ITERS
learning_rate = constants.LEARNING_RATE
eval_iters = constants.EVAL_ITER
n_embd = constants.n_embed
n_head = constants.n_head
n_layer = constants.N_LAYER
dropout = constants.dropout

class MHFlashAttn(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_head=n_head):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd//n_head, head_size, bias=False, dtype=torch.float16)
        self.query = nn.Linear(n_embd//n_head, head_size, bias=False, dtype=torch.float16)
        self.value = nn.Linear(n_embd//n_head, head_size, bias=False, dtype=torch.float16)

        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, n_head=n_head):
        B,T,C = x.shape
        x = x.view(B, T, n_head, C//n_head) # (B,T,C) -> (B,T,hs,C/hs)
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True,
                              window_size=(-1, -1), alibi_slopes=None, deterministic=False)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.mha_flash_attn = MHFlashAttn(head_size)
        # self.proj = nn.Linear(head_size * num_heads, n_embd, dtype=torch.float16)
        self.proj = nn.Linear(head_size, n_embd, dtype=torch.float16)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.mha_flash_attn(x)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, dtype=torch.float16),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, dtype=torch.float16),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        # head_size = n_embd // n_head
        head_size = n_embd
        x_size = torch.tensor([batch_size, block_size, n_head, n_embd])
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd, head_size, dtype=torch.float16)
        self.ln2 = nn.LayerNorm(n_embd, head_size, dtype=torch.float16)
    
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, dtype=torch.float16)
        self.position_embedding_table = nn.Embedding(block_size, n_embd, dtype=torch.float16)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size, dtype=torch.float16)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index
