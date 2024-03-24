""" Baby's First Large Language Model """
import sys
from typing import Tuple, Callable
from torchtyping import TensorType
import torch
from torch import nn
from torch.nn import functional as F

BLOCK_SIZE = 8
BATCH_SIZE = 4
MAX_ITERS = 1000
EMBEDDING_DIM = 100
SPLIT_SIZE = 0.8
LEARNING_RATE = .1
EVAL_ITER = 100
n_embed = 384
n_head = 4
dropout = 0.2

class BigramLanguageModel(nn.Module):
    """ 
        Implements a GPT language model. Predicts the next character in a text.
        Very naieve implementation. First try.
    """

    def __init__(self, vocab_size: int):
        """ Initialize method. vocab_size is the number of distinct characters in the text """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_emed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, index: int, targets: TensorType = None):
        """ mandatory implementation of the forward method """
        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arrange(T, 
                                                              device=device))
        x_int = tok_emb + pos_emb
        x_int = self.blocks(x_int)
        x_int = self.ln_f(x_int)
        logits = self.lm_head(x_int)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index: TensorType, max_new_tokens: int) -> int:
        """ Predicts max_new_tokens starting at the token at index """
        for _ in range(max_new_tokens):
            logits, _ = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=-1)
        return index

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embed, n_head):
        """ n_embed: embedding dimension, n_head: number of heads we'd like """
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

def tokenize_txt(text_file_name: str) -> Tuple[str, int, Callable, Callable]:
    # pylint: disable-msg=unnecessary-comprehension
    # pylint: disable-msg=unnecessary-lambda-assignment
    """ 
        Opens file, reads text and converts it to a list of 
        tokens. Creates 'encode' and 'decode' lambda functions.

        Parameters:
            text_file_name: the name of the text file. If the text file
                            lives in a subdirectory, use the full path
                            ex: './data/woz.txt'
        Returns:
            txt: the text
            encode: lambda function that takes a string and returns a list 
                    of integers where each integer corresponds to a tokenized
                    value of a letter in the string
            decode: lambda fuunction that takes a list of integers and returns
                    the corresponding string.
    """
    try:
        with open(text_file_name, "r", encoding="utf-8") as f:
            txt = f.read()
            chars = sorted(set(txt))
            string_to_int = {ch:i for i,ch in enumerate(chars)}
            int_to_string = {i:ch for i,ch in enumerate(chars)}
            encode = lambda s: [string_to_int[c] for c in s]
            decode = lambda l: ''.join([int_to_string[i] for i in l])
            return txt, len(chars), encode, decode
    except FileNotFoundError:
        print("File not found. Bye")
        sys.exit(0)

def get_batch(data: TensorType) -> Tuple[TensorType, TensorType]:
    """ Chops data into batches blocks of a given size """
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y

def train_test_split(data: TensorType,
                     split: float,
                     device: str) -> Tuple[TensorType, TensorType]:
    """ 
        Splits data into training and testing portions
    """
    split_len = int(split * len(data))
    train_data = data[:split_len].to(device)
    test_data = data[split_len:].to(device)
    return train_data, test_data

@torch.no_grad()
def test_step(model: nn.Module,
              train_data: TensorType,
              eval_data: TensorType) -> dict:
    """ estimate the loss """
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(EVAL_ITER)
        for k in range(EVAL_ITER):
            x, y = get_batch(train_data if split == 'train' else eval_data)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_step(model: nn.Module,
               optimizer,
               data: TensorType) -> None:
    """ train the model """
    xb, yb = get_batch(data)
    _, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

def main():
    """ Main entry point for this script """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #text_file_name = input("Text File? ")
    text_file_name = "./data/woz.txt"
    text, vocab_size, encode, decode = tokenize_txt(text_file_name)
    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, test_data = train_test_split(data, SPLIT_SIZE, device)
    model = GPTLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    #### Training loop
    for epoch in range(MAX_ITERS):
        train_step(model, optimizer, train_data)
        losses = test_step(model, train_data, test_data)
        if epoch % EVAL_ITER == 0:
            losses = test_step(model, train_data, test_data)
            print(f"{epoch=} --- loss: {losses}")
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    generated_chars = decode(model.generate(context,
                            max_new_tokens=500)[0].tolist())
    print(generated_chars)

if __name__ == "__main__":
    main()
