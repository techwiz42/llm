""" Baby's First Large Language Model """
import sys
from typing import Tuple, Callable
from torchtyping import TensorType
import torch
from torch import nn
from torch.nn import functional as F

BLOCK_SIZE = 4
BATCH_SIZE = 8
MAX_ITERS = 1000
EMBEDDING_DIM = 100
SPLIT_SIZE = 0.8
LEARNING_RATE = .1
EVAL_ITER = 100

class BigramLanguageModel(nn.Module):
    """ 
        Implements a bigram language model. Predicts the next character in a text.
        Very naieve implementation. First try.
    """

    def __init__(self, vocab_size: int):
        """ Initialize method. vocab_size is the number of distinct characters in the text """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index: int, targets: TensorType = None):
        """ mandatory implementation of the forward method """
        logits = self.token_embedding_table(index)
        if targets is None:
            loss = None
        else:
            bdim, tdim, cdim = logits.shape
            logits = logits.view(bdim*tdim, cdim)
            targets = targets.view(bdim*tdim)
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

def get_batch(data: TensorType, 
              block_size = BLOCK_SIZE,
              batch_size = BATCH_SIZE) -> Tuple[TensorType, TensorType]:
    """ Chops data into batches blocks of a given size """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    #### Training loop
    for epoch in range(MAX_ITERS):
        train_step(model, optimizer, train_data)
        if epoch % EVAL_ITER == 0:
            losses = test_step(model, train_data, test_data)
            print(f"{epoch=} --- loss: {losses}")
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    generated_chars = decode(model.generate(context,
                            max_new_tokens=1000)[0].tolist())
    print(generated_chars)

if __name__ == "__main__":
    main()
