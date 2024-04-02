""" 
    Baby's First Large Language Model 
    From: https://www.youtube.com/watch?v=UU1WVnMk4E8
    Note that tutorial became increasingly incoherent toward the end
         I reproduce the code here in the hope of understanding it by 
         trying to make it work.
"""
import sys
from typing import Tuple, Callable
from torchtyping import TensorType
import torch
from torch import nn
from torch.nn import functional as F
import constants
from  gpt_model import GPTLanguageModel

BLOCK_SIZE = constants.BLOCK_SIZE
BATCH_SIZE = constants.BATCH_SIZE
MAX_ITERS = constants.MAX_ITERS
EMBEDDING_DIM = constants.EMBEDDING_DIM
SPLIT_SIZE = constants.SPLIT_SIZE
LEARNING_RATE = constants.LEARNING_RATE
EVAL_ITER = constants.EVAL_ITER
n_embed = constants.n_embed
n_head = constants.n_head
N_LAYER = constants.N_LAYER
dropout = constants.dropout
data_path = constants.data_path
model_path = constants.model_path

device = 'cuda' if torch.cuda.is_available()  else 'cpu'

def tokenize_txt(text_file_name: str) -> Tuple[str, int, Callable, Callable]:
    # pylint: disable-msg=unnecessary-comprehension
    # pylint: disable-msg=unnecessary-lambda-assignment
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
    #text_file_name = input("Text File? ")
    text_file_name = data_path
    text, vocab_size, encode, decode = tokenize_txt(text_file_name)
    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, test_data = train_test_split(data, SPLIT_SIZE, device)

    model = GPTLanguageModel(vocab_size)
    try:
       print("Loading model")
       model.load_state_dict(torch.load(model_path))
       print("Success!")
    except:
        print("Model failed to load. Initializing new model")
        model = GPTLanguageModel(vocab_size)
    model.to(device)

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
    print("\nSaving model")
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()
