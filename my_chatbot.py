import sys
import traceback
import pickle
import torch
from torch import nn
import gpt_model
import constants

BLOCK_SIZE = constants.BLOCK_SIZE

# pylint: disable-msg=unnecessary-comprehension
# pylint: disable-msg=unnecessary-lambda-assignment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_vocab_size() -> int:
    try:
        with open("data/vocab.txt", "r", encoding="utf-8") as f:
            txt = f.read()
            chars = sorted(list(set(txt)))
            return len(chars)
    except FileNotFoundError:
        print("File not found. Bye")
        sys.exit(0)

def encode(plain_txt: str) -> str:
    string_to_int = {ch:i for i,ch in enumerate(chars)}
    return lambda s: [string_to_int[c] for c in plain_txt]

def decode(encoded_txt: str) -> str:
    int_to_string = {i:ch for i,ch in enumerate(chars)}
    return lambda l: ''.join([int_to_string[i] for i in encoded_txt])


def main():
    """ Main entry point for this script """
    vocab_size = get_vocab_size()
    model = gpt_model.GPTLanguageModel(vocab_size) # FIXME: Is this necessary?
    try:
        with open('models/model-01.pkl', 'rb') as mf:
            model = pickle.load(mf)
            model.to(device)
    except:
        traceback.print_exc()
        print("Could not load pretrained model")
        sys.exit(0)

    while True:
        prompt = input("Ask me anything: ")
        context = torch.tensor(encode(prompt),
                               dtype = torch.long,
                               device = device).unsqueeze(0)
        generated_chars = decode(model.generate(context,
                                                max_new_tokens = BLOCK_SIZE)[0].tolist())
        print(f"\n{generated_chars}")
    

if __name__ == "__main__":
    main()
