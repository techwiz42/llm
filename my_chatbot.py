import sys
import traceback
import pickle
import torch
from torch import nn
import gpt_model
import constants

BLOCK_SIZE = constants.BLOCK_SIZE
data_path = constants.data_path
model_path = constants.model_path

# pylint: disable-msg=unnecessary-comprehension
# pylint: disable-msg=unnecessary-lambda-assignment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_vocab_size():
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            txt = f.read()
            chars = sorted(list(set(txt)))
            string_to_int = {ch:i for i,ch in enumerate(chars)}
            int_to_string = {i:ch for i,ch in enumerate(chars)}
            encode = lambda s: [string_to_int[c] for c in s]
            decode = lambda l: ''.join([int_to_string[i] for i in l])
            return txt, len(chars), encode, decode
    except FileNotFoundError:
        print("File not found. Bye")
        sys.exit(0)

def main():
    """ Main entry point for this script """
    txt, vocab_size, encode, decode  = get_vocab_size()
    model = gpt_model.GPTLanguageModel(vocab_size) 
    try:
        print("loading model")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        print("success!")
    except:
        print("Could not load pretrained model")
        sys.exit(0)

    while True:
        prompt = input("Ask me anything: ")
        if prompt == "":
            print("bye!")
            sys.exit(0)
        context = torch.tensor(encode(prompt),
                               dtype = torch.long,
                               device = device).unsqueeze(0)
        generated_chars = decode(model.generate(context,
                                                max_new_tokens = BLOCK_SIZE)[0].tolist())
        print(f"\n{generated_chars}")
    

if __name__ == "__main__":
    main()
