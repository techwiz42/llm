import unittest
import random
import torch
from torch import nn
import my_gpt as gpt

class TestMyGPT(unittest.TestCase):

    def test_instantiate_model(self):
        model = gpt.GPTLanguageModel(vocab_size=10)
        self.assertTrue(isinstance(model, gpt.GPTLanguageModel))

    def test_cuda_available(self):
        self.assertTrue(torch.cuda.is_available())

    def test_tokenizer(self):
        _, vocab_len, encode, decode = gpt.tokenize_txt('./data/woz.txt')
        self.assertTrue(vocab_len == 69)
        hello_world_encoded = encode("hello world")
        self.assertTrue("hello world" == decode(hello_world_encoded))

    def test_split(self):
        data = torch.rand(100).to('cuda')
        train, test = gpt.train_test_split(data, 0.8, 'cuda')
        self.assertTrue(len(train) == 80)
        self.assertTrue(len(test) == 20)

    def test_batch(self):
        data = torch.rand(1000)
        train, test = gpt.get_batch(data, 4, 8)
        self.assertTrue(train.shape == torch.Size([8, 4]))
        self.assertTrue(test.shape == torch.Size([8,4]))

    def test_model(self):
        split = random.uniform(0,1)
        text, vocab_len, encode, decode = gpt.tokenize_txt('./data/woz.txt')
        data = torch.tensor(encode(text), dtype = torch.long)
        train_data, test_data = gpt.train_test_split(data,
                                                     split,
                                                     "cuda")
        actual = round(len(train_data)/(len(train_data) + len(test_data)),3)
        expected = round(split, 3)
        self.assertTrue(actual == expected)
        # Test instantiate model
        model = gpt.GPTLanguageModel(vocab_len).to('cuda')
        self.assertTrue(isinstance(model, gpt.GPTLanguageModel))
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.1)
        # Test training step - just call step w/o failure
        gpt.train_step(model, optimizer, train_data)
        # Test eval step
        losses = gpt.test_step(model, train_data, test_data)
        self.assertTrue(isinstance(losses['train'], torch.Tensor))
        self.assertTrue(isinstance(losses['eval'], torch.Tensor))
        context = torch.zeros((1,1), dtype=torch.long, device="cuda")
        # Test generate step
        generated_chars = decode(model.generate(context,
                                                max_new_tokens=10)[0].tolist())
        self.assertTrue(isinstance(generated_chars, str))

if __name__ == "__main__":
    unittest.main()
