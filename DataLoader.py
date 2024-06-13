import torch
import tiktoken


class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        enc = tiktoken.get_encoding("gpt2")
        with open("tiny-shakespeare.txt", "r") as f:
            text = f.read()

        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.state_pos = 0

    def next_batch(self):
        buff = self.tokens[self.state_pos : (self.state_pos + self.B * self.T + 1)]

        x = buff[:-1].view(self.B, self.T)
        y = buff[1:].view(self.B, self.T)

        self.state_pos += self.B * self.T

        # reset
        if self.state_pos + (self.B * self.T + 1) > len(self.tokens):
            self.state_pos = 0

        return x, y
