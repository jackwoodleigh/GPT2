import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset

class Loader:
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

class TokenDataset(Dataset):
    def __init__(self, sequence_length, enc, special_tokens, path="tiny-shakespeare.txt"):
        self.seq_len = sequence_length
        self.enc = enc
        self.special_token_set = set(special_tokens.keys())
        self.start_token = torch.tensor(list(special_tokens.values())[0])
        self.end_token = torch.tensor(list(special_tokens.values())[1])

        with open(path, "r") as f:
            text = f.read()
        tokens = enc.encode(text, allowed_special=self.special_token_set)

        tokens = torch.tensor(tokens)

        # splitting into groups of sequence length
        self.tokens, self.target_tokens = self.sequence_length_split(tokens)



    def sequence_length_split(self, tokens):
        seq_length = self.seq_len - 1  # remove 1 for insert of start and end token

        buff = tokens[:len(tokens) - (len(tokens) % seq_length)].view(-1, seq_length)

        '''x = buff[:, :-1].view(-1, seq_length - 1)
        y = buff[:, 1:].view(-1, seq_length - 1)'''

        x = torch.cat((self.start_token.repeat(buff.shape[0], 1), buff), dim=-1)
        y = torch.cat((buff, self.end_token.repeat(buff.shape[0], 1)), dim=-1)

        return x, y

    def __len__(self):
        return self.tokens.size(0)

    def __getitem__(self, idx):
        return self.tokens[idx], self.target_tokens[idx]


