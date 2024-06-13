import functools
import time

import torch

from GPT import GPT2
from DataLoader import DataLoader
import tiktoken
from torch.amp import GradScaler, autocast


num_return_seq = 5
max_length = 30
total_batch_size = 524288
B = 8  # micro batch
T = 1024
max_steps = 50

grad_accumulation_steps = total_batch_size // (B * T)

training_loader = DataLoader(B, T)

device = "cuda"

model = GPT2()

model.sample(input_text="Good morning, ")





