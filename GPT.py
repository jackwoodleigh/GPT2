from Transformer import Transformer
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
import time
import tiktoken

@dataclass
class GPTConfig:
    block_size = 1024  # seq length
    vocab_size = 50304
    n_layer = 12
    n_heads = 12
    n_embd = 768
    lr = 6e-4


class GPT2:
    def __init__(self, config=GPTConfig(), device="cuda"):
        self.seq_len = config.block_size
        self.device = device
        self.transformer = Transformer(config)
        self.transformer.to(device)
        self.optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=config.lr, betas=(0.9, 0.95), fused=True)
        self.text_to_token = tiktoken.get_encoding("gpt2")

    def training(self, training_loader, epoch, total_batch_size=524288):
        self.transformer.train()
        # model = torch.compile(model) doesnt work rn but could be cool in future
        B = training_loader.batch_size
        T = self.seq_len

        torch.set_float32_matmul_precision("high")  # doesn't seem to make a difference
        micro_batch_size = B * T
        batch_norm = 1.
        if total_batch_size is not None:
            batch_norm = total_batch_size // (B * T)

        for i in range(epoch):
            t1 = time.time()
            self.optimizer.zero_grad()
            loss_acc = 0

            for i, batch in enumerate(training_loader):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logit, loss = self.transformer(x, y)
                    # print(f"logit type: {logit.dtype}, loss type: {loss.dtype}")

                loss = loss / batch_norm  # correcting mean
                loss_acc += loss.detach()
                loss.backward()

                # grad accumulation
                # such that batch size = 0.5M tokens
                if total_batch_size is not None and i * micro_batch_size == total_batch_size:
                    # clip grad to 1.0 like gpt2
                    norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
                    self.optimizer.step()

            '''scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()'''

            torch.cuda.synchronize()
            print(f"Step: {i}, Loss: {loss_acc}, Time: {(time.time() - t1) * 1000:.2f}ms")

    def sample(self, seq_len=32, output_size=4, input_text=None):
        self.transformer.eval()
        text = ""
        if input_text is not None:
            text = input_text

        tokens = self.text_to_token.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(output_size, 1)   # making copies
        x = tokens.to(self.device)

        while x.size(1) < seq_len:
            with torch.no_grad():
                logit, loss = self.transformer(x)
                logit = logit[:, -1, :]
                prob = F.softmax(logit, dim=-1)

                topk_prob, topk_index = torch.topk(prob, 50, dim=-1)

                ix = torch.multinomial(topk_prob, 1)
                xcol = torch.gather(topk_index, -1, ix)
                x = torch.cat((x, xcol), dim=-1)

        for i in range(output_size):
            tokens = x[i, :seq_len].tolist()
            decode = self.text_to_token.decode(tokens)
            print(">", decode)