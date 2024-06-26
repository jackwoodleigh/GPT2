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
    lr = 0.0006 #1e-5


class GPT2:
    def __init__(self,  enc, special_tokens, config=GPTConfig(), device="cuda"):
        self.enc = enc
        self.special_token_set = set(special_tokens.keys())
        self.seq_len = config.block_size
        self.device = device
        self.transformer = Transformer(config)
        self.transformer.to(device)
        self.optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=config.lr, betas=(0.9, 0.95), fused=True)
        self.text_to_token = tiktoken.get_encoding("gpt2")
        self.start_token = list(special_tokens.keys())[0]

    def training(self, training_loader, epoch, save_path=None, total_batch_size=524288):
        self.transformer.train()
        torch.set_float32_matmul_precision("high")  # doesn't seem to make a difference
        # model = torch.compile(model) doesnt work rn but could be cool in future

        B = training_loader.batch_size
        T = self.seq_len
        total_micro_steps = total_batch_size // B

        batch_norm = 1.
        if total_batch_size is not None:
            batch_norm = total_batch_size // B

        self.optimizer.zero_grad()
        current_step = 0
        loss_acc = 0
        t1 = time.time()
        for e in range(epoch):
            for i, batch in enumerate(training_loader):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logit, loss = self.transformer(x, y)
                    # print(f"logit type: {logit.dtype}, loss type: {loss.dtype}")

                loss = loss / batch_norm  # correcting mean
                loss_acc += loss.detach()
                loss.backward()
                current_step += 1
                #print(f"Step {i}, Time: {(time.time() - t3) * 1000:.2f}ms")
                # grad accumulation
                # such that batch size = 0.5M tokens
                if total_batch_size is not None and current_step == total_micro_steps:
                    # clip grad to 1.0 like gpt2
                    torch.cuda.synchronize()
                    norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    print(f"Epoch: {e}, Loss: {loss_acc / B}, Time: {(time.time() - t1) * 1000:.2f}ms")
                    loss_acc = 0
                    current_step = 0

                    t1 = time.time()


            '''scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()'''
            if save_path is not None:
                self.save_model(save_path)

    def sample(self, seq_len=1024, output_size=1, input_text=None):
        self.transformer.eval()
        text = self.start_token
        if input_text is not None:
            text = text + input_text

        tokens = self.enc.encode(text, allowed_special=self.special_token_set)
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
            tokens = x[i, :seq_len].tolist()[1:]
            decoded = self.enc.decode(tokens)
            with open(f"samples/sample_{i}.txt", 'w') as file:
                file.write(''.join(decoded))

            '''# printing out effect
            for token in decode:
                print(token, end="")
                time.sleep(0.01)'''


    def save_model(self, path):
        torch.save(self.transformer.state_dict(), path)
        print("Model Saved.")

    def load_model(self, path):
        self.transformer.load_state_dict(torch.load(path))
        print("Model Loaded.")

    def print_parameter_count(self):
        print(sum(p.numel() for p in self.transformer.parameters()))