from GPT import GPT2
from DataHelper import TokenDataset
from torch.utils.data import DataLoader
import tiktoken

if __name__ == '__main__':
    batch_size = 8  # micro batch
    sequence_length = 1024
    start_token = "<S>"
    end_token = "<E>"
    special_tokens = {
        start_token: 50258,
        end_token: 50259,
    }
    gpt2_base = tiktoken.get_encoding("gpt2")
    enc = tiktoken.Encoding(
        name="gpt2_special",
        pat_str=gpt2_base._pat_str,
        mergeable_ranks=gpt2_base._mergeable_ranks,
        special_tokens=special_tokens
    )
    '''s = set(special_tokens.keys())
    k = list(special_tokens.keys())[0]
    start_token = enc.encode(k, allowed_special=s)'''

    t_dataset = TokenDataset(sequence_length, enc=enc, special_tokens=special_tokens)
    print(t_dataset.tokens[0])
    print(t_dataset.target_tokens[0])

    training_loader = DataLoader(t_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)

    model = GPT2(enc=enc, special_tokens=special_tokens)
    model.load_model(path="tiny_shakespeare_save.pt")
    #model.training(training_loader, 100, total_batch_size=batch_size*8, save_path="save.pt")

    model.sample(output_size=10)
