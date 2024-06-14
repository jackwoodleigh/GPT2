from GPT import GPT2
from DataHelper import TokenDataset
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    batch_size = 8  # micro batch
    sequence_length = 1024

    t_dataset = TokenDataset(sequence_length)
    print(t_dataset.tokens[0])
    print(t_dataset.target_tokens[0])

    training_loader = DataLoader(t_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)

    model = GPT2()
    #model.training(training_loader, 100, total_batch_size=batch_size*1)

    model.sample()





