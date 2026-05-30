import torch

class TextDataset:
    def __init__(self, data_vector: torch.Tensor, block_size: int):
        self.data = data_vector
        self.block_size = block_size

    def get_batch(self, batch_size: int, device: str):
        # Grab random starting positions inside the dataset
        high_index = len(self.data) - self.block_size
        ix = torch.randint(0, high_index, (batch_size,))
        
        # Stack sequential input contexts (X) and target outputs (Y)
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
        
        return x.to(device), y.to(device)
