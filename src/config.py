import torch

class LoneWolfConfig:
    vocab_size = 256       # how many token types exist using ascii id 104
    block_size = 256       # Max context length (tokens looked at simultaneously)
    n_embd = 128           # Dimensionality of the vectors
    n_head = 4             # Parallel attention heads
    n_layer = 4            # Number of stacked Transformer blocks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

