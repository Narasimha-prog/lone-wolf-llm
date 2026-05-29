import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# 1. Define our training dataset (What the Lone Wolf will read to learn)
training_text = """
The lone wolf travels deep into the digital forest. 
Vectors are coordinates in a multi-dimensional universe.
An LLM predicts the next token by calculating probabilities using matrix multiplication.
"""

# 2. Build the Vocabulary (Unique characters our model knows)
chars = sorted(list(set(training_text)))
vocab_size = len(chars)
print(f"Vocabulary Size: {vocab_size} unique characters.")
print(f"Our Alphabet: {''.join(chars)}\n")

# 3. Create the Mapping (The Tokenizer)
# stoi: string-to-integer | itos: integer-to-string
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]          # Takes a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # Takes a list of integers, outputs a string

# Test the Tokenizer
sample_phrase = "lone wolf"
encoded_sample = encode(sample_phrase)
print(f"Text:   '{sample_phrase}'")
print(f"Vector Indices: {encoded_sample}")
print(f"Decoded Back:   '{decode(encoded_sample)}'\n")

# 4. Convert our entire training text into a PyTorch Tensor (Vector Array)
data = torch.tensor(encode(training_text), dtype=torch.long)
print(f"Total training tokens as a tensor shape: {data.shape}")