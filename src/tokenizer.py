import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class BPETokenizer:
    def __init__(self, vocab_size=4000, model_path="src/bpe_tokenizer.json"):
        self.vocab_size = vocab_size
        self.model_path = model_path
        
        # Define our foundational special tokens for context routing
        self.special_tokens = ["[UNK]", "[PAD]", "<|user|>", "<|assistant|>", "<|end|>"]
        
        # If a trained tokenizer blueprint already exists on disk, load it instantly
        if os.path.exists(self.model_path):
            self.tokenizer = Tokenizer.from_file(self.model_path)
        else:
            # Initialize an empty BPE core shell that splits text at empty whitespace characters
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()

    def train_from_file(self, file_path: str):
        """
        Reads your raw text file, calculates repeating character patterns, 
        and compresses them into unified subword token IDs.
        """
        print(f"Training BPE Tokenizer on '{file_path}' (Vocab target: {self.vocab_size})...")
        
        trainer = BpeTrainer(
            special_tokens=self.special_tokens, 
            vocab_size=self.vocab_size
        )
        
        # Run the compilation algorithm through your dataset text
        self.tokenizer.train([file_path], trainer)
        
        # Save the structural mapping configuration file safely to your disk workspace
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.tokenizer.save(self.model_path)
        print(f"Tokenizer training complete! Mapping saved to '{self.model_path}'")

    def encode(self, text: str) -> list[int]:
        # Converts a string sentence down into an optimized list of subword compression IDs
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: list[int]) -> str:
        # Converts a list of integers smoothly back into readable human words
        return self.tokenizer.decode(tokens)