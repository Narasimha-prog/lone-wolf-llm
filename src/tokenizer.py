class ByteTokenizer:
    def __init__(self):
        # Using a standard 256-byte vocabulary for character-to-byte conversion
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        # Converts a string directly into its UTF-8 numerical byte values
        return list(text.encode('utf-8', errors='ignore'))

    def decode(self, tokens: list[int]) -> str:
        # Converts a list of byte integers back into a human-readable string
        return bytes(tokens).decode('utf-8', errors='ignore')
