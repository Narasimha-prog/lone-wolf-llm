import torch
from src.config import LoneWolfConfig
from src.tokenizer import ByteTokenizer
from src.model import LoneWolfLLM

def generate_text(model, tokenizer, config, prompt_text, max_new_tokens=100):
    model.eval()
    # Turn input string into starting tokens
    tokens = tokenizer.encode(prompt_text)
    idx = torch.tensor([tokens], dtype=torch.long, device=config.device)
    
    # Generation loop
    for _ in range(max_new_tokens):
        # Crop context window if it exceeds model specifications
        idx_cond = idx[:, -config.block_size:]
        
        # Get raw predictions
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        # Focus on the last token's distribution probabilities
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        # Sample the next token
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequential stream
        idx = torch.cat((idx, idx_next), dim=1)
        
    return tokenizer.decode(idx[0].tolist())

def main():
    config = LoneWolfConfig()
    tokenizer = ByteTokenizer()
    
    # Reconstruct architecture shell and load saved states
    model = LoneWolfLLM(config).to(config.device)
    model.load_state_dict(torch.load("lone_wolf_model.pt", map_location=config.device))
    
    prompt = input("Prompt Input: ")
    print()
  

    output = generate_text(model, tokenizer, config, prompt, max_new_tokens=60)
    print("--- Generated Output ---")
    print(output)

if __name__ == "__main__":
    main()
