import torch
from tqdm import tqdm

# Explicitly importing from our custom package paths
from src.config import LoneWolfConfig
from src.tokenizer import ByteTokenizer
from src.dataset import TextDataset
from src.model import LoneWolfLLM

def main():
    # 1. Initialize Configuration
    config = LoneWolfConfig()
    print(f"Loading Lone Wolf Engine on: {config.device}")

    # 2. Initialize Tokenizer & Process Raw File Data
    tokenizer = ByteTokenizer()
    
    with open("data/training_data.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # Encode text into vector integers
    encoded_raw = tokenizer.encode(raw_text)
    data_tensor = torch.tensor(encoded_raw, dtype=torch.long)
    
    # 3. Initialize Dataset Loader (Splits data into train/validation)
    dataset = TextDataset(data_tensor, config.block_size)

    # 4. Initialize Neural Net Model
    model = LoneWolfLLM(config).to(config.device)
    
    # 5. Initialize AdamW Optimizer (The weight adjuster)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("Beginning Training Optimization Loop...")
    model.train()
    
    # Run a localized training loop of 500 steps
    for step in tqdm(range(500)):
        # Fetch a parallel batch of input vectors (X) and target vectors (Y)
        x_batch, y_batch = dataset.get_batch(batch_size=16, device=config.device)
        
        # Forward pass: compute predictions and current error (loss)
        logits, loss = model(x_batch, y_batch)
        
        # Backward pass: clean old gradients, calculate new ones, nudge weights
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f" | Step {step}: Training Loss = {loss.item():.4f}")

    # 6. Save the trained vector weights to disk (Like saving a database snapshot)
    torch.save(model.state_dict(), "lone_wolf_model.pt")
    print("Training complete! Model weights saved safely to 'lone_wolf_model.pt'.")

if __name__ == "__main__":
    main()
