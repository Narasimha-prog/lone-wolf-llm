import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Explicitly importing from our custom package paths
from src.config import LoneWolfConfig
from src.tokenizer import BPETokenizer
from src.dataset import TextDataset
from src.model import LoneWolfLLM

def main():
    #  Initialize Configuration
    config = LoneWolfConfig()

    print(f"Loading Lone Wolf Engine on: {config.device}")

    # Initialize Tokenizer & Process Raw File Data
    tokenizer = BPETokenizer(config.vocab_size)
    

    # taking data to train the model from file 'hello'
    with open("data/training_data.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    

    #  Encode text into vector integers [104, 101, 108, 108, 111...]
    encoded_raw = tokenizer.encode(raw_text)

    #  tensor([104, 101, 108, 108, 111...])
    data_tensor = torch.tensor(encoded_raw, dtype=torch.long)

    
    # Initialize Dataset Loader (Splits data into train/validation)
    dataset = TextDataset(data_tensor, config.block_size)

    # Initialize Neural Net Model
    model = LoneWolfLLM(config).to(config.device)
    
    # Initialize AdamW Optimizer (The weight adjuster)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

   # Instantiate Event Logger (creates 'runs/' dir automatically)
    writer = SummaryWriter(log_dir="runs/lone_wolf_transformer")

    print("Beginning Training Optimization Loop...")

    model.train()
    
    # Run a localized training loop of 2000 steps
    for step in tqdm(range(2000)):
        # Fetch a parallel batch of input vectors (X) and target vectors (Y)
        x_batch, y_batch = dataset.get_batch(batch_size=16, device=config.device)
        
        # Forward pass: compute predictions and current error (loss)
        logits, loss = model(x_batch, y_batch)
        
        # Backward pass: clean old gradients, calculate new ones, nudge weights
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        #  Stream data coordinates directly into TensorBoard database
        writer.add_scalar("Loss/train", loss.item(), step)
        
        if step % 100 == 0:
            print(f" | Step {step}: Training Loss = {loss.item():.4f}")



    # Flush and release binary write handle lock
    writer.close()

    #  Save the trained vector weights to disk (Like saving a database snapshot)
    torch.save(model.state_dict(), "lone_wolf_model.pt")

    print("Training complete! Model weights saved safely to 'lone_wolf_model.pt'.")

if __name__ == "__main__":
    main()
