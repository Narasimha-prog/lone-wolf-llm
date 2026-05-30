# lone_wolf_llm
```
lone_wolf_llm/
│
├── lone_wolf_env/           # Isolated Python libraries (Your virtual env)
│
├── data/                    # The "Database" layer
│   └── training_data.txt    # Raw text dataset the model reads to learn
│
├── src/                     # The core engine source code
│   ├── __init__.py          # Marks this directory as a Python package
│   ├── config.py            # Hyperparameters (Vector sizes, layers, device)
│   ├── tokenizer.py         # Text-to-Vector encoder/decoder (BPE or Byte-level)
│   ├── model.py             # Neural net layers (RMSNorm, Attention, SwiGLU)
│   └── dataset.py           # PyTorch Data Loader (Feeds batches to the model)
│
├── train.py                 # The optimization runner (Backpropagation loop)
└── generate.py              # The execution interface (Input prompt -> Output text)
```
