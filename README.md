# modernbert-finetune

Fine-tune ModernBERT with custom tokenizers, curriculum learning, and next-gen optimizers.

## Why

- **Custom Tokenization**: Train domain-specific tokenizers on streaming datasets of any size.
- **Curriculum Learning**: Accelerate convergence with progressive masking decay.
- **Next-Gen Optimization**: Native support for **Muon** and **ADOPT**.
- **Efficiency**: FlashAttention 2, dynamic batching, and gradient accumulation.

## Quick Start

Prerequisites: Python 3.10+, CUDA ≥7.0 for FlashAttention.

```bash
# 1. Install dependencies
uv sync

# 2. Auth
export HUGGINGFACE_TOKEN="your_token"

# 3. Train Tokenizer (Optional)
uv run train_tokenizer.py --vocab-size 32768

# 4. Fine-tune
uv run train.py
```

## Configuration

**Tokenizer** (`train_tokenizer.py`):
- `VOCAB_SIZE`: 32768
- `DATASET_NAME`: "ssmits/fineweb-2-dutch"

**Training** (`train.py`):
- `per_device_train_batch_size`: 4
- `gradient_accumulation_steps`: 2
- `learning_rate`: 5e-4
- `masking_probabilities`: [0.3, 0.2, 0.18, 0.16, 0.14]

## Architecture

```
modernbert-finetune/
├── train_tokenizer.py       # Custom tokenizer training
├── train.py                 # Main training loop
└── domain_tokenizer/        # Output directory
```

**Pipeline**: Stream dataset → Load/Train Tokenizer → Resize Embeddings → Curriculum Train → Push to Hub.

## Troubleshooting

- **OOM**: Reduce batch size or increase gradient accumulation.
- **FlashAttention**: Requires GPU cap ≥7.0. Disable with `--disable-flash-attention`.
