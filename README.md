# modernbert-finetune

Fine-tune ModernBERT with custom tokenizers, curriculum learning, and ADOPT optimization. Single-GPU training with FlashAttention 2 support.

## Why modernbert-finetune

- Trains WordPiece tokenizers on streaming datasets without loading full corpus into memory.
- Implements curriculum learning through progressive MLM masking decay.
- Uses ADOPT optimizer for improved convergence.
- Handles dynamic batching and gradient accumulation for memory-constrained environments.

## Quick start

Prerequisites: Python 3.10+, GPU with compute capability ≥7.0 for FlashAttention 2

```bash
git clone https://github.com/s-smits/modernbert-finetune.git
cd modernbert-finetune
pip install -r requirements.txt

export HUGGINGFACE_TOKEN="your_token"
huggingface-cli login --token $HUGGINGFACE_TOKEN

# Optional: train custom tokenizer
python tokenize.py

# Fine-tune model
python train.py
```

Load fine-tuned model:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("username/modernbert-dutch")
tokenizer = AutoTokenizer.from_pretrained("username/modernbert-dutch")
```

## Configuration

**Tokenizer** (`tokenize.py`): `VOCAB_SIZE=32768`, `NUM_EXAMPLES_TO_TRAIN=10000`, `DATASET_NAME="ssmits/fineweb-2-dutch"`

**Training** (`train.py`): `per_device_train_batch_size=4`, `gradient_accumulation_steps=2`, `learning_rate=5e-4`, `masking_probabilities=[0.3, 0.2, 0.18, 0.16, 0.14]`, `tokenizer_path="domain_tokenizer"`

Full parameter reference in source files.

## Architecture at a glance

```
modernbert-finetune/
├── tokenize.py              # WordPiece tokenizer training
├── train.py                 # MLM fine-tuning with curriculum learning
└── domain_tokenizer/        # custom tokenizer output
```

Pipeline: Stream dataset → Load tokenizer → Resize embeddings → Apply curriculum masking → Push to Hub

## Key features

**Curriculum learning**: Progressive masking decay from 0.3 to 0.14 over training for improved convergence.

**ADOPT optimizer**: Replaces AdamW for better convergence properties.

**FlashAttention 2**: Automatic 2-3x speedup on compatible GPUs.

**Streaming datasets**: Processes arbitrarily large corpora without memory constraints.

## Troubleshooting

**CUDA OOM**: Reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps`.

**FlashAttention errors**: Set `USE_FLASH_ATTENTION=False` or verify GPU compute capability ≥7.0.

**Tokenizer loading**: Ensure `tokenizer.json` exists in `tokenizer_path`, otherwise falls back to base model tokenizer.

## License

MIT
