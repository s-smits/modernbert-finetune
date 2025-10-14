#!/usr/bin/env python3
"""
Unified tokenizer training script for ModernBERT fine-tuning.

Supports both SentencePiece (unigram/bpe) and WordPiece tokenizers.
Automatically handles datasets with 'text' or 'content' fields.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from itertools import islice
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

DEFAULT_DATASET = "ssmits/fineweb-2-dutch"
DEFAULT_SAVE_PATH = "domain_tokenizer"
DEFAULT_VOCAB_SIZE = 32_768
DEFAULT_NUM_EXAMPLES = 1_000_000
DEFAULT_BATCH_SIZE = 1_000
HUGGINGFACE_TOKEN_ENV = "HUGGINGFACE_TOKEN"
SPECIAL_TOKENS = {
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
}


def extract_text(example: dict) -> str:
    if (text := example.get("text")):
        return text
    if (content := example.get("content")):
        return content
    raise KeyError("Expected 'text' or 'content' field in dataset example")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif any(path.iterdir()):
            raise FileExistsError(f"Directory '{path}' already exists; use --overwrite to replace it")
    path.mkdir(parents=True, exist_ok=True)


def train_sentencepiece_tokenizer(
    dataset_iterator: Iterator[dict],
    vocab_size: int,
    save_dir: Path,
    model_type: str = "unigram",
    num_examples: int = DEFAULT_NUM_EXAMPLES,
) -> None:
    from sentencepiece import SentencePieceTrainer

    print(f"Training SentencePiece tokenizer (type: {model_type})...")

    with tempfile.TemporaryDirectory(prefix="modernbert-spm-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        corpus_path = tmp_path / "corpus.txt"

        print(f"Writing {num_examples:,} examples to temporary file...")
        with corpus_path.open("w", encoding="utf-8") as handle:
            for idx, item in enumerate(dataset_iterator, start=1):
                if idx > num_examples:
                    break
                text = extract_text(item).replace("\n", " ")
                handle.write(text + "\n")
                if idx % 10_000 == 0:
                    print(f"  Processed {idx:,} examples...")

        print("Training SentencePiece model...")
        model_prefix = tmp_path / "spm"
        spm_args = [
            f"--input={corpus_path}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size}",
            f"--model_type={model_type}",
            "--split_digits=true",
            "--normalization_rule_name=nmt_nfkc_cf",
            "--add_dummy_prefix=false",
            "--remove_extra_whitespaces=true",
            "--pad_id=3",
            "--unk_id=0",
            "--bos_id=1",
            "--eos_id=2",
        ]
        SentencePieceTrainer.train(" ".join(spm_args))

        shutil.move(str(model_prefix.with_suffix(".model")), save_dir / "spm.model")
        shutil.move(str(model_prefix.with_suffix(".vocab")), save_dir / "spm.vocab")

    convert_sentencepiece_to_hf(save_dir, model_type)
    print(f"✓ SentencePiece tokenizer saved to {save_dir}")


def convert_sentencepiece_to_hf(save_dir: Path, model_type: str) -> None:
    spm_model_path = save_dir / "spm.model"
    try:
        from transformers import PreTrainedTokenizerFast

        tokenizer_obj = None
        if model_type == "unigram":
            from tokenizers import SentencePieceUnigramTokenizer

            tokenizer_obj = SentencePieceUnigramTokenizer(str(spm_model_path))
        else:
            raise ValueError("Fast tokenizer conversion is only supported for unigram SentencePiece models")

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            **SPECIAL_TOKENS,
        )
        hf_tokenizer.save_pretrained(save_dir)
        print("✓ Tokenizer converted to HuggingFace format")
    except ImportError as exc:
        print(f"⚠ Warning: Additional dependencies required for HuggingFace conversion: {exc}")
        print("  Install 'transformers' and 'tokenizers' to enable automatic conversion")
    except Exception as exc:
        print(f"⚠ Warning: Could not convert to HuggingFace format: {exc}")
        print("  The raw SentencePiece files are still available and usable")


def train_wordpiece_tokenizer(
    dataset_iterator: Iterator[dict],
    vocab_size: int,
    save_dir: Path,
    num_examples: int = DEFAULT_NUM_EXAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordPieceTrainer
    from transformers import PreTrainedTokenizerFast

    print("Training WordPiece tokenizer...")

    tokenizer = Tokenizer(WordPiece(unk_token=SPECIAL_TOKENS["unk_token"]))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS.values()),
        min_frequency=2,
    )

    def batch_iterator() -> Iterator[list[str]]:
        for start in range(0, num_examples, batch_size):
            chunk = list(islice(dataset_iterator, batch_size))
            if not chunk:
                break
            yield [extract_text(item) for item in chunk]
            processed = min(start + batch_size, num_examples)
            if processed % 10_000 == 0:
                print(f"  Processed {processed:,} examples...")

    print(f"Training on {num_examples:,} examples...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=num_examples)

    with tempfile.TemporaryDirectory(prefix="modernbert-wp-") as tmp_dir:
        tmp_json = Path(tmp_dir) / "tokenizer.json"
        tokenizer.save(str(tmp_json))
        shutil.copy2(tmp_json, save_dir / "tokenizer.json")

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **SPECIAL_TOKENS)
    hf_tokenizer.save_pretrained(save_dir)
    print(f"✓ WordPiece tokenizer saved to {save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom tokenizer for ModernBERT fine-tuning")
    parser.add_argument(
        "--type",
        choices=["sentencepiece", "wordpiece", "spm", "wp"],
        default="sentencepiece",
        help="Tokenizer type: 'sentencepiece' (or 'spm') or 'wordpiece' (or 'wp')",
    )
    parser.add_argument(
        "--model-type",
        choices=["unigram", "bpe"],
        default="unigram",
        help="For SentencePiece: 'unigram' or 'bpe' (default: unigram)",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--save-path",
        default=DEFAULT_SAVE_PATH,
        help=f"Directory to save tokenizer (default: {DEFAULT_SAVE_PATH})",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help=f"Number of examples to train on (default: {DEFAULT_NUM_EXAMPLES:,})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for WordPiece training (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get(HUGGINGFACE_TOKEN_ENV),
        help="HuggingFace access token (defaults to HUGGINGFACE_TOKEN env var if set)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tokenizer directory if it already exists",
    )

    args = parser.parse_args()

    tokenizer_type = args.type.lower()
    if tokenizer_type in {"spm", "sentencepiece"}:
        tokenizer_type = "sentencepiece"
    elif tokenizer_type in {"wp", "wordpiece"}:
        tokenizer_type = "wordpiece"

    save_dir = Path(args.save_path)

    print("=" * 70)
    print("Tokenizer Training Configuration")
    print("=" * 70)
    print(f"Type:          {tokenizer_type}")
    if tokenizer_type == "sentencepiece":
        print(f"Model Type:    {args.model_type}")
    print(f"Dataset:       {args.dataset}")
    print(f"Vocab Size:    {args.vocab_size:,}")
    print(f"Num Examples:  {args.num_examples:,}")
    print(f"Save Path:     {save_dir}")
    if args.hf_token:
        print("Auth Token:    Provided")
    print("=" * 70)
    print()

    try:
        prepare_output_dir(save_dir, args.overwrite)
    except FileExistsError as exc:
        print(f"✗ {exc}")
        return

    print("Loading dataset in streaming mode...")
    dataset = load_dataset(
        args.dataset,
        split="train",
        streaming=True,
        use_auth_token=args.hf_token,
    )
    dataset_iterator = iter(dataset)

    if tokenizer_type == "sentencepiece":
        train_sentencepiece_tokenizer(
            dataset_iterator,
            vocab_size=args.vocab_size,
            save_dir=save_dir,
            model_type=args.model_type,
            num_examples=args.num_examples,
        )
    else:
        train_wordpiece_tokenizer(
            dataset_iterator,
            vocab_size=args.vocab_size,
            save_dir=save_dir,
            num_examples=args.num_examples,
            batch_size=args.batch_size,
        )

    print()
    print("=" * 70)
    print("✓ Tokenizer training complete!")
    print(f"✓ Saved to: {save_dir}")
    print()
    print("To use this tokenizer with training:")
    print(f"  python train.py  # (auto-detects {save_dir})")
    print("=" * 70)


if __name__ == "__main__":
    main()
