from __future__ import annotations

import argparse
import logging
import sys

from src.modernbert_finetune.config import EnvironmentConfig, ScriptConfig
from src.modernbert_finetune.pipeline import setup_logging
from src.modernbert_finetune.tokenizer_trainer import train_tokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom tokenizer for ModernBERT")
    parser.add_argument("--testing", action="store_true", help="Enable short-run testing mode")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Vocabulary size")
    parser.add_argument("--limit", type=int, default=10_000, help="Number of examples to train on (if testing)")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logging()
    
    config = ScriptConfig(testing=args.testing)
    env = EnvironmentConfig()
    
    # Update config based on mode
    if args.testing:
        # Override config or pass custom args to train_tokenizer?
        # For now, let's keep it simple.
        pass

    try:
        train_tokenizer(
            config, 
            env, 
            logger, 
            vocab_size=args.vocab_size, 
            limit=args.limit if not args.testing else 1000
        )
    except Exception as exc:
        logger.exception("Tokenizer training failed: %s", exc)
        sys.exit(1)

if __name__ == "__main__":
    main()
