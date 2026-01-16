from __future__ import annotations

import logging
import os
from typing import Iterator

from datasets import load_dataset
from transformers import AutoTokenizer

from .config import EnvironmentConfig, ScriptConfig


def get_training_corpus(dataset_name: str, split: str = "train", batch_size: int = 1000) -> Iterator[str]:
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for i in range(0, len(dataset), batch_size): # streaming datasets don't have len() usually, so this might fail.
        # Wait, for streaming datasets we just iterate.
        pass
    
    # Correct approach for streaming:
    batch = []
    for i, example in enumerate(dataset):
        text = example.get("text", example.get("content", ""))
        if text:
            batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def get_training_corpus_streaming(dataset_name: str, split: str = "train", limit: int = 100_000, batch_size: int = 1000) -> Iterator[list[str]]:
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    batch = []
    count = 0
    for example in dataset:
        text = example.get("text", example.get("content", ""))
        if text:
            batch.append(text)
            count += 1
        
        if len(batch) == batch_size:
            yield batch
            batch = []
        
        if limit and count >= limit:
            break
            
    if batch:
        yield batch

def train_tokenizer(config: ScriptConfig, env: EnvironmentConfig, logger: logging.Logger, vocab_size: int = 32768, limit: int = 5_000_000) -> None:
    logger.info(f"Loading base tokenizer from {config.model_checkpoint}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        config.model_checkpoint,
        use_auth_token=env.huggingface_token
    )

    logger.info(f"Preparing corpus from {config.dataset_name} (limit={limit} rows)")
    
    corpus = get_training_corpus_streaming(
        config.dataset_name, 
        limit=limit
    )

    logger.info("Starting tokenizer training... this may take a while")
    new_tokenizer = base_tokenizer.train_new_from_iterator(
        corpus,
        vocab_size=vocab_size, 
    )

    save_path = config.tokenizer_path
    if not save_path:
        save_path = "domain_tokenizer"
        
    logger.info(f"Saving tokenizer to {save_path}")
    new_tokenizer.save_pretrained(save_path)
    logger.info("Tokenizer training complete")
