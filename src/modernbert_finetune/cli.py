from __future__ import annotations

import argparse

from .config import EnvironmentConfig, ScriptConfig
from .pipeline import (
    enable_flash_attention_if_available,
    initialize_wandb,
    load_model,
    load_tokenizer,
    prepare_dataset,
    resolve_device,
    resolve_flash_attention,
    setup_logging,
    setup_repository,
    tokenize_dataset,
)
from .trainer import CurriculumTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ModernBERT with curriculum masking")
    parser.add_argument("--testing", action="store_true", help="Enable short-run testing mode")
    parser.add_argument(
        "--disable-flash-attention",
        action="store_true",
        help="Disable FlashAttention integration even if available",
    )
    parser.add_argument(
        "--push-interval",
        type=int,
        default=None,
        help="Override the push interval in training steps",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ScriptConfig:
    config = ScriptConfig(testing=args.testing)
    if args.disable_flash_attention:
        config.flash_attention = False
    if args.push_interval is not None:
        config.push_interval = args.push_interval
    return config


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    config = build_config(args)
    env = EnvironmentConfig()

    tokenizer, resize_required = load_tokenizer(config, env, logger)
    model = load_model(config, env, logger)
    if resize_required:
        model.resize_token_embeddings(len(tokenizer))

    flash_attention_cls = resolve_flash_attention(config.flash_attention, logger)
    enable_flash_attention_if_available(model, flash_attention_cls, logger)

    dataset = prepare_dataset(config, env, logger)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, logger)
    repo_artifacts = setup_repository(config, env, logger)
    device = resolve_device(logger)

    with initialize_wandb(config, env, logger) as wandb_run:
        trainer = CurriculumTrainer(
            model=model,
            tokenizer=tokenizer,
            tokenized_dataset=tokenized_dataset,
            config=config,
            device=device,
            repo_artifacts=repo_artifacts,
            logger=logger,
            wandb_run=wandb_run,
        )
        try:
            trainer.train()
            logger.info("Fine-tuning complete")
        except Exception as exc:
            logger.exception("Training failed: %s", exc)


__all__ = ["main", "parse_args", "build_config"]
