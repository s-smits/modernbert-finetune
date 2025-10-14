from __future__ import annotations

import logging
import os
import shutil
from contextlib import contextmanager
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import Repository, whoami
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from .config import EnvironmentConfig, RepoArtifacts, ScriptConfig

try:  # pragma: no cover
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger("modernbert.trainer")


def resolve_device(logger: logging.Logger) -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)
    return device


def resolve_flash_attention(enable: bool, logger: logging.Logger):
    if not enable:
        logger.info("FlashAttention disabled via configuration")
        return None
    try:
        from flash_attn.flash_attention import FlashAttention  # type: ignore

        logger.info("FlashAttention module detected")
        return FlashAttention
    except ImportError:
        logger.warning("FlashAttention requested but not available; falling back to standard attention")
        return None


def load_tokenizer(config: ScriptConfig, env: EnvironmentConfig, logger: logging.Logger):
    if config.tokenizer_path and os.path.isdir(config.tokenizer_path):
        logger.info("Loading custom tokenizer from %s", config.tokenizer_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)
            logger.info("Successfully loaded custom tokenizer")
        except Exception as exc:
            logger.warning(
                "Failed to load custom tokenizer from %s: %s. Falling back to base model tokenizer.",
                config.tokenizer_path,
                exc,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_checkpoint,
                use_auth_token=env.huggingface_token,
            )
    else:
        logger.info("Loading tokenizer from hub: %s", config.model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_checkpoint,
            use_auth_token=env.huggingface_token,
        )

    resize_required = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        resize_required = True
        logger.info("Added [PAD] token to tokenizer")

    return tokenizer, resize_required


def load_model(config: ScriptConfig, env: EnvironmentConfig, logger: logging.Logger):
    logger.info("Loading model config from %s", config.model_checkpoint)
    model_config = AutoConfig.from_pretrained(
        config.model_checkpoint,
        use_auth_token=env.huggingface_token,
    )
    model_config.torch_dtype = torch.float16

    logger.info("Loading model weights")
    model = AutoModelForMaskedLM.from_pretrained(
        config.model_checkpoint,
        config=model_config,
        use_auth_token=env.huggingface_token,
        torch_dtype=torch.float16,
    )
    return model


def enable_flash_attention_if_available(
    model: nn.Module,
    flash_attention_cls,
    logger: logging.Logger,
) -> None:
    if flash_attention_cls is None:
        return

    replacements = 0
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.attention = flash_attention_cls()  # type: ignore[attr-defined]
            replacements += 1

    if replacements:
        logger.info("Replaced %d attention modules with FlashAttention", replacements)
    else:
        logger.warning("FlashAttention available but no nn.MultiheadAttention modules were replaced")


def prepare_dataset(config: ScriptConfig, env: EnvironmentConfig, logger: logging.Logger):
    logger.info("Loading dataset %s in streaming mode", config.dataset_name)
    return load_dataset(
        config.dataset_name,
        streaming=True,
        split="train",
        use_auth_token=env.huggingface_token,
    )


def tokenize_dataset(dataset, tokenizer, logger: logging.Logger):
    logger.info("Tokenizing dataset stream")

    def _tokenize(batch: Mapping[str, Sequence[str]]):
        # Handle both 'text' and 'content' field names
        text_field = "text" if "text" in batch else "content"
        return tokenizer(batch[text_field], return_special_tokens_mask=True)

    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )


def setup_repository(
    config: ScriptConfig,
    env: EnvironmentConfig,
    logger: logging.Logger,
) -> RepoArtifacts:
    output_dir = config.run_name
    repo: Optional[Repository] = None
    repo_id: Optional[str] = None

    if env.huggingface_token:
        try:
            user_info = whoami(token=env.huggingface_token)
            username = user_info.get("name") or env.username
        except Exception as exc:
            logger.warning("Falling back to configured username due to hub error: %s", exc)
            username = env.username

        repo_id = f"{username}/{output_dir}"
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        try:
            repo = Repository(
                local_dir=output_dir,
                clone_from=repo_id,
                repo_type="model",
                use_auth_token=env.huggingface_token,
            )
            logger.info("Repository %s ready at %s", repo_id, output_dir)
        except Exception as exc:
            logger.warning("Could not clone %s: %s", repo_id, exc)
            repo = None

    if repo is None:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Using local output directory at %s", output_dir)

    return RepoArtifacts(output_dir=output_dir, repo=repo, repo_id=repo_id)


def fix_batch_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if key in inputs:
            tensor = inputs[key]
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                inputs[key] = tensor.squeeze(0)
            elif tensor.dim() > 2:
                raise ValueError(f"Unexpected tensor shape for {key}: {tensor.shape}")
    if "input_ids" in inputs and inputs["input_ids"].dtype != torch.long:
        inputs["input_ids"] = inputs["input_ids"].long()
    return inputs


class StreamingMLMCollator(DataCollatorForLanguageModeling):
    def __call__(self, batch: Mapping[str, Sequence[Sequence[int]]]):
        if isinstance(batch, Mapping):
            examples = [
                {"input_ids": ids, "attention_mask": mask}
                for ids, mask in zip(batch["input_ids"], batch["attention_mask"])
            ]
        else:
            examples = batch

        collated = super().__call__(examples)
        return fix_batch_inputs(dict(collated))


def forward_pass(
    model: nn.Module,
    device: torch.device,
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    inputs = fix_batch_inputs(inputs)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
        outputs = model(**inputs, return_dict=True)
    if outputs.loss is None:
        raise ValueError("Model forward pass returned no loss")
    return outputs.loss


def evaluate(
    model: nn.Module,
    device: torch.device,
    eval_dataset,
    data_collator: StreamingMLMCollator,
    batch_size: int,
    logger: logging.Logger,
) -> float:
    model.eval()
    losses = []
    iterator = eval_dataset.iter(batch_size=batch_size)
    for batch in tqdm(iterator, desc="Evaluating", leave=False):
        try:
            with torch.no_grad():
                inputs = data_collator(batch)
                loss = forward_pass(model, device, inputs)
            losses.append(loss.item())
        except Exception as exc:
            logger.warning("Evaluation batch skipped due to error: %s", exc)
    model.train()
    return float(sum(losses) / len(losses)) if losses else float("inf")


def log_metrics(run, metrics: Dict[str, float], step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


@contextmanager
def initialize_wandb(
    config: ScriptConfig,
    env: EnvironmentConfig,
    logger: logging.Logger,
):
    if wandb is None:
        logger.info("wandb not installed; skipping experiment tracking")
        yield None
        return

    if not env.wandb_api_key:
        logger.info("WANDB_API_KEY not provided; skipping wandb logging")
        yield None
        return

    try:
        wandb.login(key=env.wandb_api_key)
        run = wandb.init(
            project="modernbert-dutch",
            name=config.run_name,
        )
    except Exception as exc:
        logger.warning("wandb initialisation failed: %s", exc)
        yield None
        return

    try:
        yield run
    finally:
        run.finish()
