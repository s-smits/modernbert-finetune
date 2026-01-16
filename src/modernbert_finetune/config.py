from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover
    from huggingface_hub import Repository


@dataclass
class ScriptConfig:
    model_checkpoint: str = "answerdotai/ModernBERT-base"
    dataset_name: str = "ssmits/fineweb-2-dutch"
    tokenizer_path: str = "domain_tokenizer"  # Set to None or empty to use base model tokenizer
    estimated_dataset_rows: int = 86_500_000
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    eval_size_ratio: float = 0.05
    flash_attention: bool = True
    testing: bool = False
    mlm_probabilities: Sequence[float] = field(
        default_factory=lambda: (0.3, 0.2, 0.18, 0.16, 0.14)
    )
    push_interval: int = 100_000
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    optimizer: str = "adopt"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def total_steps_per_epoch(self) -> int:
        return max(1, math.ceil(self.estimated_dataset_rows / self.effective_batch_size))

    @property
    def total_train_steps(self) -> int:
        return self.total_steps_per_epoch * self.num_train_epochs

    @property
    def eval_size_per_chunk(self) -> int:
        return max(1, int(100_000 * self.eval_size_ratio))

    @property
    def resolved_push_interval(self) -> int:
        return 10_000 if self.testing else self.push_interval

    @property
    def run_name(self) -> str:
        suffix = "test" if self.testing else "full"
        return f"{self.model_checkpoint.split('/')[-1]}-dutch-{suffix}"


@dataclass
class EnvironmentConfig:
    huggingface_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("HUGGINGFACE_TOKEN")
    )
    wandb_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("WANDB_API_KEY")
    )
    username: str = "ssmits"


@dataclass
class RepoArtifacts:
    output_dir: str
    repo: Optional["Repository"] = None
    repo_id: Optional[str] = None
