from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from adopt import ADOPT
from transformers import get_linear_schedule_with_warmup

from .config import RepoArtifacts, ScriptConfig
from .pipeline import (
    StreamingMLMCollator,
    evaluate,
    forward_pass,
    log_metrics,
)


class CombinedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)
        # We don't call super().__init__ because we don't strictly follow its structure
        # but we need to satisfy LRScheduler which accesses param_groups and state_dict.
        # This is a minimal wrapper.
        self.state = {} # Shared state? No, separate states.
        # However, LRScheduler might modify param_groups[i]['lr']. 
        # Since self.param_groups references the lists in underlying optimizers, it should work.

    def step(self, closure=None):
        loss = None
        if closure is not None:
             with torch.enable_grad():
                 loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dict):
        for opt, state in zip(self.optimizers, state_dict):
            opt.load_state_dict(state)


class CurriculumTrainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        tokenized_dataset,
        config: ScriptConfig,
        device: torch.device,
        repo_artifacts: RepoArtifacts,
        logger,
        wandb_run,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = tokenized_dataset
        self.config = config
        self.device = device
        self.logger = logger
        self.wandb_run = wandb_run
        self.output_dir = repo_artifacts.output_dir
        self.repo = repo_artifacts.repo
        
        if config.optimizer == "muon":
            from torch.optim import Muon
            # Separate parameters for Muon
            muon_params = []
            adamw_params = []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                # Use Muon for 2D hidden weights, AdamW for others (embeddings, norms, biases, output head)
                if p.ndim == 2 and "embeddings" not in name and "head" not in name:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)
            
            logger.info(f"Muon params: {len(muon_params)}, AdamW params: {len(adamw_params)}")
            
            opt1 = Muon(
                muon_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.95
            )
            opt2 = torch.optim.AdamW(
                adamw_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            self.optimizer = CombinedOptimizer([opt1, opt2])
            
        elif config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else: # adopt
            self.optimizer = ADOPT(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=config.total_train_steps,
        )
        try:
            self.scaler = torch.amp.GradScaler(
                device_type="cuda",
                enabled=device.type == "cuda",
            )
        except TypeError:
            # Fallback for older torch versions where torch.amp.GradScaler might not take device_type
            # or if we should use torch.cuda.amp.GradScaler
            if device.type == "cuda":
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            else:
                # CPU or other device where GradScaler might just be a pass-through if enabled=False
                # If enabled=False, torch.cuda.amp.GradScaler() works fine too.
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)

    def train(self) -> None:
        chunk_size_dataset = max(
            self.config.eval_size_per_chunk + 1,
            self.config.estimated_dataset_rows // max(1, len(self.config.mlm_probabilities)),
        )
        global_step = 0
        push_interval = self.config.resolved_push_interval

        self.model.to(self.device)

        for epoch in range(self.config.num_train_epochs):
            for index, mlm_probability in enumerate(self.config.mlm_probabilities):
                self.logger.info(
                    "Epoch %d/%d with MLM probability %.2f",
                    epoch + 1,
                    self.config.num_train_epochs,
                    mlm_probability,
                )

                data_collator = StreamingMLMCollator(
                    tokenizer=self.tokenizer,
                    mlm_probability=mlm_probability,
                )

                offset = index * chunk_size_dataset
                train_dataset = (
                    self.dataset.skip(offset + self.config.eval_size_per_chunk)
                    .take(chunk_size_dataset)
                    .shuffle(seed=42, buffer_size=10_000)
                )
                eval_dataset = self.dataset.skip(offset).take(
                    self.config.eval_size_per_chunk
                )

                iterator = train_dataset.iter(
                    batch_size=self.config.per_device_train_batch_size
                )

                for step, batch in enumerate(iterator, start=1):
                    try:
                        inputs = data_collator(batch)
                        loss = forward_pass(self.model, self.device, inputs)
                    except Exception as exc:
                        self.logger.warning("Training batch skipped due to error: %s", exc)
                        continue

                    scaled_loss = loss / self.config.gradient_accumulation_steps
                    self.scaler.scale(scaled_loss).backward()

                    if step % self.config.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        global_step += 1

                        log_metrics(self.wandb_run, {"loss": float(loss.item())}, global_step)

                        eval_interval = max(1, self.config.total_steps_per_epoch // (self.config.num_train_epochs * 4))
                        if global_step % eval_interval == 0:
                            eval_loss = evaluate(
                                self.model,
                                self.device,
                                eval_dataset,
                                data_collator,
                                self.config.per_device_train_batch_size,
                                self.logger,
                            )
                            self.logger.info(
                                "Evaluation loss at step %d: %.4f",
                                global_step,
                                eval_loss,
                            )
                            log_metrics(self.wandb_run, {"eval_loss": float(eval_loss)}, global_step)

                        if global_step % push_interval == 0:
                            self._save_and_push(global_step, epoch, mlm_probability)

        self._save_and_push(None, self.config.num_train_epochs - 1, 0.0)

    def _save_and_push(
        self,
        global_step: Optional[int],
        epoch: int,
        mlm_probability: float,
    ) -> None:
        suffix = (
            f"step-{global_step}-epoch-{epoch + 1}-mlm-{mlm_probability:.2f}"
            if global_step is not None
            else f"final-epoch-{self.config.num_train_epochs}"
        )
        self.logger.info("Persisting model checkpoint (%s)", suffix)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        if self.repo is None:
            return
        commit_message = (
            f"Step {global_step} - Epoch {epoch + 1} (MLM {mlm_probability:.2f})"
            if global_step is not None
            else f"Final model - Epoch {self.config.num_train_epochs}"
        )
        try:
            self.repo.push_to_hub(
                commit_message=commit_message,
                blocking=False,
            )
        except Exception as exc:
            self.logger.warning("Failed to push checkpoint to hub: %s", exc)
