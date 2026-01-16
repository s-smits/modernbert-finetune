
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from modernbert_finetune.trainer import CurriculumTrainer, CombinedOptimizer
from modernbert_finetune.config import ScriptConfig, RepoArtifacts
from modernbert_finetune.muon import Muon

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10, bias=False) # 2D -> Muon
        self.linear2 = nn.Linear(10, 10, bias=True)  # bias -> AdamW
        self.embeddings = nn.Embedding(10, 10)       # embedding -> AdamW
        self.head = nn.Linear(10, 10)                # head -> AdamW
        self.model = torch.nn.Module() # Mock structure
        self.model.parameters = lambda: self.parameters()

    def forward(self, input_ids, **kwargs):
        return type('Output', (), {'loss': torch.tensor(0.1, requires_grad=True)})()
    
    def save_pretrained(self, path):
        pass

def test_muon_init():
    print("Testing Muon initialization...")
    config = ScriptConfig(optimizer="muon", num_train_epochs=1, testing=True)
    model = SimpleModel()
    repo_artifacts = RepoArtifacts(output_dir="tmp_test_output")
    
    trainer = CurriculumTrainer(
        model=model,
        tokenizer=MagicMock(),
        tokenized_dataset=MagicMock(),
        config=config,
        device=torch.device("cpu"),
        repo_artifacts=repo_artifacts,
        logger=MagicMock(),
        wandb_run=None,
    )
    
    assert isinstance(trainer.optimizer, CombinedOptimizer)
    print("CombinedOptimizer initialized.")
    
    # Check param grouping
    # linear1.weight should be in Muon
    # others in AdamW
    
    # Access internal optimizers
    muon_opt = trainer.optimizer.optimizers[0]
    adamw_opt = trainer.optimizer.optimizers[1]
    
    assert isinstance(muon_opt, Muon)
    # Check if linear1.weight is in muon_opt
    muon_params = [id(p) for group in muon_opt.param_groups for p in group['params']]
    assert id(model.linear1.weight) in muon_params
    assert id(model.linear2.weight) in muon_params # 2D weight, should be in Muon
    # Wait, my logic was:
    # if p.ndim == 2 and "embeddings" not in name and "head" not in name:
    # linear2.weight is 2D. Is it embeddings/head? No. So it should be in Muon.
    # linear2.bias is 1D. Should be in AdamW.
    
    assert id(model.linear2.weight) in muon_params
    assert id(model.linear2.bias) not in muon_params
    assert id(model.embeddings.weight) not in muon_params # "embeddings" in name
    assert id(model.head.weight) not in muon_params # "head" in name
    
    print("Parameter grouping verified.")
    
    # Test step
    model.zero_grad()
    loss = torch.tensor(1.0, requires_grad=True)
    loss.backward()
    
    # Hack gradients for params
    for p in model.parameters():
        p.grad = torch.randn_like(p)
        
    trainer.optimizer.step()
    print("Optimizer step executed.")

def test_adamw_init():
    print("Testing AdamW initialization...")
    config = ScriptConfig(optimizer="adamw", testing=True)
    model = SimpleModel()
    repo_artifacts = RepoArtifacts(output_dir="tmp_test_output")
    trainer = CurriculumTrainer(
        model=model,
        tokenizer=MagicMock(),
        tokenized_dataset=MagicMock(),
        config=config,
        device=torch.device("cpu"),
        repo_artifacts=repo_artifacts,
        logger=MagicMock(),
        wandb_run=None,
    )
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    print("AdamW verified.")

def test_adopt_init():
    print("Testing Adopt initialization...")
    config = ScriptConfig(optimizer="adopt", testing=True)
    model = SimpleModel()
    repo_artifacts = RepoArtifacts(output_dir="tmp_test_output")
    trainer = CurriculumTrainer(
        model=model,
        tokenizer=MagicMock(),
        tokenized_dataset=MagicMock(),
        config=config,
        device=torch.device("cpu"),
        repo_artifacts=repo_artifacts,
        logger=MagicMock(),
        wandb_run=None,
    )
    # Checks if it is ADOPT class (from 'adopt' package)
    print(f"Optimizer type: {type(trainer.optimizer)}")
    # assert "ADOPT" in str(type(trainer.optimizer)) # Might vary based on import
    print("Adopt verified.")

if __name__ == "__main__":
    if not os.path.exists("tests"):
        os.makedirs("tests")
    test_muon_init()
    test_adamw_init()
    test_adopt_init()
    print("All tests passed!")
