import os
import math
import shutil
import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
    AdamW
)
from datasets import load_dataset
from huggingface_hub import whoami, Repository
from tqdm.auto import tqdm
import wandb

# Disable Torch Dynamo to avoid backend/triton issues on older GPUs
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Configuration
model_checkpoint = "answerdotai/ModernBERT-base"
dataset_name = "ssmits/fineweb-2-dutch"
username = "ssmits"
huggingface_token = os.environ["HUGGINGFACE_TOKEN"]
wandb_token = os.environ["WANDB_API_KEY"]

os.environ["TORCH_LOGS"] = ""
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

# Dataset size (in rows)
estimated_dataset_size_in_rows = 86_500_000

# Training Config
num_train_epochs = 1
chunk_size = 8192
gradient_accumulation_steps = 32
per_device_train_batch_size = 1
eval_size_ratio = 0.05
total_save_limit = 2

effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
total_steps_per_epoch = math.ceil(estimated_dataset_size_in_rows / effective_batch_size)
total_train_steps = total_steps_per_epoch * num_train_epochs
eval_size_per_chunk = int(1_000_000 * eval_size_ratio)

# Tokens
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN", None)
wandb_token = os.environ.get("WANDB_API_KEY", None)

# Initialize WandB
if wandb_token is not None:
    wandb.login(key=wandb_token)
else:
    wandb.login()

# Set up WandB project name
wandb.init(project="modernbert-dutch", name=f"{model_checkpoint.split('/')[-1]}-dutch")

# Load Tokenizer and Model
print(f"Loading model and tokenizer from {model_checkpoint}...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_auth_token=huggingface_token)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint, use_auth_token=huggingface_token)
print("Model and tokenizer loaded.")

# Load Dataset (Streaming)
print(f"Loading dataset {dataset_name} (streaming)...")
dataset = load_dataset(dataset_name, streaming=True, split="train", use_auth_token=huggingface_token)
print("Dataset loaded.")

# Tokenization Function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=chunk_size,
        padding="longest",
        return_special_tokens_mask=True,
    )

# Tokenize Dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=dataset.column_names
)
print("Dataset tokenized.")

# Set up Hugging Face Repository
print("Setting up Hugging Face repository...")
try:
    user_info = whoami(token=huggingface_token)
    username = user_info["name"]
except Exception as e:
    print(f"Error fetching username: {e}. Using default username '{username}'.")

model_name = model_checkpoint.split("/")[-1]
output_dir = f"{model_name}-dutch"
repo_name = f"{username}/{output_dir}"

# Clone or Create Repository
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

repo = Repository(
    local_dir=output_dir,
    clone_from=repo_name,
    repo_type="model",
    use_auth_token=huggingface_token,
)
print(f"Repository '{repo_name}' set up at '{output_dir}'.")

# Device Configuration: Single GPU (cuda:0)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_train_steps
)

# Disable Torch Dynamo on model forward
model.forward = torch._dynamo.disable(model.forward)

# AMP scaler for mixed precision
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# Helper Function to Fix Batch Inputs
def fix_batch_inputs(inputs: dict) -> dict:
    """
    Ensures that input tensors have the correct shape and dtype.
    - Removes any extra dimensions (e.g., [1, batch, seq_len] -> [batch, seq_len]).
    - Casts input_ids to torch.long.
    """
    # Ensure input_ids and attention_mask are 2D [batch, seq_length]
    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        if key in inputs:
            if inputs[key].dim() == 3 and inputs[key].shape[0] == 1:
                inputs[key] = inputs[key].squeeze(0)
            elif inputs[key].dim() > 2:
                raise ValueError(f"Unexpected tensor shape for {key}: {inputs[key].shape}")
    
    # Cast input_ids to long
    if "input_ids" in inputs and inputs["input_ids"].dtype != torch.long:
        inputs["input_ids"] = inputs["input_ids"].long()
    
    return inputs

# Forward Pass Function
def forward_pass(model, inputs):
    """
    Performs a forward pass with autocast for FP16.
    Returns the loss.
    """
    inputs = fix_batch_inputs(inputs)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        outputs = model(**inputs, return_dict=True)
    
    if outputs.loss is None:
        raise ValueError("Model did not return a loss.")
    
    return outputs.loss

# Evaluation Function
def evaluate(model, eval_dataset, data_collator):
    """
    Evaluates the model on the evaluation dataset.
    Returns the average loss.
    """
    model.eval()
    losses = []
    eval_iterator = eval_dataset.iter(batch_size=per_device_train_batch_size)
    
    for batch in tqdm(eval_iterator, desc="Evaluating"):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            inputs = data_collator([batch])
            try:
                loss = forward_pass(model, inputs)
                losses.append(loss.item())
            except Exception as e:
                print(f"Evaluation batch failed: {e}. Skipping.")
                continue
    
    model.train()
    average_loss = sum(losses) / len(losses) if losses else float("inf")
    return average_loss

# Stable Data Collator
class StableDataCollator(DataCollatorForLanguageModeling):
    """
    A stable data collator that ensures correct tensor shapes and dtypes.
    """
    def __call__(self, examples):
        batch = super().__call__(examples)
        batch = fix_batch_inputs(batch)
        return batch

# Training Function with Curriculum Learning
def train_with_curriculum(mlm_probabilities, chunk_size_dataset):
    """
    Trains the model using curriculum learning with varying MLM probabilities.
    """
    model.train()
    global_step = 0

    for epoch in range(num_train_epochs):
        for i, mlm_probability in enumerate(mlm_probabilities):
            print(f"\nEpoch {epoch + 1}/{num_train_epochs}, MLM Probability: {mlm_probability}")
    
            data_collator = StableDataCollator(
                tokenizer=tokenizer, mlm_probability=mlm_probability
            )
    
            # Define training and evaluation datasets for the current chunk
            train_dataset = (
                tokenized_dataset
                .skip(i * chunk_size_dataset + eval_size_per_chunk)
                .take(chunk_size_dataset)
                .shuffle(seed=42, buffer_size=10_000)
            )
            eval_dataset = tokenized_dataset.skip(i * chunk_size_dataset).take(eval_size_per_chunk)
    
            train_iterator = train_dataset.iter(batch_size=per_device_train_batch_size)
            for step, batch in enumerate(tqdm(train_iterator, desc=f"Training (MLM {mlm_probability})")):
                try:
                    inputs = data_collator([batch])
                    loss = forward_pass(model, inputs)
                except Exception as e:
                    print(f"Training batch failed: {e}. Skipping.")
                    continue
    
                # Scale loss for gradient accumulation
                scaler.scale(loss / gradient_accumulation_steps).backward()
    
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
    
                    # Log loss to WandB
                    wandb.log({"loss": float(loss.item())}, step=global_step)
    
                    # Evaluation
                    eval_interval = total_steps_per_epoch // (num_train_epochs * 4)
                    if eval_interval > 0 and (global_step % eval_interval == 0):
                        eval_loss = evaluate(model, eval_dataset, data_collator)
                        print(f"Evaluation loss at step {global_step}: {eval_loss}")
                        wandb.log({"eval_loss": eval_loss}, step=global_step)
    
                    # Save intermediate models
                    save_interval = total_steps_per_epoch // (num_train_epochs * 2)
                    if save_interval > 0 and ((step + 1) % save_interval == 0):
                        print("Saving and pushing intermediate model...")
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        repo.push_to_hub(
                            commit_message=f"Intermediate model - Epoch {epoch + 1}, MLM Probability {mlm_probability}, Step {global_step}",
                            blocking=False,
                        )
                        print("Intermediate model saved and pushed.")
    
    # Final Save
    print("\nSaving and pushing final model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    repo.push_to_hub(commit_message=f"Final model - Epoch {num_train_epochs}", blocking=False)
    print("Final model saved and pushed.")

# Define MLM Probabilities and Chunk Sizes
masking_probabilities = [0.3, 0.2, 0.18, 0.16, 0.14]
chunk_size_dataset = estimated_dataset_size_in_rows // len(masking_probabilities)

# Start Training
try:
    train_with_curriculum(masking_probabilities, chunk_size_dataset)
except Exception as e:
    print(f"Error during training: {e}")

print("Fine-tuning complete!")
