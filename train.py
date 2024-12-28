import os
import math
import shutil
import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from huggingface_hub import whoami, Repository
from tqdm.auto import tqdm
import wandb
from adopt import ADOPT
from typing import List, Dict, Any

# --- Configuration ---
model_checkpoint = "answerdotai/ModernBERT-base"
dataset_name = "ssmits/fineweb-2-dutch"
username = "ssmits"
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN", None)
wandb_api_key = os.environ.get("WANDB_API_KEY", None)  # Optional
tokenizer_path = "domain_tokenizer"  # Path to custom tokenizer directory

# --- Dataset size (in rows) ---
estimated_dataset_size_in_rows = 86_500_000

# --- Training Config ---
num_train_epochs = 1
# Reduce or remove chunk size to allow for dynamic batching
chunk_size = None  # Remove chunk size
per_device_train_batch_size = 4
gradient_accumulation_steps = 2
eval_size_ratio = 0.05
total_save_limit = 2

effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
total_steps_per_epoch = math.ceil(
    estimated_dataset_size_in_rows / effective_batch_size
)
total_train_steps = total_steps_per_epoch * num_train_epochs
eval_size_per_chunk = int(100_000 * eval_size_ratio)

# --- Testing Mode ---
TESTING = False  # Set to True for testing, False for full training
FLASH_ATTENTION = True

if TESTING:
    push_interval = 10_000
else:
    push_interval = 100_000

# --- Check for FlashAttention Installation ---
if FLASH_ATTENTION:
    try:
        import flash_attn
        print("FlashAttention is already installed.")
    except ImportError:
        print("FlashAttention is not installed. Installing...")
        try:
            import subprocess
            subprocess.run(["pip", "install", "flash-attn", "--no-build-isolation"], check=True)
            import flash_attn
            print("FlashAttention installed successfully.")
        except Exception as e:
            print(f"Error installing FlashAttention: {e}")
            exit()

# --- Flash-attn Integration Check ---
try:
    from flash_attn.flash_attention import FlashAttention
    print("FlashAttention is available.")
    flash_attn_available = True
except ImportError:
    print("FlashAttention is not available. Using standard attention.")
    flash_attn_available = False

# --- Tokens ---
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN", None)
wandb_api_key = os.environ.get("WANDB_API_KEY", None)

# --- Initialize WandB ---
if wandb_api_key is not None:
    wandb.login(key=wandb_api_key)
else:
    wandb.login()

wandb.init(
    project="modernbert-dutch",
    name=f"{model_checkpoint.split('/')[-1]}-dutch-{'test' if TESTING else 'full'}",
)

# --- Load Tokenizer and Model ---
print(f"Loading model and tokenizer from {model_checkpoint}...")

# Check if custom tokenizer exists, otherwise use default
if os.path.exists(tokenizer_path) and any(fname.startswith('spm') for fname in os.listdir(tokenizer_path)):
    print(f"Loading custom SentencePiece tokenizer from {tokenizer_path}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # Add the pad_token if it's not already in the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
elif os.path.exists(tokenizer_path) and os.path.isfile(os.path.join(tokenizer_path, "tokenizer.json")):
    print(f"Loading custom tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
else:
    print(f"Using default tokenizer from {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, use_auth_token=huggingface_token
    )

print(f"Loading model config from {model_checkpoint}...")
config = AutoConfig.from_pretrained(
    model_checkpoint, use_auth_token=huggingface_token
)
config.torch_dtype = "float16"
print(f"Model config loaded and modified: {config}")

model = AutoModelForMaskedLM.from_pretrained(
    model_checkpoint, config=config, use_auth_token=huggingface_token
)
print("Model and tokenizer loaded.")

# --- Integrate Flash-attn (if available) ---
if flash_attn_available:
    print("Replacing standard attention with FlashAttention...")
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.attention = FlashAttention()
    print("FlashAttention integrated.")

# --- Load Dataset (Streaming) ---
print(f"Loading dataset {dataset_name} (streaming)...")
dataset = load_dataset(
    dataset_name,
    streaming=True,
    split="train",
    use_auth_token=huggingface_token,
)
print("Dataset loaded.")

# --- Tokenization Function ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        # No truncation and max_length to allow dynamic padding truncation=True, max_length=chunk_size, padding="longest",
        return_special_tokens_mask=True,
    )

# --- Tokenize Dataset ---
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
)
print("Dataset tokenized.")

# --- Set up Hugging Face Repository ---
print("Setting up Hugging Face repository...")
try:
    user_info = whoami(token=huggingface_token)
    username = user_info["name"]
except Exception as e:
    print(f"Error fetching username: {e}. Using default username '{username}'.")

model_name = model_checkpoint.split("/")[-1]
output_dir = f"{model_name}-dutch-{'test' if TESTING else 'full'}"
repo_name = f"{username}/{output_dir}"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

repo = Repository(
    local_dir=output_dir,
    clone_from=repo_name,
    repo_type="model",
    use_auth_token=huggingface_token,
)
print(f"Repository '{repo_name}' set up at '{output_dir}'.")

# --- Device Configuration ---
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
model.to(device)

# --- Optimizer and Scheduler ---
optimizer = ADOPT(model.parameters(), lr=5e-4, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_train_steps
)

# --- AMP scaler for mixed precision ---
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# --- Helper Function to Fix Batch Inputs ---
def fix_batch_inputs(inputs: dict) -> dict:
    """
    Ensures that input tensors have the correct shape and dtype.
    - Removes any extra dimensions (e.g., [1, batch, seq_len] -> [batch, seq_len]).
    - Casts input_ids to torch.long.
    """
    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        if key in inputs:
            if inputs[key].dim() == 3 and inputs[key].shape[0] == 1:
                inputs[key] = inputs[key].squeeze(0)
            elif inputs[key].dim() > 2:
                raise ValueError(
                    f"Unexpected tensor shape for {key}: {inputs[key].shape}"
                )
    if "input_ids" in inputs and inputs["input_ids"].dtype != torch.long:
        inputs["input_ids"] = inputs["input_ids"].long()
    return inputs

# --- Forward Pass Function ---
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

# --- Evaluation Function ---
def evaluate(model, eval_dataset, data_collator):
    """
    Evaluates the model on the evaluation dataset.
    Returns the average loss.
    """
    model.eval()
    losses = []
    eval_iterator = eval_dataset.iter(batch_size=per_device_train_batch_size)
    for batch in tqdm(eval_iterator, desc="Evaluating"):
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=(device.type == "cuda")
        ):
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

# --- Dynamic Padding Data Collator ---
class DynamicPaddingDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator that dynamically pads the inputs for language modeling.
    This ensures that all sequences within a batch have the same length,
    but the overall length can vary between batches.
    """

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Find the maximum length within the current batch
        max_length = max(len(example["input_ids"]) for example in examples)

        # Pad or truncate each example to the max_length
        batch = []
        for example in examples:
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]

            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                # Pad
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            elif padding_length < 0:
                # Truncate (if enabled in your tokenizer)
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]

            batch.append({"input_ids": input_ids, "attention_mask": attention_mask})

        # Convert to PyTorch tensors
        batch = {
            "input_ids": torch.tensor([item["input_ids"] for item in batch]),
            "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
        }

        # Apply the rest of the data collation logic (MLM masking, etc.)
        batch = self.torch_call(batch)  # Use torch_call instead of __call__ to call the parent's method

        # Ensure correct shapes and dtypes
        batch = fix_batch_inputs(batch)

        return batch

# --- Training Function with Curriculum Learning ---
def train_with_curriculum(mlm_probabilities, chunk_size_dataset):
    """
    Trains the model using curriculum learning with varying MLM probabilities.
    """
    model.train()
    global_step = 0

    for epoch in range(num_train_epochs):
        for i, mlm_probability in enumerate(mlm_probabilities):
            print(
                f"\nEpoch {epoch + 1}/{num_train_epochs}, MLM Probability: {mlm_probability}"
            )

            data_collator = DynamicPaddingDataCollator(
                tokenizer=tokenizer, mlm_probability=mlm_probability
            )

            train_dataset = (
                tokenized_dataset.skip(
                    i * chunk_size_dataset + eval_size_per_chunk
                )
                .take(chunk_size_dataset)
                .shuffle(seed=42, buffer_size=10_000)
            )
            eval_dataset = tokenized_dataset.skip(i * chunk_size_dataset).take(
                eval_size_per_chunk
            )

            train_iterator = train_dataset.iter(batch_size=per_device_train_batch_size)
            for step, batch in enumerate(
                tqdm(train_iterator, desc=f"Training (MLM {mlm_probability})")
            ):
                try:
                    inputs = data_collator([batch])
                    loss = forward_pass(model, inputs)
                except Exception as e:
                    print(f"Training batch failed: {e}. Skipping.")
                    continue

                scaler.scale(loss / gradient_accumulation_steps).backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()  # Clear cache
                    global_step += 1

                    wandb.log({"loss": float(loss.item())}, step=global_step)

                    # Evaluation
                    eval_interval = total_steps_per_epoch // (num_train_epochs * 4)
                    if eval_interval > 0 and (global_step % eval_interval == 0):
                        eval_loss = evaluate(model, eval_dataset, data_collator)
                        print(f"Evaluation loss at step {global_step}: {eval_loss}")
                        wandb.log({"eval_loss": eval_loss}, step=global_step)

                    # Push to hub incl TESTING
                    if global_step % push_interval == 0:
                        print(f"Saving and pushing model at step {global_step}...")
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        repo.push_to_hub(
                            commit_message=f"Step {global_step} - Epoch {epoch + 1}, MLM Probability {mlm_probability}",
                            blocking=False,
                        )
                        print(f"Model saved and pushed at step {global_step}.")

    # Final Save and Push
    print("\nSaving and pushing final model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    repo.push_to_hub(
        commit_message=f"Final model - Epoch {num_train_epochs}", blocking=False
    )
    print("Final model saved and pushed.")

# --- Define MLM Probabilities and Chunk Sizes ---
masking_probabilities = [0.3, 0.2, 0.18, 0.16, 0.14]
chunk_size_dataset = estimated_dataset_size_in_rows // len(
    masking_probabilities
)

# --- Start Training ---
try:
    train_with_curriculum(masking_probabilities, chunk_size_dataset)
except Exception as e:
    print(f"Error during training: {e}")

print("Fine-tuning complete!")
