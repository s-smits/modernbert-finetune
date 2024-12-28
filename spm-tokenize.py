from sentencepiece import SentencePieceTrainer
from datasets import load_dataset
import os

# --- Configuration ---
DATASET_NAME = "ssmits/fineweb-2-dutch"
TOKENIZER_SAVE_PATH = "domain_tokenizer"
VOCAB_SIZE = 32768
NUM_EXAMPLES_TO_TRAIN = 1_000_000  # Increased for SPM
MODEL_TYPE = "unigram"  # Can be changed to "bpe" for Byte Pair Encoding
BATCH_SIZE = 1000

# --- SPM Tokenizer Training ---

def train_tokenizer(dataset_iterator, vocab_size=VOCAB_SIZE, save_path=TOKENIZER_SAVE_PATH, model_type=MODEL_TYPE):
    """
    Trains a SentencePiece tokenizer on a streaming dataset.

    Args:
        dataset_iterator: An iterator over the dataset.
        vocab_size: The desired vocabulary size.
        save_path: The directory to save the trained tokenizer.
        model_type: The type of SPM model ('unigram' or 'bpe').
    """
    # Create a temporary file to store the text data
    temp_file = "temp_dataset.txt"

    with open(temp_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset_iterator):
            f.write(item["content"] + "\n")
            if (i + 1) >= NUM_EXAMPLES_TO_TRAIN:
                break

    # Define SPM training arguments
    spm_train_args = [
        f"--input={temp_file}",
        f"--model_prefix=spm",  # Output model name prefix
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--split_digits=true",
        f"--normalization_rule_name=nmt_nfkc_cf",
        f"--add_dummy_prefix=false",
        f"--remove_extra_whitespaces=true",
        f"--pad_id=3",
        f"--unk_id=0",
        f"--bos_id=1",
        f"--eos_id=2",
        # For potentially faster training:
        # f"--input_sentence_size=1000000", # Limit the size of the corpus used for training
        # f"--shuffle_input_sentence=true", # Shuffle the training data
    ]
    spm_train_args = " ".join(spm_train_args)

    # Train the SentencePiece model
    SentencePieceTrainer.train(spm_train_args)

    # Move the trained model and vocab to the save directory
    os.makedirs(save_path, exist_ok=True)
    os.rename("spm.model", os.path.join(save_path, "spm.model"))
    os.rename("spm.vocab", os.path.join(save_path, "spm.vocab"))

    # Remove the temporary file
    os.remove(temp_file)

    print(f"Tokenizer trained and saved to {save_path}")

# --- Main Execution ---

if __name__ == "__main__":
    # --- Load Dataset (Streaming) ---
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    dataset_iterator = iter(dataset)

    # --- Train Tokenizer ---
    print("Training tokenizer...")
    train_tokenizer(dataset_iterator)

    print("Tokenizer training complete.")
