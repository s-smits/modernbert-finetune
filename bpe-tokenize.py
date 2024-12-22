from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from datasets import load_dataset
from itertools import islice
from transformers import AutoTokenizer

# --- Configuration ---
DATASET_NAME = "ssmits/fineweb-2-dutch"  # Dataset for tokenizer training
TOKENIZER_SAVE_PATH = "domain_tokenizer"  # Directory to save the trained tokenizer
VOCAB_SIZE = 32768  # Desired vocabulary size
NUM_EXAMPLES_TO_TRAIN = 10000  # Number of examples to use from the streaming dataset
BATCH_SIZE = 1000

# --- Tokenizer Training ---

def train_tokenizer(dataset_iterator, vocab_size=VOCAB_SIZE, save_path=TOKENIZER_SAVE_PATH):
    """
    Trains a WordPiece tokenizer on a streaming dataset.

    Args:
        dataset_iterator: An iterator over the dataset.
        vocab_size: The desired vocabulary size.
        save_path: The directory to save the trained tokenizer.
    """
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        min_frequency=2
    )

    def batch_iterator(batch_size=BATCH_SIZE):
        for i in range(0, NUM_EXAMPLES_TO_TRAIN, batch_size):
            yield [item["content"] for item in islice(dataset_iterator, i, i + batch_size)]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=NUM_EXAMPLES_TO_TRAIN)
    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer trained and saved to {save_path}")
    return tokenizer

# --- Main Execution ---

if __name__ == "__main__":
    # --- Load Dataset (Streaming) ---
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    dataset_iterator = iter(dataset)

    # --- Train Tokenizer ---
    print("Training tokenizer...")
    tokenizer = train_tokenizer(dataset_iterator)

    print("Tokenizer training complete.")
