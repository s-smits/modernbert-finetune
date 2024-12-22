# Fine-tuning ModernBERT on a Dutch Dataset with Custom Tokenizer Training

**⚠️ Work in Progress - Contributions Welcome! ⚠️**

This repository provides scripts and instructions for:

1. **Training a WordPiece tokenizer** on a Dutch dataset (or any other dataset from the Hugging Face Hub).
2. **Fine-tuning the ModernBERT-base model** on the same Dutch dataset, optionally using the custom-trained tokenizer.

It leverages the Hugging Face Transformers, Tokenizers, and Datasets libraries for efficient training.

**We are actively developing this project and welcome contributions from the community! If you're interested in helping out, please feel free to open issues, submit pull requests, or reach out to us directly.**

## Features

*   **Custom Tokenizer Training:**
    *   Trains a WordPiece tokenizer using the `tokenizers` library.
    *   Supports streaming datasets for efficient handling of large corpora.
    *   Configurable vocabulary size and training examples.
*   **Model Fine-tuning:**
    *   Fine-tunes the `answerdotai/ModernBERT-base` model (or another specified checkpoint).
    *   Uses `Trainer` from `transformers` for streamlined training.
    *   Supports dynamic batching with a custom `DataCollator`.
    *   Implements curriculum learning by gradually decreasing the MLM masking probability.
    *   Uses gradient accumulation to simulate larger batch sizes.
    *   **Uses the ADOPT optimizer for improved convergence.**
    *   **Optionally integrates FlashAttention 2 for faster training (requires a compatible GPU - see details below).**
    *   Includes evaluation steps during training.
    *   Automatically pushes intermediate and final models to the Hugging Face Hub.
*   **Weights & Biases (WandB) Integration (Optional):** Tracks and visualizes training runs in real-time.

## Prerequisites

*   **Hugging Face Account:** You need a Hugging Face account. Sign up [here](https://huggingface.co/join).
*   **Hugging Face API Token:** Generate a User Access Token (with "write" access) from your [Hugging Face profile settings](https://huggingface.co/settings/tokens).
*   **WandB Account (Optional):** Create a free account at [wandb.ai](https://wandb.ai/).
*   **WandB API Key (Optional):** Get your API key from your [WandB settings](https://wandb.ai/settings).
*   **Environment:** A GPU environment is strongly recommended for model fine-tuning. Tokenizer training can be done on a CPU.
*   **GPU Compatibility for FlashAttention 2:** FlashAttention 2 requires a GPU with compute capability >= 7.0. This means **Turing (e.g., T4, RTX 20xx), Ampere (e.g., A100, RTX 30xx), Ada Lovelace (e.g., RTX 40xx), or newer architectures.**

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/s-smits/modernbert-finetune.git
    cd modernbert-finetune
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

### Environment Variables

Set the following environment variables:

```bash
export HUGGINGFACE_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_api_key"  # Optional
```

Replace `"your_huggingface_token"` with your actual Hugging Face token and `"your_wandb_api_key"` with your WandB API key.

### Script Parameters

The `train.py` script defines several configurable parameters. You can modify these directly in the file or override them using environment variables.

**Tokenizer Training Parameters:**

| Parameter               | Default Value          | Description                                                                   |
| :---------------------- | :--------------------- | :---------------------------------------------------------------------------- |
| `DATASET_NAME`          | "ssmits/fineweb-2-dutch" | The name of the dataset on the Hugging Face Hub to use for training.       |
| `TOKENIZER_SAVE_PATH`   | "domain_tokenizer_bpe" | The directory to save the trained tokenizer.                             |
| `VOCAB_SIZE`            | 30000                  | The desired vocabulary size.                                              |
| `TOKENIZER_TRAIN_SAMPLE_SIZE` | 10000                  | The number of examples from the dataset to use for training the tokenizer.             |

**Model Fine-tuning Parameters:**

| Parameter                       | Default Value                 | Description                                                                                                                                |
| :------------------------------ | :---------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| `model_checkpoint`              | "answerdotai/ModernBERT-base" | The base pre-trained ModernBERT model to use.                                                                                             |
| `dataset_name`                  | "ssmits/fineweb-2-dutch"      | The name of the dataset on the Hugging Face Hub to use for fine-tuning.                                                                   |
| `num_train_epochs`              | 1                             | The number of training epochs.                                                                                                             |
| `per_device_train_batch_size`    | 4                             | The batch size per GPU. Adjust based on your GPU memory.                                                                                    |
| `gradient_accumulation_steps`   | 2                            | The number of steps to accumulate gradients over before performing an optimizer step. Modify based on desired effective batch size and GPU memory. |
| `eval_size_ratio`               | 0.05                          | The proportion of the dataset to use for evaluation.                                                                                       |
| `masking_probabilities`         | \[0.3, 0.2, 0.18, 0.16, 0.14] | The curriculum learning masking probabilities.                                                                                             |
| `estimated_dataset_size_in_rows` | 86500000                     | The estimated number of rows in your dataset.                                                                                            |
| `username`                      | "ssmits"                      | Your Hugging Face username.                                                                                                                |
| `total_save_limit`              | 2                             | The maximum number of saved model checkpoints to keep.                                                                                    |
| `output_dir`                    | "modernbert-dutch-model"                | The directory to save the fine-tuned model.                                                                                |
| `repo_name`                     | "your_username/modernbert-dutch"   | The name of the repository on the Hugging Face Hub to push the model to (replace `your_username`).                                      |
| `push_interval`                 | 100000                         | How often to push the model to the Hugging Face Hub (in steps).                                                                          |
| `eval_size_per_chunk`           | 5000                          | The size of the evaluation set to use for each chunk in curriculum learning.                                                              |
| `learning_rate`                 | 5e-4                          | The learning rate for the optimizer.                                                                                                      |
| `weight_decay`                  | 0.01                          | The weight decay for the optimizer.                                                                                                       |

**Mode Selection Parameters:**

| Parameter        | Value    | Description                                                                                                                               |
| :--------------- | :------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| `TOKENIZE_ONLY`  | `True`   | Train the tokenizer only and exit.                                                                                                       |
|                  | `False`  | Proceed to model training after (or instead of) tokenizer training.                                                                        |
| `TRAIN_MODEL`   | `True`   | Train the model. If `TOKENIZE_ONLY` is `False`, a custom tokenizer will be loaded if available, otherwise, the default model tokenizer is used. |
|                  | `False`  | Do not train the model. Useful if you only want to train the tokenizer.                                                                      |

## Running the Scripts

### 1. Tokenizer Training (Optional)

If you want to train a new tokenizer:

1. **Configure Parameters:**
    *   Set `TOKENIZE_ONLY = True` in `train.py`.
    *   Adjust tokenizer training parameters (e.g., `VOCAB_SIZE`, `TOKENIZER_TRAIN_SAMPLE_SIZE`) in `train.py` as needed.

2. **Run the Script:**

    ```bash
    python train.py
    ```

This will train a tokenizer and save it to the `domain_tokenizer_bpe` directory (or the path you specified).

### 2. Model Fine-tuning

1. **Configure Parameters:**
    *   Set `TOKENIZE_ONLY = False` in `train.py`.
    *   Set `TRAIN_MODEL = True` in `train.py`.
    *   Adjust model fine-tuning parameters (e.g., `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `repo_name`) in `train.py` as needed.
    *   If you trained a custom tokenizer, make sure `TOKENIZER_SAVE_PATH` points to the correct directory. Otherwise, the script will use the default tokenizer from `model_checkpoint`.

2. **Login to Hugging Face Hub:**

    ```bash
    huggingface-cli login --token $HUGGINGFACE_TOKEN
    ```

3. **Login to WandB (Optional):**

    ```bash
    wandb login --relogin
    ```

4. **Run the Script:**

    ```bash
    python train.py
    ```

This will:

*   Load the dataset.
*   Load the tokenizer (either your custom tokenizer or the default one from the model checkpoint).
*   Load the ModernBERT model.
*   Resize the model's embedding if you are using a custom tokenizer with a different vocabulary size.
*   Fine-tune the model on the dataset using curriculum learning.
*   Evaluate the model periodically during training.
*   Push intermediate and final models to the Hugging Face Hub.

## Monitoring and Evaluation

*   **WandB Dashboard:** If you're using WandB, monitor training progress in real-time on your WandB project dashboard.
*   **Hugging Face Hub:** Your fine-tuned model will be automatically pushed to your Hugging Face Hub profile under the repository name specified in `repo_name`.

## Using Your Fine-tuned Model

After fine-tuning, use your model for downstream tasks with the Transformers library:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "your_username/modernbert-dutch"  # Replace with your model name on the Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: Filling in masked tokens
inputs = tokenizer("Het weer is vandaag [MASK].", return_tensors="pt")
outputs = model(**inputs)
# ... process the outputs ...
```

## Tips and Considerations

*   **GPU Memory:** ModernBERT is large. Adjust `per_device_train_batch_size`, and `gradient_accumulation_steps` to fit your GPU.
*   **Dataset Size:** The script is designed for large, streaming datasets. Adjust `estimated_dataset_size_in_rows` for smaller datasets.
*   **Hyperparameter Tuning:** Experiment with different hyperparameters (learning rate, masking probabilities, etc.) to find optimal settings.
*   **Tokenizer Training:** If training a new tokenizer, consider the `VOCAB_SIZE` and `NUM_EXAMPLES_TO_TRAIN` carefully.
*   **Evaluation:** Customize the evaluation frequency using `eval_interval` in the script.
*   **Saving:** Adjust the saving frequency of intermediate and final models with `push_interval`.

## Troubleshooting

*   **CUDA Errors:** If you get CUDA errors, reduce `per_device_train_batch_size`, or increase `gradient_accumulation_steps`.
*   **Shape Errors:** The `fix_batch_inputs` function and `DynamicPaddingDataCollator` handle most shape issues. If you encounter any, ensure your dataset is properly formatted and you're using the latest `transformers` version.
*   **Tokenizer Issues:** If you have problems loading or using your custom tokenizer, make sure it was saved correctly using `save_pretrained` and that `TOKENIZER_SAVE_PATH` is accurate.
*   **FlashAttention 2 Issues**: Ensure your GPU is compatible (compute capability >= 7.0). If you encounter errors specific to FlashAttention, try disabling it by setting the environment variable `USE_FLASH_ATTENTION` to `False`.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

*   [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
*   [Hugging Face Transformers](https://huggingface.co/transformers)
*   [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index)
*   [Hugging Face Datasets](https://huggingface.co/docs/datasets/index)
*   [Weights & Biases](https://wandb.ai/)
