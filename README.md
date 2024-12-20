# Fine-tuning ModernBERT on a Dutch Dataset

This repository provides a script and instructions for fine-tuning the [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) model on a Dutch dataset (or any other dataset from the Hugging Face Hub). It leverages the Hugging Face Transformers library and incorporates techniques like curriculum learning and gradient accumulation for efficient training.

## Features

*   **Stable Data Collator:** Ensures correct input tensor shapes and dtypes (`input_ids` as `torch.long`) to prevent training errors.
*   **Automatic Mixed Precision (AMP):** Uses a mix of FP16 and FP32 operations to reduce memory usage and speed up training.
*   **Curriculum Learning:** Gradually decreases the masking probability during training, starting from 0.3 and going down to 0.14.
*   **Gradient Accumulation:** Simulates larger batch sizes by accumulating gradients over multiple steps.
*   **Hugging Face Transformers Integration:** Leverages `AutoTokenizer` and `AutoModelForMaskedLM` for easy loading of pre-trained models and configurations.
*   **Hugging Face Hub Integration:** Automatically pushes intermediate and final fine-tuned models to the Hugging Face Hub.
*   **Weights & Biases (WandB) Support (Optional):** Tracks and visualizes training runs in real-time.

## Prerequisites

*   **Hugging Face Account:** You need a Hugging Face account. Sign up [here](https://huggingface.co/join).
*   **Hugging Face API Token:** Generate a User Access Token (with "write" access) from your [Hugging Face profile settings](https://huggingface.co/settings/tokens).
*   **WandB Account (Optional):** Create a free account at [wandb.ai](https://wandb.ai/).
*   **WandB API Key:** Get your API key from your [WandB settings](https://wandb.ai/settings).
*   **Environment:** A GPU environment is strongly recommended. This was developed and tested with the latest pytorch version.

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

1. **Environment Variables:**

    Set the following environment variables:

    ```bash
    export HUGGINGFACE_TOKEN="your_huggingface_token"
    export WANDB_API_KEY="your_wandb_api_key"  # Optional
    ```

    Replace `"your_huggingface_token"` with your actual Hugging Face token and `"your_wandb_api_key"` with your WandB API key.

2. **Script Parameters:**

    The `train.py` script defines several configurable parameters. You can modify these directly in the file or override them using environment variables:

    | Parameter                       | Default Value                 | Description                                                                                                                                |
    | :------------------------------ | :---------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
    | `model_checkpoint`              | "answerdotai/ModernBERT-base" | The base pre-trained ModernBERT model to use.                                                                                             |
    | `dataset_name`                  | "ssmits/fineweb-2-dutch"      | The name of the dataset on the Hugging Face Hub to use for fine-tuning.                                                                   |
    | `num_train_epochs`              | 1                             | The number of training epochs.                                                                                                             |
    | `chunk_size`                    | 8192                          | The maximum sequence length for input sequences. Adjust based on your GPU memory.                                                            |
    | `gradient_accumulation_steps`   | 32                            | The number of steps to accumulate gradients over before performing an optimizer step. Modify based on desired effective batch size and GPU memory. |
    | `per_device_train_batch_size`    | 1                             | The batch size per GPU. Adjust based on your GPU memory.                                                                                    |
    | `eval_size_ratio`               | 0.05                          | The proportion of the dataset to use for evaluation.                                                                                       |
    | `masking_probabilities`         | \[0.3, 0.2, 0.18, 0.16, 0.14] | The curriculum learning masking probabilities.                                                                                             |
    | `estimated_dataset_size_in_rows` | 86500000                     | The estimated number of rows in your dataset.                                                                                            |
    | `username` | username | Your huggingface username                                                                                         |
    | `total_save_limit` | 2 | Max number of saved models|
    | `repo_name` | modernbert-dutch | Repo name to save to |
    | `eval_size_per_chunk` | 50000 | Eval size for every chunk in curriculum learning |
    | `learning_rate` | 1e-3 | Learning rate |
    | `weight_decay` | 0.01 | Weight decay |
    | `total_steps_per_epoch` | estimated_dataset_size_in_rows // (per_device_train_batch_size * gradient_accumulation_steps)| Total training steps per epoch|

## Running the Fine-tuning Script

1. **Login to Hugging Face Hub:**

    ```bash
    huggingface-cli login --token $HUGGINGFACE_TOKEN
    ```

2. **Login to WandB (Optional):**

    ```bash
    wandb login --relogin
    ```

3. **Run the Script:**

    ```bash
    python train.py
    ```

## Monitoring and Evaluation

*   **WandB Dashboard:** If you're using WandB, monitor training progress in real-time on your WandB project dashboard.
*   **Hugging Face Hub:** Your fine-tuned model will be automatically pushed to your Hugging Face Hub profile under the repository name specified in `repo_name`.

## Using Your Fine-tuned Model

After fine-tuning, use your model for downstream tasks with the Transformers library:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "your_username/modernbert-base-dutch"  # Replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: Filling in masked tokens
inputs = tokenizer("Het weer is vandaag [MASK].", return_tensors="pt")
outputs = model(**inputs)
# ... process the outputs ...
```

## Tips and Considerations

*   **GPU Memory:** ModernBERT is large. Adjust `chunk_size`, `per_device_train_batch_size`, and `gradient_accumulation_steps` to fit your GPU.
*   **Dataset Size:** The script is designed for large, streaming datasets. Adjust `estimated_dataset_size_in_rows` for smaller datasets.
*   **Hyperparameter Tuning:** Experiment with different hyperparameters (learning rate, masking probabilities, etc.) to find optimal settings.
*   **Evaluation:** Customize the evaluation frequency using `eval_interval` in the script.
*   **Saving:** Adjust the saving frequency of intermediate and final models with `save_interval`.

## Troubleshooting

*   **CUDA Errors:** If you get CUDA errors, reduce `per_device_train_batch_size`, `chunk_size`, or increase `gradient_accumulation_steps`.
*   **Shape Errors:** The `StableDataCollator` handles most shape issues. If you encounter any, ensure your dataset is properly formatted and you're using the latest `transformers` version.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

*   [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
*   [Hugging Face Transformers](https://huggingface.co/transformers)
*   [Weights & Biases](https://wandb.ai/)
