# Project SecureAi Labs

This project is designed for fine-tuning language models using the Unsloth library with LoRA adapters, and it provides utilities for training, testing, and formatting data for various models like Phi-3, Gemma, and Meta-Llama.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [File Descriptions](#file-descriptions)
   - [TRAINER.ipynb](#traineripynb)
   - [TESTER.ipynb](#testeripynb)
   - [dataFormat.ipynb](#dataformatipynb)
3. [Usage](#usage)
   - [Environment Setup](#environment-setup)
   - [Training a Model](#training-a-model)
   - [Testing the Model](#testing-the-model)
   - [Formatting Data](#formatting-data)
4. [Additional Resources](#additional-resources)

---

## Prerequisites

Before running the project, ensure you have the following:
- A [Hugging Face](https://huggingface.co) account and token.
- Google Colab or a local environment with Python 3.x and CUDA support.
- Installed packages like `unsloth`, `huggingface_hub`, `peft`, `trl`, and others (automatically installed in the notebooks).

NOTE GPU Requirements:

```python
models = [
    'Phi-3.5-mini-instruct-bnb-4bit', # |Min Training Gpu : T4, Min Testing GPU: T4, Max Model size : 14.748 GB|
    'gemma-2-27b-it-bnb-4bit',  # |Min Training Gpu: A100, Min Testing GPU: L4, Max Model size: 39.564GB|
    'Meta-Llama-3.1-8B-Instruct-bnb-4bit' # |Min Training Gpu: T4, Min Testing GPU: T4, Max Model size : 22.168GB|
]
```

Refer to the [Unsloth Documentation](https://unsloth.ai/) for more details.

## File Descriptions

### 1. `TRAINER.ipynb`

This notebook is responsible for training a language model with LoRA adapters using the Unsloth library. The core functionality includes:
- Loading a pre-trained model from Hugging Face using `FastLanguageModel`.
- Attaching LoRA adapters for efficient fine-tuning of large models.
- Setting training configurations (e.g., learning rate, number of epochs, batch size) using the `SFTTrainer` from the `transformers` library.
- Optionally, resuming training from the last checkpoint.
- Uploading checkpoints and models to Hugging Face during or after training.

#### How to Use:
1. Open this notebook in Google Colab or a similar environment.
2. Ensure you have set up your Hugging Face token (refer to the section below for setup).
3. Customize the training parameters if needed.
4. Run the notebook cells to train the model.

### 2. `TESTER.ipynb`

This notebook handles the evaluation of a fine-tuned model. It allows testing the model's accuracy and efficiency on a test dataset using pre-defined metrics like accuracy, precision, recall, and F1 score. It provides the following functionalities:
- Loads the fine-tuned model with its LoRA adapters.
- Defines a function to evaluate the model's predictions on a test dataset.
- Outputs accuracy and other classification metrics.
- Displays confusion matrices for better insight into model performance.

#### How to Use:
1. Load this notebook in your environment.
2. Specify the test dataset and model details.
3. Run the evaluation loop to get accuracy, predictions, and metrics visualizations.

### 3. `dataFormat.ipynb`

This notebook formats datasets into the correct structure for training and testing models. It provides functionality to map raw text data into a format suitable for language model training, particularly for multi-turn conversations:
- Formats conversations into a chat-based template using Unsloth's `chat_templates`.
- Maps data fields like "role", "content", and user/assistant conversations.
- Prepares the dataset for tokenization and input to the model.

#### How to Use:
1. Open the notebook and specify the dataset you wish to format.
2. Adjust any template settings based on the model you're using.
3. Run the notebook to output the formatted dataset.

---

## Usage

### Environment Setup

1. **Install Unsloth**:
   The following command is included in the notebooks to install Unsloth:
   ```bash
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```

2. **Install Additional Dependencies**:
   These dependencies are also required:
   ```bash
   !pip install --no-deps xformers==0.0.27 trl peft accelerate bitsandbytes triton
   ```

3. **Hugging Face Token Setup**:
   - Add your Hugging Face token as an environment variable in Google Colab or in your local environment.
   - Use the Hugging Face token to download models and upload checkpoints:
     ```python
     from google.colab import userdata
     from huggingface_hub import login
     login(userdata.get('TOKEN'))
     ```

### Training a Model

1. Open `TRAINER.ipynb`.
2. Customize the model, template, and LoRA settings in the notebook.
3. Set training configurations (e.g., epochs, learning rate).
4. Run the notebook to start the training process.

The model will automatically be saved at checkpoints and uploaded to Hugging Face.

### Testing the Model

1. Load `TESTER.ipynb` in your environment.
2. Load the fine-tuned model with LoRA adapters.
3. Specify a test dataset in the appropriate format.
4. Run the evaluation function to get predictions, accuracy, and other metrics.

### Formatting Data

1. Use `dataFormat.ipynb` to format raw data into a training-friendly structure.
2. Map the conversation fields using the `formatting_prompts_func`.
3. Output the formatted data and use it in the training or testing notebooks.

---

## Additional Resources

- Unsloth Documentation: [Unsloth.ai](https://unsloth.ai/)
- Hugging Face Security Tokens: [Hugging Face Tokens](https://huggingface.co/docs/hub/en/security-tokens)
- For issues, please refer to each library's official documentation or GitHub pages.

---