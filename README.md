
# GRPO-Transformer Text Generation

This repository contains a PyTorch implementation of a decoder-only Transformer model, optimized using Group Relative Policy Optimization (GRPO) for text generation.
The project aims to explore reward-based fine-tuning of language models to encourage specific text characteristics, in this case, "shouting" (using uppercase letters).

Shao, Z. et al. (2024) ‘DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models’, pp. 1–30. Available at: http://arxiv.org/abs/2402.03300.

## Overview

This project consists of a Jupyter Notebook `transformer_grpo.ipynb` that demonstrates the following steps:

1. **Data Preparation:** Loading text data from `input.txt` and tokenizing it using a simple `NaiveTokenizer`.
2. **Transformer Model:** Building a decoder-only Transformer model (`DecoderTrans`) from scratch using PyTorch (inspired from https://www.youtube.com/watch?v=kCc8FmEb1nY)
3. **Baseline Training:** Training the Transformer model using standard cross-entropy loss on the input text data.
4. **Reward Definition:** Defining reward functions to encourage specific text properties (e.g., `reward_shouting` to reward uppercase letters).
5. **GRPO Optimization:** Implementing Group Relative Policy Optimization (GRPO) to fine-tune the pre-trained Transformer model using the defined reward function.
6. **Evaluation:** Comparing the text generation performance of the baseline Transformer and the GRPO-optimized Transformer based on the reward function.

## Files in this Repository

* **`transformer_grpo.ipynb`**: Jupyter Notebook containing the complete implementation of the Transformer model, training, GRPO optimization, and evaluation. 
* **`input.txt`**:  The input text file used for training the language model. You should replace this with your desired dataset.
* **`tokenizer.py`**: Python file containing the `NaiveTokenizer` class.
* **`transformer.py`**: Python file containing the `DecoderTrans` class, implementing the Transformer model (inspired from https://www.youtube.com/watch?v=kCc8FmEb1nY)
* **`README.md`**: This file, providing an overview of the project.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone [repository_url]
   cd [repository_name]
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```

3. **Install required packages:**
   Ensure you have PyTorch installed with CUDA support if you intend to run the notebook on a GPU. Refer to the [PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions based on your system.


## Usage

1. **Run the Jupyter Notebook:**
   Open the notebook in your browser and execute the cells sequentially.
   You might need to change some local configuration, e.g. the GPU discovery by changing CUDA_VISIBLE_DEVICES environment variable

3. **Training and Optimization:**
   The notebook is structured to train a baseline Transformer and then optimize it using GRPO. You can modify the hyperparameters (learning rate, batch size, number of iterations, transformer parameters, GRPO parameters) within the notebook cells.

4. **Model Evaluation and Generation:**
   The notebook includes code to evaluate both the baseline and GRPO-optimized models by generating text and calculating rewards. You can examine the generated text samples and reward scores to observe the effect of GRPO optimization.

## Dataset

The project uses a simple text dataset loaded from `input.txt`. You can replace this file with any text dataset you want to train your language model on. For best results, the dataset should be reasonably large and relevant to the desired text generation task.

## Model Architecture
The repository implements a decoder-only Transformer model (`DecoderTrans`). Key components include:

* **Embedding Layer:** Converts input tokens into dense vector representations.
* **Decoder Blocks:** Stacked layers of:
    * **Multi-Head Self-Attention:** Allows the model to attend to different parts of the input sequence.
    * **Feed-Forward Network:** Processes the attention output.
    * **Layer Normalization:** Stabilizes training.
* **Linear Layer:** Maps the final decoder output to logits for each token in the vocabulary.

## Group Relative Policy Optimization (GRPO)

This project utilizes Group Relative Policy Optimization (GRPO) to fine-tune the Transformer model. GRPO is a reinforcement learning technique that guides the model towards generating text that maximizes a defined reward function. In this implementation, the reward function (`full_reward`) is designed to encourage "shouting" by rewarding the use of uppercase letters.

## Results

By running the notebook, you can observe that the GRPO-optimized Transformer tends to generate text with a higher proportion of uppercase letters compared to the baseline Transformer, as reflected in the reward scores and generated text samples.  Run the notebook to see quantitative results and generated text examples.


## Acknowledgments

* This project is inspired by research in Transformer models and Reinforcement Learning for Language Generation, particularly the DeepseekMath paper mentioned in the notebook comments (Shao, Z. et al. (2024) ‘DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models’, pp. 1–30. Available at: http://arxiv.org/abs/2402.03300.)
* The transformer is inspired from https://www.youtube.com/watch?v=kCc8FmEb1nY
