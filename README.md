# TinyTranscoder: Cross-Layer Transcoder for Language Models

TinyTranscoder provides a minimal, easy-to-use implementation for training a two-layer GPT-style model on the TinyStories dataset. It then shows how to train two transcoders on the model's internal representations. This approach find and interpret the features the model uses to process information.

The core idea is to train a separate, smaller model (the transcoder) to map the hidden states of a transformer's MLP layers to a set of sparse, human-interpretable features. By doing this, we can trace the flow of information through the model constructing attribution graphs and see how specific learned features contribute to the final output.

I've made an article on circuit tracing and transcoders on my personal blog. Read it [here](https://lorenzoruggerii.github.io/blog/2025/transcoders-new/)

---

## ⚙️ Installation

To get started, you'll need to install the dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 🚀 Usage

The project has two main stages: training the base GPT model and then training the transcoders. All scripts are located in the `src/` directory.

### 1. Train the GPT Model

First, you need a trained language model. The `src/gpt.py` script handles this. It will download the TinyStories dataset, train a small two-layer GPT model, and save its weights.

```bash
python src/gpt.py
```

This will save the trained model weights to `models/TinyStories/gpt_1_50000.pth` (you can change this from `config.py`).

### 2. Train the Transcoders

Once the GPT model is trained, you can train two transcoders on its intermediate activations. The `src/transcoder.py` script performs this task.

```bash
python src/transcoders.py
```

This will save the transcoders model weights to `models/TinyStories/transcoder.pth` (you can change this from `config.py`).

### 3. Circuit Tracing and Analysis

The `src/circuit_tracing.py` module contains functions for analyzing the trained models. It includes tools to trace how information flows through the model's layers and to determine the contribution of specific transcoder features.
This was adapted from [here](https://github.com/jacobdunefsky/transcoder_circuits)

For an interactive analysis and visualization of the features, you can use the provided Jupyter notebook `src/interpretability.py`.

This notebook contains visualization frameworks developed in Dash to explore which features are active on specific input tokens.

## 📁 Repository Structure

-   **`src/`**: The main project directory containing all Python scripts.
    -   **`__init__.py`**: An empty file that makes the directory a Python package.
    -   **`circuit_tracing.py`**: Contains the core logic for mechanistic interpretability and circuit tracing. It defines data structures like `Component` and `FeatureVector` to represent activation paths and provides functions for attribution and path-finding.
    -   **`config.py`**: Defines configuration classes (`TranscoderConfig` and `ModelConfig`) for the transcoder and the GPT model, including hyperparameters, file paths, and model architecture details.
    -   **`gpt.py`**: Implements the GPT-style language model from scratch, including training and evaluation loops. This script is used to generate the base model on which the transcoder operates.
    -   **`interpretability.ipynb`**: An interactive Jupyter notebook for visualizing and exploring the transcoder features.
    -   **`transcoder.py`**: Implements the transcoder model, which learns to encode and decode the MLP activations of the GPT model into sparse features. It handles the training process for the transcoder itself.
    -   **`utils.py`**: Contains utility functions that support the main modules (e.g., data loading).
-   **`requirements.txt`**: A list of all Python dependencies required for the project.
