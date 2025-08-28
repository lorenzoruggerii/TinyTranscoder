from dataclasses import dataclass
import torch
from typing import Callable
import torch.nn as nn
import tiktoken

class TopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        _, indices = torch.topk(x, k=self.k, dim=-1)
        gate = torch.zeros_like(x)
        gate.scatter_(dim=-1, index=indices, value=1)
        return x * gate.to(x.dtype)

@dataclass
class TranscoderConfig:
    """
    Config file for Cross-layer transcoder.

    Args:
        model_path: path to the trained model to use
        num_features: number of interpretable features to extract
        device: device to run the model
        activation_type: type of activation function to use in the transcoder
        topk_features: if using topk activation, specify the number of top features
    """
    save_path: str = r"models/TinyStories/transcoder.pth"
    num_features: int = 8192 # 16x of ffd hidden_size
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    topk_features: int = 16
    activation_fn: Callable = TopK(k=topk_features)
    lr: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 50
    batch_size: int = 64
    num_epochs: int = 100
    num_train_prompts: int = 10_000 # then increase to 200000
    

@dataclass
class ModelConfig:
    """
    Config file for the model and its training.

    Args:
        - dataset_path: path to the HF dataset used during training
        - num_examples: number of dataset examples used
        - train_perc: percentage of training set compared to the whole dataset
        - model_path: where to save the model
        - batch_size: batch size used for training
        - block_size: maximum context length
        - max_iters: number of training iterations
        - eval_interval: number of iters before evaluating the model on validation set
        - learning_rate: learning rate used for training
        - device: device to run the model
        - eval_iters: number of validating iterations
        - n_embd: hidden size of the residual stream
        - n_head: number of heads in MultiHeadAttention layer
        - n_layer: number of Transformer layers
        - dropout: dropout percentage
        - tokenizer: tokenizer used for dataset
        - vocab_size: size of tokenizer vocabulary
    """
    dataset_path: str = "roneneldan/TinyStories"
    num_examples: int = 12_000
    train_perc: float = 0.9
    model_path: str = r"models/TinyStories/gpt_1_50000.pth"
    batch_size: int = 128  # how many independent sequences will we process in parallel?
    block_size: int = 64  # 128 # what is the maximum context length for predictions?
    max_iters: int = 2000  # 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters: int = 200
    n_embd: int = 512  # 512``
    n_head: int = 12
    n_layer: int = 2
    dropout: float = 0.1
    tokenizer: tiktoken.core.Encoding = tiktoken.get_encoding("gpt2")
    vocab_size: int = tokenizer.n_vocab