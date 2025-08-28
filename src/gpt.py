import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from datasets import load_dataset
from dataclasses import dataclass
from config import ModelConfig


def get_batch(split, train_data, val_data, cfg):
    """
    Helper function to load batches of data for training (DataLoader).

    Args:
        - split: str. 'Train' or 'Val'
        - train_data: HF_dataset. Train data split.
        - val_data: HF_dataset. Validation data split.
        - cfg: ModelConfig. Config file for Model. 
    
    Returns:
        - x: torch.Tensor. Model's inputs
        - y: torch.Tensor. Model's labels (next tokens)
    """
    
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i:i+cfg.block_size] for i in ix])
    y = torch.stack([data[i+1:i+cfg.block_size+1] for i in ix])
    x, y = x.to(cfg.device), y.to(cfg.device)
    return x, y


@torch.no_grad()
def estimate_loss(cfg, model, train_data, val_data):
    """
    Helper function to calculate loss.

    Args:
        - cfg : ModelConfig. Config file for model
        - model : GPTLanguageModel. The model being trained
        - train_data : Training data
        - val_data : Validation data

    Returns:
        - out: Dict[str, float]. Average loss for 'train' and 'val' splits.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split, train_data, val_data, cfg)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ Self-contained implementation of an attention head. """

    def __init__(self, head_size, cfg):
        super().__init__()

        # Initialize key, query and value projections
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)

        # Save mask as a buffer (torch does not need to propagate gradients here)
        self.register_buffer('tril', torch.tril(
            torch.ones(cfg.block_size, cfg.block_size)
            )
        )

        # Initialize dropout
        self.dropout = nn.Dropout(cfg.dropout)

        # Add cache to store activations
        self.cache = {}

    def forward(self, x):

        # Input of size (batch, time-step, hidden_dim)
        B, T, C = x.shape

        # Calculate keys (what the token represents) and queries (what the token search for)
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # scale by sqrt(head_dim) to normalize variance

        # Mask previous tokens to get causal attention
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf')
        )

        # Apply softmax to get probabilities
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Compute a weighted mean for the values using attention scores
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        # Store everything in cache
        self.cache["q"] = q.detach()
        self.cache["k"] = k.detach()
        self.cache["v"] = v.detach()
        self.cache["attn_scores"] = wei.detach()
        self.cache["out"] = out.detach()

        return out


class MultiHeadAttention(nn.Module):
    """ Self-contained implementation of Multi-Head Attention """

    def __init__(self, num_heads, head_size, cfg):
        super().__init__()

        # Initialize heads and output projection
        self.heads = nn.ModuleList([Head(head_size, cfg) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

        # Add cache to store activations
        self.cache = {}

    def forward(self, x):
        out_per_head = [h(x) for h in self.heads]
        out = torch.cat(out_per_head, dim=-1)
        out = self.dropout(self.proj(out))

        # Store activations
        self.cache["out"] = out.detach()
        for i, h in enumerate(self.heads):
            self.cache[f"head_{i}"] = h.cache
    
        return out


class FeedFoward(nn.Module):
    """ The FFD layer is composed by an two MLPs and a ReLU """

    def __init__(self, n_embd, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, cfg):
        super().__init__()
        head_size = n_embd // n_head
        
        # Communication layer (mixing information between tokens)
        self.sa = MultiHeadAttention(n_head, head_size, cfg)

        # Computation layer (refining information within tokens)
        self.ffwd = FeedFoward(n_embd, cfg)

        # Layer norms to ensure stability
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # Add cache to store activations
        self.cache = {}

    def forward(self, x):
        pre = x

        # Compute attention and add residual input
        ln1_out = self.ln1(x)
        sa_out = self.sa(ln1_out)
        x = x + sa_out
        mid = x

        # Compute FFD layer and add residual input
        ln2_out = self.ln2(x)
        ffwd_out = self.ffwd(ln2_out)
        x = x + ffwd_out

        # Store activations
        self.cache['pre'] = pre.detach()
        self.cache['ln1'] = ln1_out.detach()
        self.cache['attn'] = sa_out.detach()
        self.cache['mid'] = mid.detach()
        self.cache['ln2'] = ln2_out.detach()
        self.cache['ffwd'] = ffwd_out.detach()
        self.cache['x_out'] = x.detach()

        return x


class GPTLanguageModel(nn.Module):
    """ Minimal implementation of a GPT language model """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Initialize embedding and positional embeddings
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, self.cfg.n_embd)
        self.position_embedding_table = nn.Embedding(self.cfg.block_size, self.cfg.n_embd)

        # Stack N Blocks
        self.blocks = nn.Sequential(
            *[Block(self.cfg.n_embd, n_head=self.cfg.n_head, cfg=cfg) for _ in range(self.cfg.n_layer)])
        
        # Add final layerNorm and Head
        self.ln_f = nn.LayerNorm(self.cfg.n_embd) 
        self.lm_head = nn.Linear(self.cfg.n_embd, cfg.vocab_size)

        # Init Weights
        self.apply(self._init_weights)

    @staticmethod
    def from_pretrained(cfg: ModelConfig):
        """
        Load a GPTLanguageModel pretrained instance.

        Args:
            - cfg: ModelConfig. The config file for model

        Returns:
            - model: GPTLanguageModel. The model 
        """
        weights = torch.load(cfg.model_path)
        model = GPTLanguageModel(cfg).to(cfg.device)
        model.load_state_dict(weights)
        return model

    def run_with_cache(self, idx):
        """
        Run the model and return both the logits and a cache of all intermediate activations.

        Args:
            - idx: torch.Tensor. The tokenized sequences.
        
        Return:
            - logits: torch.Tensor. The output logits from the model
            - cache: Dict[str, torch.Tensor]. The intermediate activations of the model
        """

        # Clear existing caches
        for block in self.blocks:
            block.cache.clear()
            for head in block.sa.heads:
                head.cache.clear()
            block.sa.cache.clear()

        # Standard forward pass
        B, T = idx.shape
        embs = self.token_embedding_table(idx)
        pos_embs = self.position_embedding_table(
            torch.arange(T, device=self.cfg.device)
        )
        x = embs + pos_embs

        # Initialize cache
        cache = {'blocks': []}

        # Put inside embeddings and pos_embs
        cache['embs'] = embs
        cache['pos_embs'] = pos_embs

        # Run through transformer layers
        for i, block in enumerate(self.blocks):
            x = block(x)
            block_cache = {
                'pre': block.cache.get("pre"),
                'ln1': block.cache.get("ln1"),
                'attn': block.sa.cache,
                'mid': block.cache.get("mid"),
                'ln2': block.cache.get("ln2"),
                'ffwd': block.cache.get("ffwd"),
                'x_out': block.cache.get("x_out")
            }
            cache['blocks'].append(block_cache)

        # Compute final layers
        ln_f_out = self.ln_f(x)
        logits = self.lm_head(ln_f_out)

        # Add final acts
        cache['ln_f'] = ln_f_out.detach()
        cache['logits'] = logits.detach()

        return logits, cache


    def _init_weights(self, module):
        """Init weights as original GPT2 paper."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Applies a forward pass of GPT2LanguageModel.

        Args:
            - idx: torch.Tensor. The tokenized input.
            - targets: Optional[torch.Tensor]. The next tokens to predict.

        Returns:
            - logits: torch.Tensor. The output logits of the model.
            - loss: float. The (Cross-entropy) loss of the model.

        """
        B, T = idx.shape

        # Run through every layer of the model
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.cfg.device)
        ) # (T, C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # Calculate loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Consider only the last block_size tokens
            idx_cond = idx[:, -self.cfg.block_size:]
            # Apply forward pass to get predictions
            logits, loss = self(idx_cond)
            # Take prediction for the last token
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append new token index to the indices and repeat
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
    

def train_model(model, cfg, train_data, val_data):

    # Create an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)    

    # Cycle over the training iterations
    for iter in range(cfg.max_iters):

        # Every eval_interval print the loss
        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:

            losses = estimate_loss(cfg, model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', train_data, val_data, cfg)

        # evaluate the loss
        logits, loss = model(xb, yb)

        # Apply AdamW to optimize parameters
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def main():

    torch.manual_seed(1337)

    # Initialize config file
    cfg = ModelConfig()

    # Load dataset and extract num_examples samples
    dataset = load_dataset(cfg.dataset_path)
    text = '\n'.join(dataset['train']['text'][:cfg.num_examples])

    # Divide into train and test splits
    data = torch.tensor(cfg.tokenizer.encode(text), dtype=torch.long)
    n = int(cfg.train_perc*len(data)) 
    train_data = data[:n]
    val_data = data[n:]

    # Initialize model
    model = GPTLanguageModel(cfg)
    m = model.to(cfg.device)

    # Train the model and save it
    train_model(model, cfg, train_data, val_data)
    torch.save(model.state_dict(), cfg.model_path)
    
    # Generate from the model
    model.load_state_dict(torch.load(cfg.model_path))
    context = torch.tensor(
        cfg.tokenizer.encode('\n'), 
        dtype=torch.long,
        device=cfg.device
    ).unsqueeze(0) # Start with a newline

    # Generate 500 tokens and print them
    print(cfg.tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))

    # Try accessing the cache of the model starting from "Once upon a time"
    logits, cache = m.run_with_cache(
        torch.tensor(
            cfg.tokenizer.encode('Once upon a time\n'),
            dtype=torch.long,
            device=cfg.device
        ).unsqueeze(0)
    )
    # Access attention weights of head 3 in block 1
    print(cache["blocks"][1]["attn"]["head_3"]["attn_scores"])


if __name__ == '__main__':
    main()