""" Adapted from https://github.com/etredal/openCLT"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt import GPTLanguageModel
from transformers import GPT2Tokenizer
import numpy as np
from config import TranscoderConfig, ModelConfig
from typing import Dict, Optional, List
from tqdm import tqdm
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class CrossLayerTranscoder(nn.Module):

    def __init__(self, t_cfg: TranscoderConfig, model_cfg: ModelConfig):
        """
        Initialize the cross-layer transcoder

        Args:
            cfg: Config file for Cross-layer Transcoder
        """
        super().__init__()
        self.cfg = t_cfg
        self.model_cfg = model_cfg

        # Load the model
        self.base_model = GPTLanguageModel.from_pretrained(model_cfg)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # use gpt2 tokenizer

        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Get models dimensions
        self.num_layers = self.model_cfg.n_layer
        self.hidden_size = self.model_cfg.n_embd

        # Define model_name for saving
        self.model_name = f"TC_{self.num_layers}l_{self.hidden_size}H"

        # Create encoder for each layer
        self.encoders = nn.ModuleList([
            nn.Linear(self.hidden_size, t_cfg.num_features)
            for _ in range(self.num_layers)
        ]).to(t_cfg.device)

        # Extract activation function
        self.activation_functions = nn.ModuleList([
            t_cfg.activation_fn
            for _ in range(self.num_layers)
        ]).to(t_cfg.device)

        # Create decoder for each layer
        self.decoders = nn.ModuleList([
            nn.Linear(t_cfg.num_features, self.hidden_size)
            for _ in range(self.num_layers)
        ]).to(t_cfg.device)

        self._initialize_weights()

        # Initialize hooks and storage for activations
        self.hooks = []
        self.mlp_activations = {}
        self.mlp_inputs_captured = {}
        self._register_hooks()

        # Feature importance tracking
        # For each feature see where it is popping out
        self.feature_importance = torch.zeros(self.num_layers, t_cfg.num_features).to(t_cfg.device)

    def _initialize_weights(self):

        for encoder_layer in range(self.num_layers):
            # Initialize encoder weights
            std_encoder = 1.0 / np.sqrt(self.cfg.num_features) # as in paper
            nn.init.uniform_(self.encoders[encoder_layer].weight, -std_encoder, std_encoder)
            nn.init.zeros_(self.encoders[encoder_layer].bias)

        for decoder_layer in range(self.num_layers):
            # Initialize decoder weights
            std_decoder = 1.0 / np.sqrt(self.num_layers * self.hidden_size) # as in paper
            nn.init.uniform_(self.decoders[decoder_layer].weight, -std_decoder, std_decoder)
            nn.init.zeros_(self.decoders[decoder_layer].bias)

    def print_activations(self, layer_idx, features_relu, num_features):
        if features_relu.numel() > 0: # Check if tensor is not empty
            print(f"\n--- Debug Sample: Layer {layer_idx} Feature Activations (features_relu) ---")
            print(f"Shape of features_relu: {features_relu.shape} (Batch, SeqLen, NumFeatures)")
            
            # Print all feature activations for the first token of the first item in the batch
            if features_relu.shape[0] > 0 and features_relu.shape[1] > 0:
                print(f"Activations for B[0], T[0] (all {num_features} features):")
                print(features_relu[0, 0, :])
            
            print(f"--- End Debug Sample ---")
            
    def _register_hooks(self):
        """Register forward hooks to capture MLP activations from each layer."""
        def hook_fn(layer_idx):
            def hook(module, input_args, output_tensor):
                self.mlp_inputs_captured[layer_idx] = input_args[0]
                self.mlp_activations[layer_idx] = output_tensor
                return output_tensor
            return hook
        
        # Remove any existing hook
        self.remove_hooks()

        # Register new hooks for each MLP layer
        for i in range(self.num_layers):
            mlp = self.base_model.blocks[i].ffwd.net
            hook = mlp.register_forward_hook(hook_fn(i))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all existing hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model and cross-layer transcoder.

        Args: 
            input_ids: Input token IDs
            attention_mask: Attention mask for padding

        Returns:
            Dictionary containing:
                - 'last_hidden_state': The base model's output
                - 'feature_activations': Feature activations for each layer
                - 'reconstructed_activations': Reconstructed MLP activations
        """

        # Clear all previous activations
        self.mlp_activations.clear()
        self.mlp_inputs_captured.clear()

        # Forward pass through base model to collect MLP activations via hooks
        with torch.no_grad():
            logits, _ = self.base_model(input_ids)
        # Here self.mlp_inputs_captured and self.mlp_activations gets filled

        # Process each layer's activations through the transcoder
        feature_activations = {} # save activations of CLT
        reconstructed_activations = {}

        for layer_idx in range(self.num_layers):
            if layer_idx in self.mlp_inputs_captured:
                # MLP input
                mlp_input = self.mlp_inputs_captured[layer_idx] # this is what gets read
                mlp_output = self.mlp_activations[layer_idx] # this is what it gets written

                mlp_input_normalized = F.layer_norm(mlp_input, (mlp_input.shape[-1],))
                mlp_output_normalized = F.layer_norm(mlp_output, (mlp_output.shape[-1],))

                # Encode to features
                features = self.encoders[layer_idx](mlp_input_normalized)
                features_activated = self.activation_functions[layer_idx](features)
                feature_activations[layer_idx] = features_activated

                # Reconstruct activations using current layer decoder
                reconstructed = self.decoders[layer_idx](feature_activations[layer_idx])
                reconstructed_activations[layer_idx] = reconstructed

                # Update feature importance based on activation magnitude
                with torch.no_grad():
                    importance = torch.mean(torch.abs(features_activated), dim=(0, 1))
                    self.feature_importance[layer_idx] += importance

        return {
            'last_hidden_state': logits,
            'feature_activations': feature_activations,
            'reconstruced_activations': reconstructed_activations
        }
    
    def train_transcoder(self,
                         texts: List[str],
                         ) -> Dict[str, List[float]]:
        """
        Train the cross layer transcoder on a corpus of text.
        """

        # Set best loss to a high number
        best_loss = 10_000

        # Set model to training mode
        self.train()

        all_params = (list(self.encoders.parameters())
                      + list(self.decoders.parameters())
                      + list(self.activation_functions.parameters()))
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            all_params, lr=self.cfg.lr
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', # we want to minimize the loss
            factor = self.cfg.lr_scheduler_factor,
            patience = self.cfg.lr_scheduler_patience,
        )

        # Training metrics
        metrics = {
            'total_loss': [],
            'reconstruction_loss': [],
            'sparsity_loss': [],
            'l0_metric': [],
            'learning_rate': []
        }

        # Tokenize all texts
        print("Tokenizing examples...")
        encoded_texts = [self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length).to(self.cfg.device) for text in texts]
        print("End of tokenization!")

        # Training loop
        for epoch in range(self.cfg.num_epochs):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_sparsity_loss = 0
            epoch_l0_metric = 0
            
            # Process in batches
            num_batches = (len(encoded_texts) + self.cfg.batch_size - 1) // self.cfg.batch_size

            for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.cfg.num_epochs}"):
                # Get batch
                start_idx = batch_idx * self.cfg.batch_size
                end_idx = min(start_idx + self.cfg.batch_size, len(encoded_texts))
                batch_texts = encoded_texts[start_idx:end_idx]

                # Pad to same length within batch
                # Truncate to block_size and pad to same length within batch
                max_len = min(max(text.size(1) for text in batch_texts), self.base_model.cfg.block_size)  # Cap at block_size
                padded_texts = []

                for text in batch_texts:
                    # Truncate if longer than block_size
                    if text.size(1) > self.base_model.cfg.block_size:
                        text = text[:, :self.base_model.cfg.block_size]
                    
                    pad_len = max_len - text.size(1)
                    padded_text = F.pad(text, (0, pad_len), value=self.tokenizer.pad_token_id)
                    padded_texts.append(padded_text)

                input_ids = torch.cat(padded_texts, dim=0)
                
                # Clear gradients
                optimizer.zero_grad()

                self.mlp_inputs_captured.clear()
                self.mlp_activations.clear()

                # Forward pass to collect MLP activations
                self.base_model(input_ids)

                # Compute loss for each layer
                total_loss = 0
                reconstruction_loss = 0
                sparsity_loss = 0

                all_features = {}  # Store features for reconstruction

                for layer_idx in range(self.num_layers):
                    if layer_idx in self.mlp_inputs_captured:
                        mlp_input = self.mlp_inputs_captured[layer_idx]
                        mlp_output = self.mlp_activations[layer_idx]

                        # Normalize input to MLP
                        mlp_input_normalized = F.layer_norm(mlp_input, mlp_input.shape[-1:])  # Normalize input to MLP
                        mlp_output_normalized = F.layer_norm(mlp_output, mlp_output.shape[-1:])  # Normalize output of MLP

                        # Encode to features
                        features = self.encoders[layer_idx](mlp_input_normalized)
                        
                        # Apply activation function
                        features_activated = self.activation_functions[layer_idx](features)

                        all_features[layer_idx] = features_activated

                        # Reconstruct MLP outputs
                        reconstructed = self.decoders[layer_idx](features_activated)

                        # L0 metric (sparsity)
                        l0_metric = torch.mean((features_activated > 1e-6).float())

                        # Reconstruction loss (MSE)
                        recon_loss = F.mse_loss(reconstructed, mlp_output_normalized)
                        reconstruction_loss += recon_loss

                        # L1 loss (sparsity of activations)
                        l1_loss = 0 # because using topk, k is fixed

                        # Update total loss
                        total_loss += l1_loss + recon_loss

                # Backward pass and optimization
                total_loss.backward()

                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

                # Update metrics
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += reconstruction_loss.item()
                if isinstance(sparsity_loss, torch.Tensor):
                    epoch_sparsity_loss += sparsity_loss.item()
                else:
                    epoch_sparsity_loss += sparsity_loss
                epoch_l0_metric += l0_metric.item()

            # Record epoch metrics
            avg_total_loss = epoch_total_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_sparsity_loss = epoch_sparsity_loss / num_batches
            avg_l0_metric = epoch_l0_metric / num_batches
            
            metrics['total_loss'].append(avg_total_loss)
            metrics['reconstruction_loss'].append(avg_recon_loss)
            metrics['sparsity_loss'].append(avg_sparsity_loss)
            metrics['l0_metric'].append(avg_l0_metric)
            metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])

             # Step the scheduler
            scheduler.step(avg_total_loss) # Step with the monitored metric
            
            print(f"Epoch {epoch+1}/{self.cfg.num_epochs}: "
                  f"Loss = {avg_total_loss:.4f}, "
                  f"Recon = {avg_recon_loss:.4f}, "
                  f"Sparsity = {avg_sparsity_loss:.4f}, "
                  f"L0 Metric = {avg_l0_metric:.4f}")
            
            # Save model if best one
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                self.save_model()
        
            
        return metrics



    def get_feature_activations(self, text: str) -> Dict[int, torch.Tensor]:
        """
        Get feature activations for all layers on a given text

        Args: 
            text: Input text

        Returns:
            Dictionary mapping layer indices to feature activations
        """

        # Set to evaluation mode
        self.eval()

        # Tokenize input
        input_ids = torch.tensor(self.tokenizer.encode(text)).to(self.cfg.device)

        # Forward pass
        with torch.no_grad():
            outputs = self(input_ids)

        return outputs['feature_activations']
    
    def get_top_features(self, n: int = 10) -> Dict[int, List[int]]:
        """
        Get the indices of the top N most important features for each layer.

        Args:
            n: Number of top features to return
        
        Returns:
            Dictionary mapping layer indices to lists of top feature indices
        """

        top_features = {}

        for layer_idx in range(self.num_layers):
            # Get importance scores for current layer
            importance = self.feature_importance[layer_idx]

            # get indices of top N features
            _, indices = torch.topk(importance, min(n, self.cfg.num_features))
            top_features[layer_idx] = indices.cpu().numpy().tolist()

        return top_features

    def save_model(self):
        """Save the cross-layer transcoder model."""
        torch.save({
            'encoders': self.encoders.state_dict(),
            'decoders': self.decoders.state_dict(),
            'feature_importance': self.feature_importance,
            'config': {
                'model_name': self.model_name,
                'num_features': self.cfg.num_features,
                'num_layers': self.num_layers,
                'hidden_size': self.hidden_size
            }
        }, self.cfg.save_path)

    @classmethod
    def load_model(cls, t_cfg, m_cfg):
        """Load a saved cross-layer transcoder model."""
        checkpoint = torch.load(t_cfg.save_path, map_location=t_cfg.device)
        
        # Create model with saved config
        model = cls(
            t_cfg,
            m_cfg
        )
        
        # Load state dictionaries
        model.encoders.load_state_dict(checkpoint['encoders'])
        model.decoders.load_state_dict(checkpoint['decoders'])
        model.feature_importance = checkpoint['feature_importance']
        
        return model
    
    
if __name__ == '__main__':

    m_cfg = ModelConfig()
    t_cfg = TranscoderConfig()

    tc = CrossLayerTranscoder(t_cfg, m_cfg)

    # Load dataset
    dataset = load_dataset('roneneldan/TinyStories', split='train')
    print(t_cfg.num_train_prompts)
    dataset = dataset[:t_cfg.num_train_prompts]['text']

    # Start training
    tc.train_transcoder(dataset)

    