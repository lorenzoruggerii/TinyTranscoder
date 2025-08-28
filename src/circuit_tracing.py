"""Adapted from https://arxiv.org/abs/2406.11944"""

import torch
import torch.nn as nn
from torch import einsum
from typing import Optional, Union, List
from functools import partial
import numpy as np
import copy

@torch.no_grad()
def get_attn_head_contribs(model, cache, layer, feature_vec):
    """
    Attribution from an attention head via the OV circuit to a target feature vector.
    Args:
        model: Transformer model
        cache: Cached activations
        layer: Transcoder's layer of the features
        feature_vec: torch.Tensor
    Returns:
        contribs: Tensor of shape [batch, head, dst_token, src_token] representing contribution of every source token to feature vec
    """
    #
    if isinstance(feature_vec, FeatureVector):
        feature_vec = feature_vec.vector

    # Pass feature_vec to device
    feature_vec = feature_vec.to(model.cfg.device)
    
    v = [cache["blocks"][layer]["attn"][f"head_{h}"]["v"] for h in range(model.cfg.n_head)]
    v = torch.stack(v, dim=2) # [b, src, head, d_head]

    pattern = [cache["blocks"][layer]["attn"][f"head_{h}"]["attn_scores"] for h in range(model.cfg.n_head)]
    pattern = torch.stack(pattern, dim=1) # [b, head, dest, src]

    W_O = model.blocks[layer].sa.proj.weight
    W_O = W_O.t().view(model.cfg.n_head, model.cfg.n_embd // model.cfg.n_head, -1) # [head, d_head, d_model]

    # Weighted values
    weighted_vals = einsum(
        'b h d s, b s h f -> b h d s f',
        pattern, v
    )

    # OV projection
    weighted_outs = einsum(
        'b h d s f, h f m -> b h d s m',
        weighted_vals, W_O
    )

    # Project onto feature vector
    contribs = einsum(
        'b h d s m, m -> b h d s',
        weighted_outs, feature_vec 
    )
    
    return contribs

@torch.no_grad()
def get_transcoder_ixg(transcoder, cache, feature_vec, layer, token_idx, post_ln=True, return_feature_activs=True):
    """
    Attribution from transcoder features to a final feature vector via decoder encoder.
    Implement Equation (18) in the paper.
    Returns the attribution of feature i in transcoder l on token token_idx.
    Args:
        transcoder: Module with W_dec
        cache: model cache
        feature_vec: target vector f' (e.g. from W_enc[:, i'])
        layer: input layer index
        token_idx: analyzed tokens
        post_ln: if True, use normalized input to transcoder (post LayerNorm)
    Returns:
        ixg: input-times-gradient vector in residual stream space
    """

    # Export feature vec to the same device of transcoder
    feature_vec = feature_vec.to(transcoder.cfg.device)

    pulledback_feature = transcoder.decoders[layer].weight.T @ feature_vec # [num_features]

    resid = (
        cache["blocks"][layer]["ln2"]
        if post_ln
        else cache["blocks"][layer]["attn"]["out"]
    )

    # Extract features from transcoder
    feature_activs = transcoder.encoders[layer](resid)[0, token_idx] # [num_features]

    # Pulled-back features
    pulledback_feature = pulledback_feature * feature_activs

    if not return_feature_activs:
        return pulledback_feature
    
    return pulledback_feature, feature_activs


@torch.no_grad()
def get_ln_constant(cache, vector, layer, token, ln2=False, reciprocal=False):
    """
    Approximates LayerNorm as a constant multiplier along direction 'vector'.
    Args:
        vector: feature we're analyzing
        ln2: if True, use post-MLP (ln2), else pre-attn (ln1)
    Returns:
        scalar constant
    """
    x = cache["blocks"][layer]['attn']['out'] if ln2 else cache["blocks"][layer]['pre']
    x = x[0, token]

    y = cache["blocks"][layer]['ln2'] if ln2 else cache["blocks"][layer]['ln1']
    y = y[0, token]

    dot_x = torch.dot(vector, x)
    dot_y = torch.dot(vector, y)

    if dot_x == 0:
        return torch.tensor(0.0)
    return dot_y / dot_x if not reciprocal else dot_x / dot_y

@torch.no_grad()
def get_mean_ixg(tokens, transcoder, layer, feature_idx, batch_size=32, token_positions=None):
    """
    Compute mean input-times-gradient vector for a transcoder feature across dataset.
    Args: 
        model: Transformer model
        tokens: [N, seq_len] tokenized input
        transcoder: module with W_dec, W_enc
        feature_idx: index of target transcoder feature
    Returns:
        Mean input-times-gradient vector [d_model]
    """
    # Put tokens on same device as transcoder
    tokens = tokens.to(transcoder.cfg.device)

    W_enc_row = transcoder.encoders[layer].weight[feature_idx, :].squeeze()
    pulledback = transcoder.decoders[layer].weight.T @ W_enc_row # [num_features]

    all_ixgs = []

    indices = token_positions if token_positions else list(range(0, tokens.shape[0], batch_size))

    for i in indices:
        if token_positions:
            ex_idx, tok_idx = 1
            toks = tokens[ex_idx, :tok_idx+1].unsqueeze(0)
            token_idx = tok_idx
        else:
            toks = tokens[i:i+batch_size]
            token_idx = -1 # full sequence, use all
        
        # Run transcoder on inputs
        outs = transcoder(tokens)['feature_activations'] # [B, T, num_feats]

        # Select features from transcoder
        feats = outs[layer]
        print(feats)
        # feats = all_feats[layer][:, oken_idx] if token_idx != -1 else all_feats[layer].reshape(-1, all_feats[layer].shape[-1])
        # feats is [B, num_features]

        # einsum to apply pulledback vector across batch
        cur_ixg = pulledback * feats # [B, num_feats]
        all_ixgs.append(torch.mean(cur_ixg, dim=0))

    return torch.mean(torch.stack(all_ixgs), dim=0)

import enum
from dataclasses import dataclass

class ComponentType(enum.Enum):
    MLP = 'mlp'
    ATTN = 'attn'
    EMBED = 'embed'
    TC_ERROR = 'tc_error'
    PRUNE_ERROR = 'prune_error'
    BIAS_ERROR = 'bias_error'

class FeatureType(enum.Enum):
    NONE = 'none'
    SAE = 'sae'
    TRANSCODER = 'tc'

class ContribType(enum.Enum):
    RAW = 'raw'
    ZERO_ABLATION = 'zero_ablation'

@dataclass
class Component:
    layer: int
    component_type: ComponentType
    token: Optional[int] = None
    attn_head: Optional[int] = None
    feature_type: Optional[FeatureType] = None
    feature_idx: Optional[int] = None

    def __str__(self, show_token=True):
        retstr = ''
        feature_type_str = ''

        base_str = f'{self.component_type.value}{self.layer}'
        attn_str = '' if self.component_type != ComponentType.ATTN else f'[{self.attn_head}]'
        
        feature_str = ''
        if self.feature_type is not None and self.feature_idx is not None:
            feature_str = f"{self.feature_type.value}[{self.feature_idx}]"
            
        token_str = ''
        if self.token is not None and show_token:
            token_str = f'@{self.token}'

        retstr = ''.join([base_str, attn_str, feature_str, token_str])
        return retstr

    def __repr__(self):
        return f'<Component object {str(self)}>'

    # Example usage: Component(layer=3, component_type=ComponentType.ATTN, token=12, attn_head=5)

# Help track causal chains: which MLP or attn_heads caused the vector to arise?
@dataclass
class FeatureVector:
    component_path: List[int]
    vector: torch.Tensor
    layer: int
    sublayer: str # e.g. resid_pre
    token: Optional[int] = None
    contrib: Optional[float] = None
    contrib_type: Optional[ContribType] = None
    error: float = 0.0

    def __str__(self, show_full=True, show_contrib=True, show_last_token=True):
        retstr = ''
        token_str = '' if self.token is None or not show_last_token else f'@{self.token}'
        if len(self.component_path) > 0:
            if show_full:
                retstr = ''.join(x.__str__(show_token=True) for x in self.component_path[:-1])
            retstr = ''.join([retstr, self.component_path[-1].__str__(show_token=True), token_str])
        else:
            retstr = f'*{self.sublayer}{self.layer}{token_str}'
        if show_contrib and self.contrib is not None:
            retstr = ''.join([retstr, f': {self.contrib:.2}'])
        return retstr

    def __repr__(self):
        contrib_type_str = '' if self.contrib_type is None else f' contrib_type={self.contrib_type.value}'
        return f'<FeatureVector object {str(self)}, sublayer={self.sublayer}{contrib_type_str}>'

@torch.no_grad()
def get_top_transcoder_features(model, transcoder, cache, feature_vector, layer, k=5):

    # Move feature vector to device
    feature_vector.vector = feature_vector.vector.to(transcoder.cfg.device)

    # Determine input token
    my_token = feature_vector.token if feature_vector.token >= 0 else cache['blocks'][0]['ln1'].shape[1] + feature_vector.token

    # For now take the normalized values (after LayerNorm)
    activations = cache['blocks'][layer]['ln1']

    # Run transcoder on inputs
    features_acts = transcoder.encoders[layer](activations)
    transcoder_out = transcoder.decoders[layer](features_acts)[0, my_token]
    mlp_out = cache['blocks'][layer]['ffwd'][0, my_token]

    # Compute error
    error = torch.dot(feature_vector.vector, mlp_out - transcoder_out) / torch.dot(feature_vector.vector, mlp_out)

    # Compute pulledback features
    pulledback_features, feature_actives = get_transcoder_ixg(transcoder, cache, feature_vector.vector, layer, feature_vector.token)

    # Take top k contributors
    top_contribs, top_idxs = torch.topk(pulledback_features, k=k)

    # Rebuild now feature vector objects for each top contributor
    top_contribs_list = []

    for contrib, index in zip(top_contribs, top_idxs):
        vector = transcoder.encoders[layer].weight[index, :]
        vector = vector * (transcoder.decoders[layer].weight.T @ feature_vector.vector)[index]

        new_component = Component(
            layer=layer,
            component_type=ComponentType.MLP,
            token=my_token,
            feature_type=FeatureType.TRANSCODER,
            feature_idx=index.item(),
        )    

        top_contribs_list.append(FeatureVector(
            component_path=[new_component],
            vector = vector,
            layer=layer,
            sublayer="resid_mid",
            contrib=contrib.item(),
            contrib_type=ContribType.RAW,
            error=error,
        ))

        return top_contribs_list
    

@torch.no_grad()
def get_OV_matrix(model, layer, head):
    W_V = model.blocks[layer].sa.heads[head].value.weight

    # Calculate dim_head to extract head's WO
    dim_head = model.cfg.n_embd // model.cfg.n_head
    W_O = model.blocks[layer].sa.proj.weight[:, (head*dim_head):((head+1)*dim_head)]

    return W_O @ W_V

   
@torch.no_grad()
def get_top_contribs(model, transcoder, cache, feature_vector, k=5):

    # Put feature vector on device
    feature_vector.vector = feature_vector.vector.to(transcoder.cfg.device)

    if feature_vector.sublayer == "mlp_out":
        return get_top_transcoder_features(model, transcoder, cache, feature_vector, feature_vector.layer, k=k)
    
    my_layer = feature_vector.layer
    all_contribs = []

    # Because of linearity you can sum the contributions together
    # MLPs
    for l in range(feature_vector.layer):
        all_contribs += get_top_transcoder_features(model, transcoder, cache, feature_vector, l, k=k)

    # Attention
    for l in range(feature_vector.layer):
        attn_contribs = get_attn_head_contribs(model, cache, l, feature_vector)[0, :, feature_vector.token, :]
        topk_vals, topk_idxs = torch.topk(attn_contribs.flatten(), k=min(k, attn_contribs.numel()))
        for val, flat_idx in zip(topk_vals, topk_idxs):
            head, src = np.unravel_index(flat_idx.item(), attn_contribs.shape)

            # Get OV matrix
            OV = get_OV_matrix(model, l, head)
            vector = OV @ feature_vector.vector

            # Multiply by attention score
            attn_pattern = cache['blocks'][l]['attn'][f'head_{head}']['attn_scores']
            vector *= attn_pattern[0, feature_vector.token, src]

            all_contribs.append(FeatureVector(
                component_path = feature_vector.component_path + [Component(layer=l, component_type=ComponentType.ATTN, token=src, attn_head=head)],
                vector=vector,
                layer=l,
                sublayer='resid_pre',
                contrib=val.item(),
                contrib_type=ContribType.RAW
            ))

    # Embedding part
    emb_vec = cache['embs'][0, feature_vector.token]
    emb_score = torch.dot(emb_vec, feature_vector.vector).item()

    all_contribs.append(FeatureVector(
        component_path = feature_vector.component_path + [Component(layer=0, component_type=ComponentType.EMBED, token=feature_vector.token)],
        vector = feature_vector.vector,
        layer = 0,
        sublayer = 'resid_pre',
        contrib = emb_score,
        contrib_type = ContribType.RAW
    ))

    # Return top-k
    top_vals, top_idxs = torch.topk(torch.tensor([x.contrib for x in all_contribs]), k=min(k, len(all_contribs)))
    return [all_contribs[i] for i in top_idxs]

@torch.no_grad()
def greedy_get_top_paths(model, transcoder, cache, feature_vector, num_iters=2, num_branches=5):

    # Put feature vector into device
    feature_vector.vector = feature_vector.vector.to(transcoder.cfg.device)

    all_paths = []
    root = copy.deepcopy(feature_vector)
    cur_paths = [[root]]

    for _ in range(num_iters):
        new_paths = []
        for path in cur_paths:
            last_feat = path[-1]
            if last_feat.layer == 0 and last_feat.sublayer == 'resid_pre':
                continue

            top_contribs = get_top_contribs(model, transcoder, cache, feature_vector, k=num_branches)
            for contrib in top_contribs:
                new_paths.append(path + [contrib])

        # prune top num_branches paths by final contrib score
        scores = torch.tensor([p[-1].contrib for p in new_paths])
        _, top_idx = torch.topk(scores, k=min(num_branches, len(new_paths)))
        cur_paths = [new_paths[i] for i in top_idx]
        all_paths.append(cur_paths)

    return all_paths

def print_all_paths(paths):
    for i, layer_paths in enumerate(paths):
        print(f"--- Depth {i+1} ---")
        for j, path in enumerate(layer_paths):
            print(f"Path [{j}]: " + " <- ".join(str(x) for x in path))


def print_all_paths(paths):
    if len(paths) == 0: return
    if type(paths[0][0]) is list:
        for i, cur_paths in enumerate(paths):
            try:
                print(f"--- Paths of size {len(cur_paths[0])} ---")
            except:
                continue
            for j, cur_path in enumerate(cur_paths):
                print(f"Path [{i}][{j}]: ", end="")
                print(" <- ".join(map(lambda x: x.__str__(show_full=False, show_last_token=True), cur_path)))
    else:
        for j, cur_path in enumerate(paths):
            print(f"Path [{j}]: ", end="")
            print(" <- ".join(map(lambda x: x.__str__(show_full=False, show_last_token=True), cur_path)))
    




    
