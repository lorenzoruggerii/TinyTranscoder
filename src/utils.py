import torch
from gpt import GPTLanguageModel
from transformers import GPT2Tokenizer
from transcoder import CrossLayerTranscoder
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def search_string_in_tokens(pattern, text, tokenizer):

    tokenizer.pad_token = tokenizer.eos_token
    
    if isinstance(text, str):
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        char_token_position = [
            idx for idx, token_id in enumerate(input_ids)
            if pattern in tokenizer.convert_ids_to_tokens(token_id)
        ]
    elif isinstance(text, list):
        input_ids = tokenizer(text, padding=True)['input_ids']
        char_token_position = [
            (b, idx) for b in range(len(input_ids)) for idx, token_id in enumerate(input_ids[b]) 
            if pattern in tokenizer.convert_ids_to_tokens(token_id)
        ]
    else:
        raise ValueError("Text must be one of str or List[str]")
    
    return char_token_position

def extract_most_act_feat(char_token_pos, outs, layer, k=5, report=False):
    batch_indices, token_indices = zip(*char_token_pos)
    selected_activations = outs['feature_activations'][layer][batch_indices, token_indices]

    # Sum activations over tokens
    selected_activations = selected_activations.sum(dim=-2)
    
    # Now search for most activated features
    vals, idxs = torch.topk(selected_activations, k=k)

    # Print the report of most act features
    if report:
        for val, idx in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
            print(f"Feature n.{idx}: {val}")

    return idxs

def get_k_prompts(tokenizer, dataset, tc, k=30):

    # Set pad token for tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.select(range(k))
    input_ids = tokenizer(list(dataset['text']), return_tensors='pt', truncation=True, max_length=tc.model_cfg.block_size)['input_ids']
    return input_ids.cuda()

def which_most_act(text, tokenizer, tc, k=5, report=True):

    # Set pad token for tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=tc.model_cfg.block_size)['input_ids'].cuda()

    outs = tc(input_ids)

    for layer_idx in range(tc.model_cfg.n_layer):

        print(f"--- Layer {layer_idx} ---")

        activations_from_layer = outs['feature_activations'][layer_idx]
        act_last_token = activations_from_layer[:, -1, :].mean(dim=0)
        topk_vals, topk_feats = torch.topk(act_last_token, k=k)
        topk_feats = topk_feats.cpu().tolist()
        
        if report:
            for i, feat in enumerate(topk_feats):
                print(f"Feat #{i+1}: {feat}")

    return outs['last_hidden_state']

def suppress_features_across_layers(
    text_prompt: str,
    feature_suppression: dict[int, list[int]],
    m: GPTLanguageModel,
    tc: CrossLayerTranscoder,
    tok: GPT2Tokenizer
) -> torch.Tensor:
    """
    Suppresses selected sparse features at arbitrary MLP layers in a transformer,
    and returns the modified logits for the next token prediction.
    
    Args:
        text_prompt: input text string
        feature_suppression: dict mapping layer_idx -> list of feature indices to suppress
        m: the model used for making predictions
        tc: the trained transcoder
        tok: the tokenizer used to tokenize text_prompt

    Returns:
        logits for the next token after suppression
    """
    # Tokenize
    input_ids = tok(text_prompt, return_tensors='pt')['input_ids'].cuda()
    
    # Run model and capture cache
    logits, cache = m.run_with_cache(input_ids)

    # Determine first modified layer (to resume forward from there)
    if not feature_suppression:
        return logits[0, -1]  # no changes needed
    
    # Save original logits for comparison
    original_logits = logits[0, -1]

    min_layer = min(feature_suppression.keys())

    # Start from the residual stream at the *start* of min_layer
    resid = cache['blocks'][min_layer]['pre']  # shape [1, T, D]

    # From min_layer to end, run manually
    for layer in range(min_layer, m.cfg.n_layer):
        block = m.blocks[layer]

        # If this layer is in the suppression dict:
        if layer in feature_suppression:
            # 1. Run attention block
            attn_out = block.sa(block.ln1(resid))
            resid = resid + attn_out

            # 2. Get MLP input (ln2)
            mlp_input = block.ln2(resid)
            last_tok_acts = mlp_input[0, -1]  # [hidden_dim]

            # 3. Sparse feature encoding and suppression
            feats = tc.encoders[layer](last_tok_acts)
            feats = tc.activation_functions[layer](feats)
            feats[feature_suppression[layer]] = 0

            # 4. Decode to get modified MLP output
            modified_mlp_out = tc.decoders[layer](feats)
            mlp_out = block.ffwd(mlp_input).clone()
            mlp_out[0, -1] = modified_mlp_out

            # 5. Update resid
            resid = resid + mlp_out

        else:
            # Run layer normally
            resid = block(resid)

    # Final layer norm and logits
    pre_logits = m.ln_f(resid)
    modified_logits = m.lm_head(pre_logits)

    # Print report
    new_token = original_logits.softmax(dim=-1).argmax()
    print(f"Original logits. Best Token {new_token} with value {original_logits[new_token]}. Token: {tok.decode(new_token)} with p={original_logits.softmax(dim=-1).max()}")
    new_token_modified = modified_logits[0, -1].softmax(dim=-1).argmax()
    print(f"Modified logits. Best Token {new_token_modified} with value {modified_logits[0, -1][new_token_modified]}. Token: {tok.decode(new_token_modified)} with p={modified_logits[0, -1].softmax(dim=-1).max()}")

    # Print change of probability
    print(f"Probabilities change from {original_logits.softmax(dim=-1)[new_token]} to {modified_logits[0, -1].softmax(dim=-1)[new_token]}")
    
    return original_logits, modified_logits[0, -1]

def activate_features_across_layers(
    text_prompt: str,
    feature_activation: dict[int, list[int]],
    m: GPTLanguageModel,
    tok: GPT2Tokenizer,
    tc: CrossLayerTranscoder,
    activation_value: float = 10.0  # Can adjust based on desired strength
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Activates selected sparse features at arbitrary MLP layers in a transformer,
    and returns the modified logits for the next token prediction.
    
    Args:
        text_prompt: input text string
        feature_activation: dict mapping layer_idx -> list of feature indices to activate
        activation_value: float value to assign to activated features

    Returns:
        (modified_probs, original_probs): softmax distributions over vocab
    """

    # Tokenize
    input_ids = tok(text_prompt, return_tensors='pt')['input_ids'].cuda()

    # Run model and capture cache
    logits, cache = m.run_with_cache(input_ids)

    # Save original logits for comparison
    original_logits = logits[0, -1]

    # Early exit if no feature modifications
    if not feature_activation:
        return original_logits.softmax(dim=-1), original_logits.softmax(dim=-1)

    min_layer = min(feature_activation.keys())

    # Start from the residual stream at the *start* of min_layer
    resid = cache['blocks'][min_layer]['pre']  # shape [1, T, D]

    # From min_layer to end, run manually
    for layer in range(min_layer, m.cfg.n_layer):
        block = m.blocks[layer]

        if layer in feature_activation:
            # 1. Attention block
            attn_out = block.sa(block.ln1(resid))
            resid = resid + attn_out

            # 2. Get MLP input (ln2)
            mlp_input = block.ln2(resid)
            last_tok_acts = mlp_input[0, -1]  # [hidden_dim]

            # 3. Encode to sparse feature space
            feats = tc.encoders[layer](last_tok_acts)
            feats = tc.activation_functions[layer](feats)

            # 4. Activate selected features (force high values)
            feats[feature_activation[layer]] = activation_value

            # 5. Decode back to hidden space
            modified_mlp_out = tc.decoders[layer](feats)

            # 6. Replace MLP output (last token only)
            mlp_out = block.ffwd(mlp_input).clone()
            mlp_out[0, -1] = modified_mlp_out

            # 7. Update residual stream
            resid = resid + mlp_out

        else:
            # Run layer normally
            resid = block(resid)

    # Final logits
    pre_logits = m.ln_f(resid)
    modified_logits = m.lm_head(pre_logits)

    # Print report
    new_token = original_logits.softmax(dim=-1).argmax()
    print(f"Original logits. Best Token {new_token} with value {original_logits[new_token]}. Token: {tok.decode(new_token)} with p={original_logits.softmax(dim=-1).max()}")
    new_token_modified = modified_logits[0, -1].softmax(dim=-1).argmax()
    print(f"Modified logits. Best Token {new_token_modified} with value {modified_logits[0, -1][new_token_modified]}. Token: {tok.decode(new_token_modified)} with p={modified_logits[0, -1].softmax(dim=-1).max()}")

    # Print probabilities
    print(f"Probabilities changed from {original_logits.softmax(dim=-1)[new_token_modified]} to {modified_logits[0, -1].softmax(dim=-1)[new_token_modified]}")
    
    return original_logits, modified_logits[0, -1]

def plot_top_tokens_comparison(original_logits, modified_logits, tokenizer, top_k=10):
    """
    Plot a barplot comparing top-k most probable tokens from original and modified logits.

    Args:
        original_logits (torch.Tensor): Logits before intervention (1D tensor of shape [vocab_size])
        modified_logits (torch.Tensor): Logits after intervention (same shape)
        tokenizer (transformers.PreTrainedTokenizer): Hugging Face tokenizer to decode token ids
        top_k (int): Number of top tokens to show
    """
    # Convert logits to probabilities
    original_probs = torch.softmax(original_logits, dim=-1)
    modified_probs = torch.softmax(modified_logits, dim=-1)

    # Get top-k token indices for both
    orig_topk_vals, orig_topk_idxs = torch.topk(original_probs, top_k)
    mod_topk_vals, mod_topk_idxs = torch.topk(modified_probs, top_k)

    # Decode tokens
    orig_tokens = tokenizer.convert_ids_to_tokens(orig_topk_idxs.tolist())
    mod_tokens = tokenizer.convert_ids_to_tokens(mod_topk_idxs.tolist())

    # Build bar plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Original
    axs[0].barh(range(top_k), orig_topk_vals.tolist(), color='skyblue')
    axs[0].set_yticks(range(top_k))
    axs[0].set_yticklabels(orig_tokens)
    axs[0].invert_yaxis()
    axs[0].set_title("Original logits: Top {} tokens".format(top_k))
    axs[0].set_xlabel("Probability")
    axs[0].set_xlim([0, 1])

    # Modified
    axs[1].barh(range(top_k), mod_topk_vals.tolist(), color='salmon')
    axs[1].set_yticks(range(top_k))
    axs[1].set_yticklabels(mod_tokens)
    axs[1].invert_yaxis()
    axs[1].set_title("Modified logits: Top {} tokens".format(top_k))
    axs[1].set_xlabel("Probability")
    axs[1].set_xlim([0, 1])

    plt.tight_layout()
    plt.show()



def plot_activations_plotly(tc, activations, tokens, selected_layer, selected_feature):
    """
    Plot a barplot for each passed sequence using plotly.

    Args:
        tc (torch.Tensor): The trained transcoder
        activations (torch.Tensor): Activations from transcoder on input sequences
        tokens (torch.Tensor): The tokenized input sequences
        selected_layer (int): Transcoder feature's layer to show
        selected_feature (int): Transcoder feature's to show
    """
    num_layers = tc.model_cfg.n_layer
    num_features = tc.cfg.num_features
    acts = activations  # shape: (num_layers, B, T, num_features)

    fig = go.Figure()

    # Add one bar trace per sequence, only the first visible initially
    for seq_idx, toks in enumerate(tokens):
        acts_seq = acts[selected_layer][seq_idx, :, selected_feature]
        acts_seq = acts_seq.cpu().detach().numpy()
        clean_toks = [t.replace('Ġ', '').strip() for t in toks]

        fig.add_trace(go.Bar(
            x=clean_toks,
            y=acts_seq,
            name=f"Seq {seq_idx}",
            marker=dict(
                color=acts_seq,
                colorscale="YlOrRd",
                showscale=True
            ),
            visible=(seq_idx == 0)
        ))

    # Add dropdown menu to select which sequence to display
    buttons = []
    for seq_idx in range(len(tokens)):
        visibility = [False] * len(tokens)
        visibility[seq_idx] = True
        buttons.append(dict(label=f"Sequence {seq_idx}",
                            method="update",
                            args=[{"visible": visibility},
                                {"title": f"Sequence {seq_idx} - Layer {selected_layer}, Feature {selected_feature}"}]))

    fig.update_layout(
        title=f"Sequence 0 - Layer {selected_layer}, Feature {selected_feature}",
        barmode="group",
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=0.5, xanchor="center", y=1.2, yanchor="top")]
    )

    fig.show()
