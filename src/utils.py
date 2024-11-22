import random
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from statistics import mean
import copy
import json
from typing import Any, Mapping
import itertools
import sys
sys.path.append('src/')
import open_clip
from seqprotes import seqprotes
import torch


from STEPSPrompt import STEPSPrompt

import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])



from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)
# defer loading other stuff until we confirm the images loaded
import argparse
import torch.nn.functional as F
from PIL import Image
from types import SimpleNamespace
import sys
from PIL import Image


import logging


# config_path = "sample_config.json"
# with open(config_path, 'r') as f:
#     config = json.load(f)
# args = SimpleNamespace(**config)

def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def nn_project(curr_embeds, embedding_layer, print_hits=False):
    with torch.no_grad():
        bsz,seq_len,emb_dim = curr_embeds.shape
        
        # Using the sentence transformers semantic search which is 
        # a dot product exact kNN search between a set of 
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1,emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds) # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)
        
        hits = semantic_search(curr_embeds, embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=1,
                                score_function=dot_score)

        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            logging.info(f"mean hits:{mean(all_hits)}")
        
        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        nn_indices = nn_indices.reshape((bsz,seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()

    texts = []

    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))

            texts.append('|'.join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))

    return texts


def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


def get_target_feature(model, preprocess, tokenizer_funct, device, target_images=None, target_prompts=None):
    if target_images is not None:
        with torch.no_grad():
            curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
            curr_images = torch.concatenate(curr_images).to(device)
            all_target_features = model.encode_image(curr_images)
    else:
        texts = tokenizer_funct(target_prompts).to(device)
        all_target_features = model.encode_text(texts)

    return all_target_features


def initialize_prompt(tokenizer, token_embedding, input_ids, args, device):
    prompt_len = args.prompt_len

    # randomly optimize prompt embeddings
    # prompt_ids = torch.randint(len(tokenizer.encoder), (args.prompt_bs, prompt_len)).to(device)
    # transpose input_ids
    input_ids = input_ids.transpose(0, 1)
    prompt_embeds = token_embedding(input_ids).detach()
    prompt_embeds.requires_grad = True

    # initialize the template
    template_text = "{}"
    padded_template_text = template_text.format(" ".join(["<start_of_text>"] * prompt_len))
    dummy_ids = tokenizer.encode(padded_template_text)

    # -1 for optimized tokens
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    dummy_ids = [49406] + dummy_ids + [49407]
    dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * args.prompt_bs).to(device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False
    
    return prompt_embeds, dummy_embeds, dummy_ids


# create the loss function for seqprotes
def create_loss_fun(model, top_indices, input_ids, target_features):
    def index_top_indices(I):
        # Ensure I and top_indices are numpy arrays
        I = np.array(I)
        
        # Get batch_size
        batch_size = I.shape[0]
        
        # Create a range array for selecting correct rows
        row_indices = np.arange(input_ids.shape[0]).reshape(1, input_ids.shape[0]).repeat(batch_size, axis=0)
        
        # Use advanced indexing to get results
        result = top_indices[row_indices, I]
        
        return result

    def loss_fun(I):
        batch_ids = index_top_indices(I)
        # evaluate the loss
        batch_cosim_scores = batch_ids_scores(model, batch_ids, target_features)
        loss_np = batch_cosim_scores.detach().cpu().numpy().squeeze()
        return loss_np
    
    return loss_fun


def find_top_n_elements(vectors_list, top_n=10, per_vector_top_n=10):
    """
    Find the n largest elements in a list of NumPy vectors and return their values and positions.
    Only process the top `per_vector_top_n` elements from each vector.
    
    Parameters:
    - vectors_list: A list containing NumPy vectors
    - top_n: Number of maximum elements to find, default is 10
    - per_vector_top_n: Number of top elements to process per vector, default is 10
    
    Returns:
    - A list containing the n largest elements, each element is a tuple of (value, list_index, vector_index)
    """
    
    all_elements = []

    # Iterate through each vector in the list
    for i, vector in enumerate(vectors_list):
        # If vector length is greater than per_vector_top_n, find indices of top per_vector_top_n largest values
        if len(vector) > per_vector_top_n:
            top_indices = np.argsort(vector)[-per_vector_top_n:]
        else:
            # If vector length is less than or equal to per_vector_top_n, use all indices
            top_indices = np.argsort(vector)

        # Store each maximum value and its corresponding position in all_elements
        for idx in top_indices:
            all_elements.append((vector[idx], i, idx))

    # Sort all values in descending order and take top_n elements
    top_elements = sorted(all_elements, key=lambda x: x[0], reverse=True)[:top_n]

    return top_elements


def extract_rows_from_I_list(I_list, top_elements):
    """
    Extract rows from I_list based on list_idx and vector_idx in top_elements.
    
    Parameters:
    - I_list: A list containing n matrices.
    - top_elements: A list containing tuples of (value, list_idx, vector_idx)
    
    Returns:
    - A list containing extracted rows from I_list.
    """
    extracted_rows = []
    
    for _, list_idx, vector_idx in top_elements:
        # Find the corresponding matrix in I_list
        matrix = I_list[list_idx]
        # Extract the corresponding row based on vector_idx
        row = matrix[vector_idx, :]
        # Append this row to the result list
        extracted_rows.append(row)
    
    return extracted_rows


def extract_tensors(G_list, Index_matrix):
    G1, G2, G3 = G_list
    m, last_top_k = Index_matrix.shape
    r, n = G1.shape[1:]
    G1_hat = G1[:, Index_matrix[0], :]

    G2_hat_list = []
    for i in range(1, m - 1):
        G2_i = G2[i - 1:i, :, Index_matrix[i], :]
        G2_hat_list.append(G2_i)
    G2_hat = np.squeeze(np.stack(G2_hat_list, axis=0))
    G3_hat = G3[:, Index_matrix[-1], :]
    G_hat = [G1_hat, G2_hat, G3_hat]
    return G_hat

def optimize_prompt_loop(model, tokenizer, token_embedding, all_target_features, args, device):

    best_sim = -1000 * args.loss_weight
    best_text = ""
    target_embedding = all_target_features

    # Process target features with smaller batch size
    if args.batch_size is None:
        target_features = all_target_features
    else:
        curr_indx = torch.randperm(len(all_target_features))[:args.batch_size]
        target_features = all_target_features[curr_indx]
    target_features /= target_features.norm(dim=-1, keepdim=True)

    token_embedding = model.token_embedding
    prompt_attack = STEPSPrompt(model=model, tokenizer=tokenizer, token_embedding=token_embedding, args=args, device=device)
    best_text, best_sim, best_step = prompt_attack.run(target_embedding)
    
    logging.info(f"step: {best_step}, cosine sim: {best_sim}, prompt: {best_text}")

    # Clean up unnecessary tensors after each iteration
    torch.cuda.empty_cache()

    return best_text, best_sim 

def batch_ids_scores(model, batch_token_ids, target_features):
    seq_len = batch_token_ids.shape[1]
    expanded_token_ids = torch.zeros(batch_token_ids.shape[0], 77, dtype=batch_token_ids.dtype, device=batch_token_ids.device)
    expanded_token_ids[:, 0] = 49406
    expanded_token_ids[:, 1:seq_len+1] = batch_token_ids
    expanded_token_ids[:, seq_len+1] = 49407

    # Encode expanded token ids into text features
    with torch.no_grad():
        batch_text_features = model.encode_text(expanded_token_ids)

    batch_text_features /= batch_text_features.norm(dim=-1, keepdim=True)
    batch_cosim_scores = batch_text_features @ target_features.t()
    return batch_cosim_scores

def estimate_full_landscapes(model, grad, input_ids, target_features, topk=10, batch_size=3096):
    # Get the top k token IDs for each input_id based on gradient values
    top_indices = (-grad).topk(topk, dim=1).indices

    # Calculate total number of possible combinations
    num_combinations = topk ** input_ids.shape[0]

    # Initialize an empty loss tensor
    loss_tensor = torch.zeros(tuple([topk] * input_ids.shape[0]), device=grad.device)
    
    # Process combinations in batches
    for batch_start in range(0, num_combinations, batch_size):
        batch_end = min(batch_start + batch_size, num_combinations)
        batch_combinations = list(itertools.islice(itertools.product(range(topk), repeat=input_ids.shape[0]), batch_start, batch_end))

        # Convert current batch combinations to tensor
        batch_token_ids = torch.tensor([[top_indices[i, idx] for i, idx in enumerate(combination)] for combination in batch_combinations], device=grad.device)

        batch_cosim_scores = batch_ids_scores(model, batch_token_ids, target_features)

        # Assign loss values to corresponding positions in loss_tensor
        for i, combination in enumerate(batch_combinations):
            loss_tensor[combination] = batch_cosim_scores[i].item()

    # Find maximum loss value and corresponding indices in the complete loss tensor
    max_loss_value = loss_tensor.max().item()
    max_loss_indices = loss_tensor.argmax()
    max_loss_mask = torch.zeros_like(loss_tensor, dtype=torch.bool)
    max_loss_mask.flatten()[max_loss_indices] = True
    max_loss_indices_matrix = torch.nonzero(max_loss_mask, as_tuple=True)
    
    max_loss_indices = torch.stack(max_loss_indices_matrix)
    max_loss_token_ids = torch.gather(top_indices, 1, max_loss_indices)

    return loss_tensor, max_loss_value, max_loss_indices, max_loss_token_ids

def token_gradients(model, input_ids, target_features):
    embed_weights = model.token_embedding.weight
    input_len = input_ids.shape[0]  
    if input_len > 75:
        input_ids = input_ids[:75]
    dummy_ids = torch.full((input_len, 1), -1, dtype=input_ids.dtype, device=input_ids.device)
    one_hot = torch.zeros(
        input_ids.shape[0],
        embed_weights.shape[0],
        device=input_ids.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids,
        torch.ones(one_hot.shape[1], 1, device=input_ids.device, dtype=embed_weights.dtype)
    )    
    one_hot.requires_grad_()

    padding_len = 77 - input_len
    padding_ids = torch.zeros(padding_len, 1, dtype=input_ids.dtype, device=input_ids.device)
    padding_ids[0, 0] = 49406
    padding_ids[1, 0] = 49407
    padding_ids_end = padding_ids[1:, :]
    if padding_ids_end.dim() == 1:
        padding_ids_end = padding_ids_end.unsqueeze(0)
    padding_dummy_ids = torch.cat([padding_ids[0,:].unsqueeze(0), dummy_ids, padding_ids_end], dim=0).to(input_ids.device)
    # then we create a one_hot_tmp vectors based on padding_ids
    one_hot_tmp = torch.zeros(padding_ids.shape[0], embed_weights.shape[0], device=input_ids.device, dtype=embed_weights.dtype)
    one_hot_tmp.scatter_(1, padding_ids, torch.ones(one_hot_tmp.shape[1], 1, device=input_ids.device, dtype=embed_weights.dtype))
    # then we get the embedding based on one_hot_tmp
    start_embeds = (one_hot_tmp[0,:] @ embed_weights).unsqueeze(0)
    end_embeds = (one_hot_tmp[1:,:] @ embed_weights)
    if end_embeds.dim() == 1:
        end_embeds = end_embeds.unsqueeze(0)

    input_embeds = (one_hot @ embed_weights)

    full_embeds = torch.cat(
        [
            start_embeds,
            input_embeds, 
            end_embeds,
        ], 
        dim=0).unsqueeze(0)
    
    # transpose the padding_dummy_ids
    # padding_dummy_ids = padding_dummy_ids.t()
    logits_per_image, _ = model.forward_text_embedding(full_embeds, padding_dummy_ids.t(), target_features)
    loss = 1 - logits_per_image.mean()
    
    loss.backward()
    grad = one_hot.grad.clone()
    result_dummy_ids = padding_dummy_ids.t().clone()

    del one_hot, padding_dummy_ids, full_embeds, logits_per_image, loss
    torch.cuda.empty_cache()
    
    return grad, result_dummy_ids

def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(1, batch_size).t()
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks

def optimize_prompt(model, preprocess, args, device, target_images=None, target_prompts=None):
    # Get token embeddings
    token_embedding = model.token_embedding
    tokenizer = open_clip.tokenizer._tokenizer
    tokenizer_funct = open_clip.get_tokenizer(args.clip_model)

    # Get target features
    all_target_features = get_target_feature(model, preprocess, tokenizer_funct, device, target_images=target_images, target_prompts=target_prompts)

    # Optimize prompt
    learned_prompt, best_sim = optimize_prompt_loop(model, tokenizer, token_embedding, all_target_features, args, device)

    return learned_prompt, best_sim
    

def measure_similarity(orig_images, images, ref_model, ref_clip_preprocess, device):
    with torch.no_grad():
        ori_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in orig_images]
        ori_batch = torch.concatenate(ori_batch).to(device)

        gen_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in images]
        gen_batch = torch.concatenate(gen_batch).to(device)
        
        ori_feat = ref_model.encode_image(ori_batch)
        gen_feat = ref_model.encode_image(gen_batch)
        
        ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
        
        return (ori_feat @ gen_feat.t()).mean().item()
    