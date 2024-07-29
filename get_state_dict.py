from typing import Dict, Any
import torch
from torch import nn

import exllamav2.linear
import exllamav2.embedding
import exllamav2.rmsnorm
import exllamav2.model

is_norm = lambda x: x.endswith("norm")
is_linear = lambda x: x.endswith("proj") or x.endswith("lm_head")
is_embedding = lambda x: x.endswith("embed_tokens")

get_embedding = lambda x: x.embedding.weight.data
get_linear = lambda x: x.linear.weight.data
get_rms_norm = lambda x: x.weight.data


# Assumes the three types listed here are the *only* ones that need specific
# handling. This is just a guess made empirically through debugging and assuming
# Llama architectural sameness
def get_state_dict(modules_dict: Dict[str, Any]) -> Dict[str, torch.tensor]:
    state_dict = {}
    for key, value in modules_dict.items():
        if isinstance(value, exllamav2.embedding.ExLlamaV2Embedding):
            state_dict[key] = get_embedding(value)
        if isinstance(value, exllamav2.linear.ExLlamaV2Linear):
            state_dict[key] = get_linear(value)
        if isinstance(value, exllamav2.rmsnorm.ExLlamaV2RMSNorm):
            state_dict[key] = get_rms_norm(value)

    return state_dict

# Assumption is keys of modules_dict match one-to-one with state_dict, which should be the case due to how I've
# defined get_state_dict. Also assumes that `modules_dict` contains the weights that are actually used during
# forward passes.

# It seems like `model.modules` may be what's used to load the weights. Refer to line 535 of model.py
# Line 300 might be the key to understanding how they relate. If I have to, I could probably "fill"
# the weights in `model.modules` with those from `model.modules_dict` using the relationship in line 300
# if I had to.
def load_state_dict_into_model(model: exllamav2.model.ExLlamaV2, state_dict: Dict[str, torch.tensor]) -> None:
    for key, _ in state_dict.items():
        for mod in model.modules:
            key = mod.key
            if is_embedding(key):
                mod.embedding = nn.Embedding(model.config.vocab_size, model.config.hidden_size, model.config.pad_token_id, device = "meta")
                mod.embedding.weight = nn.Parameter(state_dict[key])
            elif is_norm(key):
                mod.weight = nn.Parameter(state_dict[key])
            elif is_linear(key):
                mod.linear = nn.Linear(mod.in_features, mod.out_features,
                                            mod.has_bias, device = "cuda",
                                            dtype=torch.float16)
                mod.linear.weight = nn.Parameter(state_dict[key])

            for i in range(len(mod.submodules)):
                key = mod.submodules[i].key
                if is_embedding(key):
                    mod.submodules[i].embedding.weight.data = state_dict[key]
                elif is_norm(key):
                    mod.submodules[i].weight = nn.Parameter(state_dict[key])
                elif is_linear(key):
                    mod.submodules[i].linear = nn.Linear(mod.submodules[i].in_features, mod.submodules[i].out_features,
                                            mod.submodules[i].has_bias, device="cuda",
                                            dtype=torch.float16)
                    mod.submodules[i].linear.weight = nn.Parameter(state_dict[key])

