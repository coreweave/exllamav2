from typing import Dict, Any
import torch

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
    for key, value in modules_dict:
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
def load_state_dict_into_model(model: exllamav2.model.ExLlamaV2, state_dict: Dict[str, torch.tensor]) -> None:
    for key, value in state_dict:
        if is_embedding(key):
            model.modules_dict[key].embedding.weight.data = state_dict[key]
        if is_norm(key):
            model.modules_dict[key].weight.data = state_dict[key]
        if is_linear(key):
            model.modules_dict[key].linear.data = state_dict[key]