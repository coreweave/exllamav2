from typing import Dict, Any
import torch

import exllamav2.linear
import exllamav2.embedding
import exllamav2.rmsnorm



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

def load_state_dict_into_model(state_dict: Dict[str, torch.tensor]) -> None:
    for key, value in state_dict:
        ## TODO: Need some way of knowing how to map a str key like 'model.embed_tokens' to ExLlamaV2Embedding etc
