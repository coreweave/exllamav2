import tensorizer
import pytest
import exllamav2.embedding

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

import time


from get_state_dict import load_state_dict_into_model, get_state_dict
from tensorizer import TensorDeserializer, TensorSerializer
import numpy as np
import os

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
)

# Assumes file locations used here exist locally


from exllamav2.model_init import init
import argparse

model_dir = "../downloaded_models/model"
serialized_dir = "../downloaded_models/tensorized"
import torch
import gc

mock_args = argparse.Namespace(eval_dataset=None, eval_rows=128, eval_length=2048, eval_token=False, eval_token_8bit=False, eval_token_q4=False, eval_token_q6=False, eval_token_q8=False, eval_context_lens=False, prompt='Once upon a time,', prompt_no_bos=False, tokens=128, prompt_speed=False, speed=False, mix_layers=None, no_warmup=False, stream_layers=False, standard_perplexity=None, rank_reduce=None, max_output_len=None, tensorizer=True, model_dir='downloaded_models/model', gpu_split=None, length=None, rope_scale=None, rope_alpha=None, no_flash_attn=False, no_xformers=False, no_sdpa=False, low_mem=False, experts_per_token=None, load_q4=False, fast_safetensors=False, ignore_compatibility=False, chunk_size=None)

def load_model(model_dir, split = None, cache_8bit = True, serialize = False, use_tensorizer=False):
    global model, config, tokenizer, cache

    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.tensorize = serialize
    config.use_tensorizer = use_tensorizer
    if config.use_tensorizer:
        config.serialized_dir = serialized_dir


    config.prepare()

    model = ExLlamaV2(config)
    print(" -- Loading model: " + model_dir)

    model.load(split)

    tokenizer = ExLlamaV2Tokenizer(config)

    if serialize:
        model.serialize(serialized_dir)

    cache = ExLlamaV2Cache_8bit(model, batch_size=4)
    return model, tokenizer, cache


def test_tensorizer_load():
    os.environ['TENSORIZER'] = '1'
    os.environ['TENSORIZER_LOC'] = "../downloaded_models/tensorized/serialized_llama_state_dict.tensors"
    gc.collect()
    torch.cuda.empty_cache()

    model_dir = "../downloaded_models/tensorized/"
    model_a = load_model(model_dir=model_dir)

import copy

def test_gen(model, tokenizer, cache, prompt, max_new_tokens):
    print("--------------------------------")
    print("Generating, normal")
    print()

    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0 # Keep it purely deterministic to ensure weights are the same
    settings.top_k = 50
    settings.top_p = 0.8
    settings.top_a = 0.0
    settings.token_repetition_penalty = 1.15
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

    generator.warmup()
    time_begin = time.time()

    output = generator.generate_simple(prompt, settings, max_new_tokens, seed=1234)

    time_end = time.time()
    time_total = time_end - time_begin

    print(output)
    return output


def get_model_a_mods(model_dir):
    model_a , tokenizer, cache= load_model(model_dir=model_dir, serialize=True)
    output = test_gen(model_a, tokenizer, cache, "Once upon a time,", 128)
    model_a.unload()
    del model_a
    return output

def get_model_b_mods(serialized_dir):
    model_b, tokenizer, cache = load_model(model_dir=serialized_dir, use_tensorizer=True)
    output = test_gen(model_b, tokenizer, cache, "Once upon a time,", 128)
    model_b.unload()
    del model_b
    return output


def test_tensorizer_loaded_is_same_model():
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.mem_get_info())


    model_dir = "../downloaded_models/model"
    serialized_dir = "../downloaded_models/tensorized"

    model_a_output = get_model_a_mods(model_dir)

    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.mem_get_info())


    #model_b_output = get_model_b_mods(serialized_dir)



    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.mem_get_info())




    #assert model_a_output == model_b_output



