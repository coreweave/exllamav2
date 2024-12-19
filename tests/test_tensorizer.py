"""
This module tests exllamav2's ability to save and load tensorizer-serialized
weights locally and over S3.

To perform these tests, a directory containing model artifacts must be
provided for local_dir. If these are not found, the test will save some
to local_dir as a pre-test fixture.

For testing serialization and deserialization to S3, S3 credentials and URI
can be provided via the following environment variables, with self-explanatory
meanings:

S3_ACCESS_KEY_ID
S3_SECRET_ACCESS_KEY
S3_ENDPOINT_URL
S3_URI

If S3_URI is not provided, the S3 machinery testing will be skipped. For local
serialization and deserialization, model weights are saved and loaded from
/tmp/tensorized, and then the directory is deleted upon completion of the test
"""

import base64
import hashlib
import logging
import os
import shutil
from util.tensorizer_utils import serialize

logging.basicConfig(level=logging.INFO)
mylogger = logging.getLogger()

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
)

import pytest
import torch
import gc

local_dir = os.environ[
    "LOCAL_DIR"] if "LOCAL_DIR" in os.environ else "/tmp/model"

# shutil.rmtree is called on this; edit with care
_serialized_dir = "/tmp/tensorized"

s3_access_key_id = os.environ[
    "S3_ACCESS_KEY_ID"] if "S3_ACCESS_KEY_ID" in os.environ else None
s3_secret_access_key = os.environ[
    "S3_SECRET_ACCESS_KEY"] if "S3_SECRET_ACCESS_KEY" in os.environ else None
s3_endpoint = os.environ[
    "S3_ENDPOINT_URL"] if "S3_ENDPOINT_URL" in os.environ else None
s3_uri = os.environ[
    "S3_URI"] if "S3_URI" in os.environ else None


def tensor_hash(t):
    mv = t.cpu().numpy().data
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("ascii"))
    h.update(b"\0")
    h.update(str(tuple(t.size())).encode("ascii"))
    h.update(b"\0")
    h.update(mv)
    return h.digest()


def get_tensors(module):
    k = module.key
    if hasattr(module, "get_weight"):
        w = module.get_weight()
        if isinstance(w, tuple):
            yield (f"{k}.weight", w[0])
            yield (f"{k}.bias", w[1])
        else:
            yield (f"{k}.weight", w)
    for submodule in module.submodules:
        yield from get_tensors(submodule)


def extract_tensors(model):
    for module in model.modules:
        yield from get_tensors(module)


def hash_dict_checker() -> callable:
    hash_dicts: list[dict] = []

    def wrapper(h_dict: dict) -> bool:
        nonlocal hash_dicts
        hash_dicts.append(h_dict)
        return all([h_dict == hd for hd in hash_dicts])

    return wrapper


@pytest.fixture(scope="session")
def setup_and_teardown_deserializer_test():
    # Initialize the hash dict checker
    checker = hash_dict_checker()
    yield checker

    # Delete the saved tensors after a test session
    shutil.rmtree(_serialized_dir)


@pytest.fixture()
def save_base_model():
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_artifacts_are_cached: bool = (os.path.isdir(local_dir)
                                        and len(os.listdir(local_dir)) > 0)

    # TODO: This is a bit ugly. This can be better
    if not model_artifacts_are_cached:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        token = os.environ.get("HF_TOKEN")

        # Ensure the save directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            print(f"Created directory: {local_dir}")

        # Load the model and tokenizer
        print("Downloading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

        # Save the tokenizer and model
        print(f"Saving model and tokenizer to {local_dir}...")
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        print("Model and tokenizer saved successfully.")

    yield local_dir


@pytest.fixture(autouse=True)
def cleanup():
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture()
def serialize_model(save_base_model, request, cleanup):
    use_s3 = request.param

    model_dir = local_dir
    config = ExLlamaV2Config()
    config.model_dir = model_dir
    s3_creds = None

    path = _serialized_dir

    config.write_state_dict = True

    if use_s3:
        s3_access_key_id = os.environ[
            "S3_ACCESS_KEY_ID"] if "S3_ACCESS_KEY_ID" in os.environ else None
        s3_secret_access_key = os.environ[
            "S3_SECRET_ACCESS_KEY"] if "S3_SECRET_ACCESS_KEY" in os.environ else None
        s3_endpoint = os.environ[
            "S3_ENDPOINT_URL"] if "S3_ENDPOINT_URL" in os.environ else None
        s3_uri = os.environ[
            "S3_URI"] if "S3_URI" in os.environ else None

        s3_creds = {
            "s3_access_key_id": s3_access_key_id,
            "s3_secret_access_key": s3_secret_access_key,
            "s3_endpoint": s3_endpoint
        }

        if not s3_uri:
            pytest.skip("No s3_uri provided so skipping..")

        path = s3_uri

    config.prepare()

    model = ExLlamaV2(config)
    model.load()

    assert path is not None

    serialize(model,
              path,
              s3_creds=s3_creds
              )

    yield


@pytest.fixture()
def load_model(save_base_model, request):
    saved_dir = save_base_model
    config = ExLlamaV2Config()
    config.model_dir = saved_dir
    config.load_with_tensorizer = request.param
    if config.load_with_tensorizer:
        # TODO: If these two have to be the same, that's dumb
        config.model_dir = _serialized_dir

    config.prepare()

    model = ExLlamaV2(config)

    model.load()

    tokenizer = ExLlamaV2Tokenizer(config)
    cache = ExLlamaV2Cache_8bit(model, batch_size=4)
    yield model, tokenizer, cache


# If S3 creds aren't set up, this should skip the last parametrization
@pytest.mark.parametrize("serialize_model, load_model",
                         [(False, False), (False, True), (True, True)],
                         indirect=True)
def test_deserialize_model(setup_and_teardown_deserializer_test,
                           tmpdir, serialize_model, load_model):
    checker = setup_and_teardown_deserializer_test
    model, tokenizer, cache = load_model
    state_dict = dict(extract_tensors(model))

    hash_dict = {k: base64.b64encode(tensor_hash(v)) for k, v in
                 state_dict.items()}

    assert hash_dict
    assert checker(hash_dict)
