import os
def serialize(model, serialized_dir: str = None):
    from tensorizer import TensorSerializer
    import shutil

    if not model.config.write_state_dict:
        raise ValueError("Model was not loaded with write_state_dict=True, "
                         "which is necessary for serialization. ")

    if not serialized_dir:
        serialized_dir = model.config.serialized_dir
    os.path.join(model.config.model_dir, "config.json")
    serializer = TensorSerializer(
        os.path.join(serialized_dir, "model.tensors"))
    serializer.write_state_dict(model.state_dict)
    serializer.close()

    # Open the destination file and write the JSON content
    shutil.copyfile(os.path.join(model.config.model_dir, "config.json"),
                    os.path.join(serialized_dir, "config.json"))
