import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    save_directory = "./model"
    token = os.environ.get("HF_TOKEN")

    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Created directory: {save_directory}")

    # Load the model and tokenizer
    print("Downloading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name,  token=token)

    # Save the tokenizer and model
    print(f"Saving model and tokenizer to {save_directory}...")
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    print("Model and tokenizer saved successfully.")

if __name__ == "__main__":
    main()