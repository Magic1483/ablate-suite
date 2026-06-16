import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformer_lens import HookedTransformer
from datasets import load_dataset, Dataset, concatenate_datasets
import toml

CONFIG = toml.load("CONFIG.toml")

LOCAL_MODEL_DIR = CONFIG['LOCAL_MODEL_DIR']
TL_MODEL_NAME   = CONFIG['TL_MODEL_NAME']

def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]

def get_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    return reformat_texts(dataset['train']['text']), reformat_texts(dataset['test']['text'])

def local_dataset_loader(fname):
    """Load dataset from plain txt file"""
    rows = []
    with open(fname,encoding='utf-8') as f:
        for line in f:
            if line.strip() == "" :continue
            rows.append([{
                "role":"user",
                "content":line.strip()
            }])
    return rows

# colab 512
def tokenize_instructions(tokenizer, instructions):
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids

def make_tokens(tokenizer, prompt, device):
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids
    
    if device is not None:
        tokens = tokens.to(device)
    return tokens

def decode_new(tokenizer, out, tokens):
    new_tokens = out[0, tokens.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def dump_model(hf_model, tokenizer,model_dir):
    print(f"Saving model to {model_dir}")
    os.makedirs(f"build_models/{model_dir}", exist_ok=True)
    hf_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def load_model(model_dir,model_name,dtype=torch.float16):
    print(f"Load model {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, 
        local_files_only=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    print("Binding structural activation trace hooks...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True
    )

    print("Wrapping architecture inside HookedTransformer...")
    model = HookedTransformer.from_pretrained_no_processing(
        model_name=model_name,
        hf_model=hf_model,                             
        tokenizer=tokenizer,
        device="cuda",
        dtype=torch.float16,
        trust_remote_code=True
    )
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    model = model.to("cuda")
    return model, tokenizer

def check_cuda():
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA:", torch.version.cuda)

    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
        print("VRAM allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Stop here.")


