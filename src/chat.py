from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import sys

def load_abliterated_hf(path="./llama32_3b_abliterated_hf"):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer

def test_chat(model_name:str):
    model, tokenizer = load_abliterated_hf(model_name)
    while 1:
        try:
            prompt = input("> ")
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

            print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
        except KeyboardInterrupt:
            sys.exit(0)