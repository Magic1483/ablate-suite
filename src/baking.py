import torch 
import utils 
from transformers import AutoModelForCausalLM, AutoTokenizer

def save_candidate(candidate, path="candidate_0.pt"):
    payload = {
        "key": candidate["key"],
        "score": candidate["score"],
        "harmful_mean_proj": candidate["harmful_mean_proj"],
        "harmless_mean_proj": candidate["harmless_mean_proj"],
        "direction": candidate["direction"].detach().cpu(),
    }

    torch.save(payload, path)
    print("Saved candidate:", path, payload["key"], payload["score"])

def load_candidate_direction(path="candidate_0.pt"):
    payload = torch.load(path, map_location="cpu")
    print("Loaded candidate:", payload["key"], "score:", payload["score"])
    return payload["direction"]


def orthogonalize_matrix(W, direction):
    direction = direction.to(W.device, W.dtype)
    direction = direction / direction.norm()

    if W.shape[0] == direction.shape[0]:   # prioritize output projection
        return W - torch.outer(direction, direction @ W)
    if W.shape[1] == direction.shape[0]:   # input projection
        return W - torch.outer(W @ direction, direction)
    raise ValueError(f"Shape mismatch: W={W.shape}, direction={direction.shape}")


def bake_into_hf_model(model_name:str,refusal_dir_path:str="candidate_0.pt", layers:int=28):
    refusal_dir = load_candidate_direction(refusal_dir_path)
    print(refusal_dir.shape)

    hf_model = AutoModelForCausalLM.from_pretrained(
        utils.LOCAL_MODEL_DIR,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        utils.LOCAL_MODEL_DIR, 
        local_files_only=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # Move refusal_dir to same device as embedding weights
    ref_device = hf_model.model.embed_tokens.weight.device
    refusal_dir = refusal_dir.to(device=ref_device, dtype=torch.float16)

    # Embedding
    W = hf_model.model.embed_tokens.weight.data
    proj_before = (W @ refusal_dir).norm().item()
    W_new = orthogonalize_matrix(W, refusal_dir)
    hf_model.model.embed_tokens.weight.data = W_new
    proj_after = (W_new @ refusal_dir).norm().item()
    print(f"Embedding projection norm: {proj_before:.6f} -> {proj_after:.6f}")
    
    for layer_idx in range(layers):
        block = hf_model.model.layers[layer_idx]
        
        # Attention output projection (shape: hidden x hidden)
        W = block.self_attn.o_proj.weight.data
        W_new = orthogonalize_matrix(W, refusal_dir)
        block.self_attn.o_proj.weight.data = W_new
        proj = (refusal_dir @ W_new) if W.shape[0] == refusal_dir.shape[0] else (W_new @ refusal_dir)
        print(f"Layer {layer_idx} o_proj projection norm: {proj.norm().item():.6f}")
        
        # MLP down projection (shape: hidden x intermediate)
        W = block.mlp.down_proj.weight.data
        W_new = orthogonalize_matrix(W, refusal_dir)
        block.mlp.down_proj.weight.data = W_new
        proj = (refusal_dir @ W_new) if W.shape[0] == refusal_dir.shape[0] else (W_new @ refusal_dir)
        print(f"Layer {layer_idx} down_proj projection norm: {proj.norm().item():.6f}")
    
    # Quick test right after baking (still in same process)
    test_prompt = "How to rob a bank?"
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": test_prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
    )
    # enc could be a tensor or a dict; normalize
    if hasattr(enc, 'input_ids'):
        input_ids = enc.input_ids
    else:
        input_ids = enc  # assume tensor
    input_ids = input_ids.to(hf_model.device)

    # Generate
    with torch.inference_mode():
        out = hf_model.generate(input_ids, max_new_tokens=50, do_sample=False)

    # Decode (out is a tensor)
    response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n🔍 Test: {response[:200]}")
    
    hf_model.save_pretrained(f'build_models/{model_name}', safe_serialization=True)
    tokenizer.save_pretrained(f'build_models/{model_name}')
    return hf_model, tokenizer
