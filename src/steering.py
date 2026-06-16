from tqdm import tqdm
import torch
from collections import defaultdict

def collect_activations_pair(
    model,
    harmful_tokens,
    harmless_tokens,
    n_inst_train,
    batch_size=1,
):
    harmful = defaultdict(list)
    harmless = defaultdict(list)
    num_batches = (n_inst_train + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min(n_inst_train, start_idx + batch_size)

        harmful_batch = harmful_tokens[start_idx:end_idx].to(model.cfg.device)
        harmless_batch = harmless_tokens[start_idx:end_idx].to(model.cfg.device)

        with torch.inference_mode():
            _, harmful_cache = model.run_with_cache(
                harmful_batch,
                names_filter=lambda name:name.endswith(('resid_post')),
                return_type=None,
                reset_hooks_end=True,
            )

            _, harmless_cache = model.run_with_cache(
                harmless_batch,
                names_filter=lambda name: name.endswith(('resid_post')),
                return_type=None,
                reset_hooks_end=True,
            )

        for key in harmful_cache:
            harmful[key].append(
                harmful_cache[key][:, -1, :].detach().float().cpu())

            harmless[key].append(
                harmless_cache[key][:, -1, :].detach().float().cpu())

        del harmful_batch, harmless_batch
        del harmful_cache, harmless_cache

    harmful = {k: torch.cat(v, dim=0) for k, v in harmful.items()}
    harmless = {k: torch.cat(v, dim=0) for k, v in harmless.items()}
    return harmful, harmless

def make_remove_hook(refusal_dir,mode='all',strength=1.0):
    calls = {"n": 0}
    def hook(value, hook):
        calls["n"] += 1

        direction = refusal_dir.to(device=value.device, dtype=value.dtype)
        if value.shape[-1] != direction.shape[0]:
            return value
        
        if mode == 'all':
            projection = value @ direction
            return value - strength * projection[..., None] * direction
        if mode == 'last':
            out = value.clone()
            last = out[:, -1, :]
            projection = last @ direction
            out[:, -1, :] = last - strength * projection[:, None] * direction
            return out
    return hook

class HookConfig:
    def __init__(self,refusal_dir,layer:int,hook_name:str):
        self.layer       = layer
        self.refusal_dir = refusal_dir
        self.hook_name   = hook_name