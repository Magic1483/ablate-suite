import torch
from steering import HookConfig
import steering
import utils
import gc
from typing import Optional,List

def generate(
        model, 
        tokenizer, 
        prompt, 
        hooks: None | Optional[List[HookConfig]],
        max_new_tokens=80,
        mode='all',
        strength=1.0,
    ):
    tokens = utils.make_tokens(tokenizer, prompt, model.cfg.device)
    kwargs = {
        "max_new_tokens":max_new_tokens,
        "temperature":0.0,
        "do_sample":False,
        "verbose":False
    }

    with torch.inference_mode():
        if hooks:
            lhooks = []
            for h in hooks:
                hook_fn = steering.make_remove_hook(h.refusal_dir,mode,strength)
                lhooks.append((h.hook_name,hook_fn))
            with model.hooks(fwd_hooks=lhooks):
                out = model.generate(tokens, **kwargs )
        else:
            out = model.generate(tokens, **kwargs )
            
    text = utils.decode_new(tokenizer, out, tokens)

    del tokens, out
    gc.collect()
    torch.cuda.empty_cache()
    return text