# pip install transformers accelerate transformer-lens model2gguf
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

param (
   [switch]$setup, # install main venv
   [switch]$export,
   [switch]$quantize
)

if ($setup){
    python -m venv .venv-export
    .\.venv-export\Scripts\Activate.ps1
    pip install -r ./llama.cpp/requirements.txt
    pip uninstall -y transformers
    pip install -y transformers --pre  
}


if ($export) {
    .\.venv-export\Scripts\Activate.ps1
    python .\llama.cpp\convert_hf_to_gguf.py .\llama32_3b_abliterated_hf --outfile ./mad_model/mad.01.gguf --outtype f16 
}