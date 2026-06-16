import subprocess
import os,sys

def quantize(model_name):
    proc = subprocess.Popen(rf".\tools\llama-b9642-bin-win-cuda-12.4-x64\llama-quantize.exe ./out_models/{model_name}.gguf ./out_models/{model_name}-q4.gguf Q4_K_M",
        stdout=sys.stdout)
    proc.wait()

def toGGUF(model_name):
    proc = subprocess.Popen(rf"python .\tools\llama.cpp\convert_hf_to_gguf.py .\build_models\{model_name} --outfile ./out_models/{model_name}.gguf --outtype f16",
        stdout=sys.stdout)
    proc.wait()