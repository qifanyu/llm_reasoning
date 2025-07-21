pip3 install -r requirements.txt
pip3 install vllm "numpy<2.0" configargparse deepspeed datasets accelerate tensorboard
pip3 install --upgrade flash-attn pyarrow transformers

# You might need your hf_access_token here
HF_TOKEN=<hf_access_token>
huggingface-cli login --token $HF_TOKEN