```bash
source ~/miniconda3/bin/activate && conda create --prefix ./env python=3.10
source ~/miniconda3/bin/activate ./env
pip install uv

uv pip install unsloth vllm
uv pip install --upgrade pillow
uv pip install wandb

tmux new -s grpo

python grpo.py
```


For GRPO Summarization
```bash
source ~/miniconda3/bin/activate && conda create -y --prefix ./env python=3.10
source ~/miniconda3/bin/activate ./env
pip install uv
uv pip install datasets trl accelerate wandb dotenv transformers certifi

tmux new -s grpo-summarization

accelerate launch grpo-summarization.py
accelerate launch --gpu_ids="0,1,2,3,5,6,7" grpo-summarization.py

CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 accelerate launch grpo-summarization.py

# Save a checkpoint and push to Hugging Face
python grpo-summarization-saving.py
```

For SFT Summarization
```bash
source ~/miniconda3/bin/activate && conda create --prefix ./env python=3.10
source ~/miniconda3/bin/activate ./env
pip install uv

uv pip install datasets trl accelerate wandb dotenv

tmux new -s sft-summarization
source ~/miniconda3/bin/activate ./env

python sft-summarization.py
```


Math round two
```bash
source ~/miniconda3/bin/activate && conda create -y --prefix ./math-v2 python=3.10
source ~/miniconda3/bin/activate ./math-v2
pip install uv

uv pip install datasets numpy pandas matplotlib dotenv 
```