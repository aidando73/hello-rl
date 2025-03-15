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
