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

uv pip install datasets numpy pandas matplotlib dotenv math-verify transformers
```


### GRPO Math rl v2
```bash
source ~/miniconda3/bin/activate && conda create --prefix ./math-rl-v2 python=3.10
source ~/miniconda3/bin/activate ./math-rl-v2
pip install uv

uv pip install unsloth vllm wandb datasets dotenv math-verify
uv pip install --upgrade pillow

tmux new -s grpo-big-math-rl-v2
source ~/miniconda3/bin/activate ./math-rl-v2

python grpo-big-math-rl-v2.py

python grpo_big_math_rl_v3.py
```


### GRPO Math rl v4
```bash
source ~/miniconda3/bin/activate && conda create --prefix ./math-rl-v4 python=3.10
source ~/miniconda3/bin/activate ./math-rl-v4
pip install uv

uv pip install unsloth vllm wandb datasets dotenv math-verify scikit-learn
uv pip install --upgrade pillow

tmux new -s grpo-big-math-rl-v4
source ~/miniconda3/bin/activate ./math-rl-v4

python grpo_big_math_rl_v4.py
```


### PPO Math rl v1
```bash
source ~/miniconda3/bin/activate && conda create --prefix ./ppo-math-rl-v1 python=3.10
source ~/miniconda3/bin/activate ./ppo-math-rl-v1
pip install uv
uv pip install trl

tmux new -s ppo-big-math-rl-v1
source ~/miniconda3/bin/activate ./ppo-math-rl-v1

python ppo_big_math_r1_v1.py
```


### SimpleRL Zero - repro v1
https://github.com/hkust-nlp/simpleRL-reason

- 4 nodes with 8 H100 GPUs
- 1.5 days
- Each node costs $2.99/hr per GPU (on runpod.io)
    => $23.92/hr
    => $95.68/hr for 4 nodes
    => $3444.48 for 1.5 days



```bash

```