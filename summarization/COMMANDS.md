```bash
cd summarization
source ~/miniconda3/bin/activate && conda create --prefix ./env python=3.10
source ~/miniconda3/bin/activate ./env
pip install uv

uv pip install datasets torch transformers tqdm numpy "accelerate>=0.26.0"
```