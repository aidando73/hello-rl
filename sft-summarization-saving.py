#!/usr/bin/env python
# coding: utf-8

# In[3]:


from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment variables
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Path to the checkpoint you want to load
checkpoint_path = "Qwen2-0.5B-summarize-SFT-2025-03-17/checkpoint-43773"
# Define the repository name for pushing to Hub
repo_name = "aidando73/Qwen2-0.5B-summarize-SFT-2025-03-17-43773"

# Load model and tokenizer from checkpoint
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

print(f"Loaded model and tokenizer from {checkpoint_path}")


# Push the model and tokenizer to the Hugging Face Hub
model.push_to_hub(repo_name, token=huggingface_token)
tokenizer.push_to_hub(repo_name, token=huggingface_token)

print(f"Successfully pushed model and tokenizer to huggingface.co/{repo_name}")

