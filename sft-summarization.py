#!/usr/bin/env python
# coding: utf-8

# In[6]:


import datasets
import pandas as pd

dataset = datasets.load_dataset("trl-lib/tldr")

dataset['train'][0]


# In[7]:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


# In[ ]:


print("Converting training dataset to tokens...")
def tokenize_chat(rows):
    chats = [[
        {"role": "system", "content": prompt},
        {"role": "assistant", "content": completion},
    ] for prompt, completion in zip(rows["prompt"], rows["completion"])]
    chat_text = [
        tokenizer.apply_chat_template(chat, tokenize=False)
        for chat in chats
    ]
    return tokenizer(chat_text, padding=True, truncation=True, max_length=1024, return_tensors="pt")

tokenized_dataset = dataset.map(tokenize_chat, batched=True)


# In[20]:


from transformers import TrainingArguments, Trainer
import wandb
import os
from dotenv import load_dotenv

load_dotenv()

output_dir = "Qwen2-0.5B-SFT"

wandb.login(os.environ["WANDB_API_KEY"])

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    report_to = "wandb", 
    save_steps = 500,
    save_total_limit = 5,
    num_train_epochs=3,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)

trainer.train()

