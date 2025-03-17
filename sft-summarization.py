#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datasets
import pandas as pd

dataset = datasets.load_dataset("trl-lib/tldr")

dataset['train'][0]


# In[2]:


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


# In[3]:


# Calculate token lengths for the dataset
def to_chat_template(prompt, completion):
    return tokenizer.apply_chat_template([
        {"role": "system", "content": prompt},
        {"role": "assistant", "content": completion},
    ], tokenize=False)

# print("Calculating token lengths...")
# def get_token_lengths(rows):
#     prompts = rows["prompt"]
#     completions = rows["completion"]
    
#     # Tokenize prompts and completions
#     total_lengths = [tokenizer(to_chat_template(prompt, completion), return_length=True)["length"] for prompt, completion in zip(prompts, completions)]
    
#     return {"total_lengths": total_lengths}

# # Map function to get token lengths
# token_lengths = dataset.map(get_token_lengths, batched=True, batch_size=100)

# # Calculate statistics
# total_lengths = token_lengths["train"]["total_lengths"]

# # Calculate max and 95th percentile
# max_total_length = max(total_lengths)
# p95_total_length = int(torch.tensor(total_lengths, dtype=torch.float32).quantile(0.95).item())

# print(f"Max token lengths: total={max_total_length}")
# print(f"P95 token lengths: total={p95_total_length}")


# In[4]:


print("Converting training dataset to tokens...")
def tokenize_chat(rows):
    chats = [to_chat_template(prompt, completion) for prompt, completion in zip(rows["prompt"], rows["completion"])]

    return tokenizer(
        chats,
        padding="max_length",
        truncation=True,
        # 618 is the max length of the dataset
        max_length=618,
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(tokenize_chat, batched=True)


# In[5]:


# Drop the original prompt and completion columns as they're no longer needed
# after tokenization to save memory
columns_to_remove = ["prompt", "completion", "attention_mask"]
tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

print(f"Columns after removal: {tokenized_dataset['train'].column_names}")


# In[ ]:


from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import wandb
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

output_dir = f"Qwen2-0.5B-SFT-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

wandb.login(key=os.environ["WANDB_API_KEY"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

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
    data_collator=data_collator,
)

trainer.train()

