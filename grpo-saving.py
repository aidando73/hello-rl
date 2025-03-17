from unsloth import FastLanguageModel
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SAVED_LORA_MODEL = "outputs/checkpoint-33000"
HUB_NAME = "aidando73/llama-3.1-8b-grpo-33000-merged"
LOAD_IN_4BIT = False

max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = SAVED_LORA_MODEL, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    load_in_4bit = LOAD_IN_4BIT
)

model.push_to_hub_merged(HUB_NAME, tokenizer, save_method = "merged_16bit", token = os.getenv("HUGGINGFACE_TOKEN"))