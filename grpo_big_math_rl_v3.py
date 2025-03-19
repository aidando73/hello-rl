from unsloth import FastLanguageModel
import torch

# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME = "meta-llama/meta-Llama-3.1-8B-Instruct"
LOAD_IN_4BIT = False
RUN_NAME = "v2.2"

max_seq_length = 4096 # Can increase for longer reasoning traces
max_prompt_length = 1300
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    load_in_4bit = LOAD_IN_4BIT, # False for LoRA 16bit
    # Getting crashes related to vLLM
    # fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

import re
from datasets import load_dataset, Dataset
import os
from dotenv import load_dotenv
from math_verify import parse, verify

# Load environment variables from .env file
load_dotenv()

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>

{{answer}}
"""

# uncomment middle messages for 1-shot prompting
def get_big_math_rl_questions(split = "train") -> Dataset:
    data = load_dataset('SynthLabsAI/Big-Math-RL-Verified', token=os.getenv("HUGGINGFACE_TOKEN"))[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['problem']}
        ],
        'answer': x['answer']
    }) # type: ignore
    return data # type: ignore

dataset = get_big_math_rl_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}")
    return [2.0 if verify(parse(r), parse(a)) else 0.0 for r, a in zip(responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [1 if match else 0.0 for match in matches]

import wandb  # Add this import at the top with other imports
import os

wandb.init(project="grpo-big-math-rl-v2", name="math-v2-exploration")
wandb.login(key=os.getenv("WANDB_API_KEY"))

from unsloth import is_bfloat16_supported
from grpo_big_math_rl_v3_rewards import correctness_reward_func, strict_format_reward_func

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    # # Qwen specific
    # use_vllm = True, # use vLLM for fast inference!
    # bf16 = is_bfloat16_supported(),
    # fp16 = not is_bfloat16_supported(),

    # General
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 100, # Set to 1 for a full training run
    # max_steps = 250,
    save_steps = 500,
    save_total_limit = 5,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()


from datetime import datetime

# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model.save_lora(f"grpo_saved_lora_{current_datetime}")