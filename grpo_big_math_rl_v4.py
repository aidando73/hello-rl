from unsloth import FastLanguageModel
import torch

# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME = "meta-llama/meta-Llama-3.1-8B-Instruct"
LOAD_IN_4BIT = False
RUN_NAME = "v4.3"
HUB_NAME = "aidando73/grpo-big-math-rl-" + RUN_NAME
DESCRIPTION = "Fix issue with configuration"

max_seq_length = 4096 # Can increase for longer reasoning traces
max_prompt_length = 1471
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    load_in_4bit = LOAD_IN_4BIT, # False for LoRA 16bit
    # Getting crashes related to vLLM
    # fast_inference = True, # Enable vLLM fast inference
    # gpu_memory_utilization = 0.6, # Reduce if out of memory
    max_lora_rank = lora_rank,
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
SYSTEM_PROMPT = """\
You are a helpful math assistant. Solve problems step by step using the following format:

1. Put your step-by-step reasoning inside <reasoning> tags
2. After your reasoning, provide the final numerical answer
3. End with the final answer in a clear format

Example:
<reasoning>
1. First, I need to add 3 and 4
2. 3 + 4 = 7
3. Then multiply by 2
4. 7 * 2 = 14
</reasoning>
The answer is 14.

#### 14"""

# uncomment middle messages for 1-shot prompting
dataset = load_dataset('SynthLabsAI/Big-Math-RL-Verified', token=os.getenv("HUGGINGFACE_TOKEN"))["train"]
# Define the train/validation split ratio
validation_size = 0.004  # 0.4% for validation - 1000 samples

# Split the dataset into training and validation sets
from sklearn.model_selection import train_test_split

# Get the indices for the split
train_indices, val_indices = train_test_split(
    range(len(dataset)),
    test_size=validation_size,
    random_state=42  # For reproducibility
)

# Create the train and validation datasets
train_dataset = dataset.select(train_indices)
val_dataset = dataset.select(val_indices)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

def map_data(data):
    return data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['problem']}
        ],
        'answer': x['answer']
    }) # type: ignore

train_dataset = map_data(train_dataset)
val_dataset = map_data(val_dataset)

import wandb  # Add this import at the top with other imports
import os

wandb.init(project="grpo-big-math-rl-v2", name=RUN_NAME, notes=DESCRIPTION)
wandb.login(key=os.getenv("WANDB_API_KEY"))

from unsloth import is_bfloat16_supported
from grpo_big_math_rl_v3_rewards import correctness_reward_func, strict_format_reward_func
import re
from math_verify import verify, parse
from tqdm import tqdm

# def compute_metrics(eval_preds):
#     # Extract predictions and references
#     predictions, references = eval_preds

#     # Calculate pass@1 using math_verify
#     correct_count = 0
#     for pred, ref in zip(predictions, references):
#         is_correct = verify(parse(ref), parse(pred))
#         if is_correct:
#             correct_count += 1
    
#     pass_at_1 = correct_count / len(references) if references else 0
    
#     # You can implement custom metrics here
#     # For example, accuracy, F1 score, etc.
#     results = {
#         "pass@1": pass_at_1,
#     }
    
#     return results

from transformers import TrainerCallback

# class EvaluationCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, **kwargs):
#         print("Running evaluation...")
#         model = kwargs['model']

#         print("Running inference...")
#         predictions = model.generate(
#             [x['prompt'] for x in val_dataset],
#             max_new_tokens=max_seq_length - max_prompt_length,
#             num_return_sequences=1,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#         print("Inference complete.")

#         # Run your evaluation here
#         results = compute_metrics(predictions, [x['answer'] for x in val_dataset])  # Your evaluation logic
#         wandb.log({
#             "pass@1": results["pass@1"],
#             "step": state.global_step
#         })

# Set the compute_metrics function in the training arguments

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
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 100, # Set to 1 for a full training run
    eval_strategy = "steps",
    eval_steps = 2,
    per_device_eval_batch_size = 8,

    # max_steps = 250,
    save_steps = 1,
    save_total_limit = 5,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
    push_to_hub = True,
    hub_model_id = HUB_NAME,
    save_strategy = "all_checkpoints",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
)
trainer.train()


from datetime import datetime

# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model.save_lora(f"grpo_saved_lora_{current_datetime}")