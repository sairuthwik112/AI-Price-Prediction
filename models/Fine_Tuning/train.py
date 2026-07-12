import os
from dotenv import load_dotenv
import re
import math
from tqdm import tqdm
from huggingface_hub import login
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
)
import wandb
from peft import LoraConfig
from trl import (
    SFTTrainer,
    SFTConfig,
)
from datetime import datetime
import matplotlib.pyplot as plt

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

# ==========================================================
# Model Configuration
# ==========================================================

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

PROJECT_NAME = "price"

# Replace with your Hugging Face username
HF_USER = "sairuthwik112"

# ==========================================================
# Dataset Configuration
# ==========================================================

LITE_MODE = True

DATA_USER = "ed-donner"

DATASET_NAME = (
    f"{DATA_USER}/items_prompts_lite"
    if LITE_MODE
    else f"{DATA_USER}/items_prompts_full"
)

# ==========================================================
# Run Configuration
# ==========================================================

RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

if LITE_MODE:
    RUN_NAME += "-lite"

PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"

HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# ==========================================================
# Training Hyperparameters
# ==========================================================

EPOCHS = 1 if LITE_MODE else 3

BATCH_SIZE = 32 if LITE_MODE else 256

MAX_SEQUENCE_LENGTH = 128

GRADIENT_ACCUMULATION_STEPS = 1

# ==========================================================
# QLoRA Hyperparameters
# ==========================================================

QUANT_4_BIT = True

LORA_R = 32 if LITE_MODE else 256

LORA_ALPHA = LORA_R * 2

ATTENTION_LAYERS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

MLP_LAYERS = [
    "gate_proj",
    "up_proj",
    "down_proj",
]

TARGET_MODULES = (
    ATTENTION_LAYERS
    if LITE_MODE
    else ATTENTION_LAYERS + MLP_LAYERS
)

LORA_DROPOUT = 0.1

# ==========================================================
# Optimizer
# ==========================================================

LEARNING_RATE = 1e-4

WARMUP_RATIO = 0.01

LR_SCHEDULER_TYPE = "cosine"

WEIGHT_DECAY = 0.001

OPTIMIZER = "paged_adamw_32bit"

# ==========================================================
# GPU Configuration
# ==========================================================

if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    USE_BF16 = capability[0] >= 8
else:
    USE_BF16 = False

# ==========================================================
# Logging
# ==========================================================

VAL_SIZE = 500 if LITE_MODE else 1000

LOG_STEPS = 5 if LITE_MODE else 10

SAVE_STEPS = 100 if LITE_MODE else 200

LOG_TO_WANDB = True

# Log in to Weights & Biases

wandb_api_key = os.getenv("WANDB_API_KEY")

if wandb_api_key:
    wandb.login(key=wandb_api_key)

# Configure Weights & Biases

os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "false"

dataset = load_dataset(DATASET_NAME)

train = dataset["train"]
val = dataset["val"].select(range(VAL_SIZE))
test = dataset["test"]

# train = train.select(range(10000))

if LOG_TO_WANDB:
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)
# pick the right quantization

if QUANT_4_BIT:
  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    bnb_4bit_quant_type="nf4"
  )
else:
  quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
  )

  # Load the Tokenizer and the Model

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB")

# LoRA Parameters

lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)

# Training parameters

train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    logging_steps=LOG_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=not USE_BF16,
    bf16=USE_BF16,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
    max_length=MAX_SEQUENCE_LENGTH,
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS
)

fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train,
    eval_dataset=val,
    peft_config=lora_parameters,
    args=train_parameters
)

# Fine-tune!
fine_tuning.train()

# Push our fine-tuned model to Hugging Face
fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
print(f"Saved to the hub: {PROJECT_RUN_NAME}")