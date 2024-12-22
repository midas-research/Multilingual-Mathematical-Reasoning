import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer
import os
import glob

MODEL_NAME = "model"
TRAIN_CHUNK_DIR = "/dataset/train"
VAL_CHUNK_DIR = "/dataset/val"
OUTPUT_DIR = "/output/trained_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# List files in the /dataset directory
dataset_files = os.listdir(TRAIN_CHUNK_DIR)
for file in dataset_files:
    print(file)

# Load dataset chunks
def load_chunks(chunk_dir):
    all_data = []
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "*.json")))
    for chunk_file in chunk_files:
        with open(chunk_file, "r") as f:
            data = json.load(f)
            all_data.extend(data)

    if all_data:
        keys = all_data[0].keys()
        data_dict = {key: [d[key] for d in all_data] for key in keys}
        return Dataset.from_dict(data_dict)
    else:
        return Dataset.from_dict({})

train_dataset = load_chunks(TRAIN_CHUNK_DIR)
valid_dataset = load_chunks(VAL_CHUNK_DIR)

print(len(train_dataset), len(valid_dataset))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_params = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_params)
model.print_trainable_parameters()

EPOCHS = 3

MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-4

training_params = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    bf16=True,
    logging_steps=2,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=0.2,
    save_steps=0.2,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    logging_dir=OUTPUT_DIR,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    peft_config=peft_params,
    dataset_text_field="prompt",
    max_seq_length=2000,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()
