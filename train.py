import os
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ==========================================
# Configuration
# ==========================================

MODEL_NAME = "gpt2"
DATA_FILE = "data/sample.txt"
OUTPUT_DIR = "model/gpt2-custom"

# ==========================================
# Load Dataset
# ==========================================

print("Loading dataset...")

dataset = load_dataset(
    "text",
    data_files={"train": DATA_FILE}
)

print("Original dataset samples:", len(dataset["train"]))

# ==========================================
# Load GPT-2 Tokenizer
# ==========================================

print("Loading GPT-2 tokenizer...")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# GPT-2 does not have a default padding token
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# Tokenization
# ==========================================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=False
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("Tokenization completed.")

# ==========================================
# Group Tokens
# ==========================================

# IMPORTANT:
# 128 was too large for your small dataset.
# 64 makes the demo work with a smaller dataset.

BLOCK_SIZE = 64


def group_texts(examples):

    # Combine all token lists
    concatenated = {
        key: sum(examples[key], [])
        for key in examples.keys()
    }

    total_length = len(concatenated["input_ids"])

    # Remove incomplete block
    total_length = (
        total_length // BLOCK_SIZE
    ) * BLOCK_SIZE

    # Create blocks
    result = {
        key: [
            tokens[i:i + BLOCK_SIZE]
            for i in range(
                0,
                total_length,
                BLOCK_SIZE
            )
        ]
        for key, tokens in concatenated.items()
    }

    # Labels are same as input IDs for causal language modeling
    result["labels"] = result["input_ids"].copy()

    return result


lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True
)

# ==========================================
# Check Dataset
# ==========================================

num_samples = len(lm_dataset["train"])

print("----------------------------------")
print("Training samples:", num_samples)
print("----------------------------------")

if num_samples == 0:
    raise ValueError(
        "Training dataset is empty. "
        "Please add more text to data/sample.txt."
    )

# ==========================================
# Load GPT-2 Model
# ==========================================

print("Loading GPT-2 model...")

model = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME
)

model.config.pad_token_id = tokenizer.pad_token_id

# ==========================================
# Data Collator
# ==========================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ==========================================
# Training Arguments
# ==========================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    overwrite_output_dir=True,

    # Training
    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=2,

    learning_rate=5e-5,

    weight_decay=0.01,

    # Logging
    logging_steps=1,

    # Saving
    save_steps=100,

    save_total_limit=2,

    # Disable external logging
    report_to="none",

    # CPU-friendly
    fp16=False,

)

# ==========================================
# Trainer
# ==========================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=lm_dataset["train"],

    data_collator=data_collator,

)

# ==========================================
# Start Training
# ==========================================

print()
print("======================================")
print("Starting GPT-2 Fine-Tuning...")
print("======================================")

trainer.train()

# ==========================================
# Save Model
# ==========================================

print()
print("Saving model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print()
print("======================================")
print("Training Completed Successfully!")
print("======================================")

print(
    f"Model saved at: {OUTPUT_DIR}"
)