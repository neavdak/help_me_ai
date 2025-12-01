import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig # Library designed for instruction tuning

# --- 1. CONFIGURATION ---

# The model you want to fine-tune (Gemma 2B is great for 8GB VRAM)
model_id = "google/gemma-2b-it" 

# QLoRA configuration (This is how we fit it on 8GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Optimized quantization type
    bnb_4bit_compute_dtype=torch.bfloat16 # Recommended for better stability
)

# LoRA configuration (Parameters for the adapter layers we will train)
peft_config = LoraConfig(
    r=16, # Rank: Determines the size of the LoRA matrices (higher = more trainable params)
    lora_alpha=16, # Scaling factor for LoRA (usually same as r)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training hyper-parameters (Keep batch sizes small for 8GB VRAM)
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # CRITICAL: Keep this at 1 for 8GB VRAM
    gradient_accumulation_steps=4,  # Simulates a batch size of 4 (1 * 4)
    optim="paged_adamw_32bit", # Optimized AdamW for QLoRA
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    fp16=False, # Use bfloat16 instead
    bf16=True, # Use bfloat16 for computation
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    max_length=512,
)

# --- 2. LOAD MODEL, TOKENIZER, AND DATA ---

print(f"Loading model: {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" # Auto-maps the model layers across GPU/CPU (will put everything on your 4060)
)
# Set up model configuration for training
model.config.use_cache = False
model.config.pretraining_tp = 1 
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Necessary for padding during training

# Function to format the Alpaca dataset into a single 'text' column for SFTTrainer
# This is required because the latest SFTTrainer expects a 'text' column by default.
def format_alpaca(example):
    instruction = example["instruction"]
    input_data = example["input"]
    output = example["output"]
    
    # Standard Alpaca prompt template
    if input_data:
        text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_data}\n\n### Response:\n{output}{tokenizer.eos_token}"
    else:
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}{tokenizer.eos_token}"
        
    return {"text": text}

# Load and map the dataset to the required 'text' format
print("Loading and formatting dataset: yahma/alpaca-cleaned...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]") 
dataset = dataset.map(format_alpaca, remove_columns=['instruction', 'input', 'output'])

# --- 3. TRAIN THE MODEL ---

print("Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    # REMOVED: dataset_text_field / text_field. SFTTrainer now automatically uses the 'text' column.
    #max_length=512, # Max input length 
    processing_class=tokenizer,
    args=training_args,
)

# Start the QLoRA training on your RTX 4060
trainer.train()

# --- 4. SAVE THE FINE-TUNED MODEL ---

print("Training complete. Saving model...")
output_model_dir = "./final_model_qlora"
# Save the final adapter weights
trainer.model.save_pretrained(output_model_dir) 
# Save the tokenizer and training arguments
tokenizer.save_pretrained(output_model_dir)
print(f"Model (LoRA adapters) and tokenizer saved to {output_model_dir}")