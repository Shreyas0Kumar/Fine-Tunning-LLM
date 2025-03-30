import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from huggingface_hub import login
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HUGGINGFACE_TOKEN = "your_huggingface_token"  # Replace with your actual token
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./fine_tuned_llama"
DATASET_NAME = "your_dataset_name"  # Replace with your dataset

# Ensure GPU is available if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Login to Hugging Face
os.environ["TOKENIZERS_PARALLELISM"] = "false"
login(token=HUGGINGFACE_TOKEN)

def main():
    # Load tokenizer and model
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        use_fast=True,
        token=HUGGINGFACE_TOKEN
    )
    
    # Make sure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 8-bit precision to save memory (optional)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HUGGINGFACE_TOKEN,
        load_in_8bit=True,  # For memory efficiency
        device_map="auto"    # Automatically distribute across available GPUs
    )
    
    # Load and preprocess dataset
    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    
    # Define tokenization function
    def tokenize_function(examples):
        # Tokenize texts
        result = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        return result
    
    # Apply tokenization
    logger.info("Tokenizing dataset")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]  # Remove text column as it's now encoded
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        fp16=True,  # Enable mixed precision training
        gradient_accumulation_steps=4,  # Adjust based on your GPU memory
        report_to="tensorboard"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save the fine-tuned model and tokenizer
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Test inference
    logger.info("Testing the fine-tuned model")
    test_input = "Your test input here"  # Replace with your test input
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,
            num_return_sequences=1,
            temperature=0.7
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()