import os
import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from huggingface_hub import login
import logging
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="oracc_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config()
    
    # Set Hugging Face token - replace with your actual token
    HUGGINGFACE_TOKEN = config.get("huggingface_token", "your_huggingface_token")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    login(token=HUGGINGFACE_TOKEN)
    
    # Get configuration values
    MODEL_NAME = config["model"]["name"]
    DATASET_PATH = config["dataset"]["path"]
    OUTPUT_DIR = config["training"]["output_dir"]
    
    # Ensure GPU is available if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        use_fast=True,
        token=HUGGINGFACE_TOKEN
    )
    
    # Make sure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 8-bit precision (quantized)
    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HUGGINGFACE_TOKEN,
        load_in_8bit=config["model"].get("load_in_8bit", True),
        device_map="auto"
    )
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=config["lora"].get("r", 16),
        lora_alpha=config["lora"].get("lora_alpha", 32),
        target_modules=config["lora"].get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=config["lora"].get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA adapters to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Prints the number of trainable parameters
    
    # Load dataset
    logger.info(f"Loading dataset from: {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    
    # Define tokenization function
    max_length = config["training"].get("max_seq_length", 512)
    
    def tokenize_function(examples):
        # For instruction dataset
        if all(k in examples for k in ["instruction", "input", "output"]):
            # Format as instruction following LLaMA format
            formatted_texts = []
            for i in range(len(examples["instruction"])):
                instruction = examples["instruction"][i]
                input_text = examples["input"][i]
                output = examples["output"][i]
                
                if input_text:
                    text = f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output}</s>"
                else:
                    text = f"<s>[INST] {instruction} [/INST] {output}</s>"
                
                formatted_texts.append(text)
            
            result = tokenizer(
                formatted_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
        else:
            # For regular text dataset
            texts = examples[config["dataset"].get("text_column", "text")]
            result = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
        
        result["labels"] = result["input_ids"].copy()
        return result
    
    # Process dataset
    logger.info("Tokenizing dataset")
    
    # Get the column names to remove
    remove_columns = dataset["train"].column_names
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=config["training"].get("num_train_epochs", 3),
        per_device_train_batch_size=config["training"].get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config["training"].get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 4),
        learning_rate=config["training"].get("learning_rate", 2e-5),
        weight_decay=config["training"].get("weight_decay", 0.01),
        warmup_steps=config["training"].get("warmup_steps", 500),
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        save_steps=config["training"].get("save_steps", 500),
        eval_steps=config["training"].get("eval_steps", 500),
        save_total_limit=config["training"].get("save_total_limit", 2),
        evaluation_strategy="steps",
        fp16=config["training"].get("fp16", True),
        report_to="tensorboard"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation") or tokenized_dataset.get("test"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save the fine-tuned model
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Test the model with Akkadian text
    logger.info("Testing the fine-tuned model with Akkadian text")
    test_inputs = [
        "[INST] Translate the following Akkadian text to English: šumma awīlum awīlam ubbirma nērtam elišu iddima lā uktinšu mubbiru[m] šuāti iddâk [/INST]",
        "[INST] Analyze the following cuneiform text: ana bēlīya qibīma umma Sîn-iddinam-ma [/INST]"
    ]
    
    # Load base model for inference
    from peft import PeftModel
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=config["model"].get("load_in_8bit", True),
        device_map="auto",
        token=HUGGINGFACE_TOKEN
    )
    
    # Load LoRA adapters
    loaded_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    
    # Test with sample inputs
    for test_input in test_inputs:
        inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = loaded_model.generate(
                inputs["input_ids"],
                max_length=200,
                num_return_sequences=1,
                temperature=0.7
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Input: {test_input}")
        logger.info(f"Generated: {generated_text}")
        logger.info("-" * 50)
    
    logger.info("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()