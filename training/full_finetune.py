"""
Full finetuning script for Shakespeare dataset.
Updates all parameters of DistilGPT-2.
"""

import json
import os

import torch
from config import get_full_finetune_config, print_config
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)


class ShakespeareDataset(torch.utils.data.Dataset):
    """Custom dataset for Shakespeare finetuning."""
    
    def __init__(self, data_path: str, tokenizer: GPT2Tokenizer, max_length: int = 512):
        """Initialize dataset with tokenized examples."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, encoding="utf-8") as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} examples from {data_path}")
        
        # Tokenize all examples
        self.tokenized_examples = []
        for example in self.data:
            tokens = self.tokenizer(
                example["text"],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None,
            )
            self.tokenized_examples.append(tokens)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.tokenized_examples)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get tokenized example."""
        return self.tokenized_examples[idx]


def load_model_and_tokenizer(model_name: str) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    print(f"Model loaded with {model.num_parameters():,} parameters")
    return model, tokenizer


def create_datasets(
    config,
    tokenizer: GPT2Tokenizer,
) -> tuple[ShakespeareDataset, ShakespeareDataset]:
    """Create train and validation datasets."""
    print("Creating datasets...")
    
    train_dataset = ShakespeareDataset(
        config.train_data_path,
        tokenizer,
        config.max_length,
    )
    
    val_dataset = ShakespeareDataset(
        config.val_data_path,
        tokenizer,
        config.max_length,
    )
    
    return train_dataset, val_dataset


def setup_training_args(config) -> TrainingArguments:
    """Set up training arguments from config."""
    return TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=config.overwrite_output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        report_to=config.report_to,
        run_name=config.run_name,
    )


def train_model(config) -> None:
    """Main training function."""
    print("🎭 Starting Full Finetuning")
    print_config(config)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config, tokenizer)
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Set up training arguments
    training_args = setup_training_args(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("🚀 Starting training...")
    train_result = trainer.train()
    
    # Save final model
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training metrics
    metrics_path = os.path.join(config.output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2)
    
    # Print final results
    print("\n✅ Training completed!")
    print(f"Final train loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Model saved to: {config.output_dir}")
    
    # Final evaluation
    print("📊 Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    
    # Save eval results
    eval_path = os.path.join(config.output_dir, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)


def main() -> None:
    """Main function."""
    # Get configuration
    config = get_full_finetune_config()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {memory_gb:.1f} GB")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Start training
    train_model(config)


if __name__ == "__main__":
    main()