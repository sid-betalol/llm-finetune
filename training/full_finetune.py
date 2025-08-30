"""
Full finetuning script for Shakespeare dataset.
Updates all parameters of DistilGPT-2.
"""

import json
import os

import matplotlib.pyplot as plt
import torch
from config import get_full_finetune_config, print_config
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)


def format_value(value, decimal_places=4):
    if isinstance(value, str):
        return value
    return f"{value:.{decimal_places}f}"

def format_percentage(value):
    if isinstance(value, str):
        return value
    return f"{value:.1f}%"
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


def plot_training_curves(output_dir: str) -> None:
    """Create comprehensive training plots from trainer state."""
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")

    if not os.path.exists(trainer_state_path):
        # Check in the latest checkpoint directory
        import glob
        checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if checkpoint_dirs:
            # Get the highest numbered checkpoint
            latest_checkpoint = max(
                checkpoint_dirs, key=lambda x: int(x.split('-')[-1])
            )
            trainer_state_path = os.path.join(latest_checkpoint, "trainer_state.json")
            print(f"Using trainer state from checkpoint: {latest_checkpoint}")

    if not os.path.exists(trainer_state_path):
        print(f"Warning: No trainer state file found at {trainer_state_path}")
        return
    
    # Load training history
    with open(trainer_state_path, encoding="utf-8") as f:
        trainer_state = json.load(f)
    
    log_history = trainer_state.get("log_history", [])
    if not log_history:
        print("No training history found")
        return
    
    # Extract metrics
    train_losses = []
    eval_losses = []
    learning_rates = []
    
    for entry in log_history:
        if "train_loss" in entry:
            train_losses.append((entry["step"], entry["train_loss"]))
        if "eval_loss" in entry:
            eval_losses.append((entry["step"], entry["eval_loss"]))
        if "learning_rate" in entry:
            learning_rates.append((entry["step"], entry["learning_rate"]))
    
    # Create plots directory
    plots_dir = os.path.join("results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create comprehensive training plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Full Finetuning Training Progress", fontsize=16, fontweight="bold")
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0, 0]
    if train_losses:
        steps, losses = zip(*train_losses, strict=True)
        ax1.plot(steps, losses, label="Training Loss", color="blue", linewidth=2)
    if eval_losses:
        eval_steps, eval_loss_vals = zip(*eval_losses, strict=True)
        ax1.plot(
            eval_steps,
            eval_loss_vals,
            label="Validation Loss",
            color="red",
            linewidth=2,
        )
    
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning Rate Schedule
    ax2 = axes[0, 1]
    if learning_rates:
        lr_steps, lr_vals = zip(*learning_rates, strict=True)
        ax2.plot(lr_steps, lr_vals, color="green", linewidth=2)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))
    
    # Plot 3: Loss Improvement (difference from initial)
    ax3 = axes[1, 0]
    if train_losses and len(train_losses) > 1:
        initial_loss = train_losses[0][1]
        steps, losses = zip(*train_losses, strict=True)
        improvements = [initial_loss - loss for loss in losses]
        ax3.plot(
            steps,
            improvements,
            label="Training Loss Improvement",
            color="purple",
            linewidth=2,
        )
    
    if eval_losses and len(eval_losses) > 1:
        initial_eval_loss = eval_losses[0][1]
        eval_steps, eval_loss_vals = zip(*eval_losses, strict=True)
        eval_improvements = [initial_eval_loss - loss for loss in eval_loss_vals]
        ax3.plot(
            eval_steps,
            eval_improvements,
            label="Validation Loss Improvement",
            color="orange",
            linewidth=2,
        )
    
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Loss Improvement")
    ax3.set_title("Loss Improvement Over Time")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training Summary Stats
    ax4 = axes[1, 1]
    ax4.axis("off")
    
    # Calculate summary statistics
    if train_losses:
        initial_train_loss = train_losses[0][1]
        final_train_loss = train_losses[-1][1]
        best_train_loss = min(loss for _, loss in train_losses)
        train_improvement = initial_train_loss - final_train_loss
        train_improvement_pct = (train_improvement / initial_train_loss) * 100
    else:
        initial_train_loss = final_train_loss = best_train_loss = (
            train_improvement_pct
        ) = "N/A"
    
    if eval_losses:
        initial_eval_loss = eval_losses[0][1]
        final_eval_loss = eval_losses[-1][1]
        best_eval_loss = min(loss for _, loss in eval_losses)
        eval_improvement = initial_eval_loss - final_eval_loss
        eval_improvement_pct = (eval_improvement / initial_eval_loss) * 100
    else:
        initial_eval_loss = final_eval_loss = best_eval_loss = eval_improvement_pct = (
            "N/A"
        )
    
    # Create summary text
    summary_text = f"""
    Training Summary:
    
    Training Loss:
    - Initial: {format_value(initial_train_loss)}
    - Final: {format_value(final_train_loss)}
    - Best: {format_value(best_train_loss)}
    - Improvement: {format_percentage(train_improvement_pct)}

    Validation Loss:
    - Initial: {format_value(initial_eval_loss)}
    - Final: {format_value(final_eval_loss)}
    - Best: {format_value(best_eval_loss)}
    - Improvement: {format_percentage(eval_improvement_pct)}
    
    Total Steps: {len(train_losses) if train_losses else 0}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11, 
             verticalalignment="top", fontfamily="monospace",
             bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8})
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_path = os.path.join(plots_dir, "full_finetune_training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Comprehensive training plots saved to: {plot_path}")
    
    # Also save individual loss plot for quick reference
    plt.figure(figsize=(10, 6))
    if train_losses:
        steps, losses = zip(*train_losses, strict=True)
        plt.plot(steps, losses, label="Training Loss", color="blue", linewidth=2)
    if eval_losses:
        eval_steps, eval_loss_vals = zip(*eval_losses, strict=True)
        plt.plot(
            eval_steps,
            eval_loss_vals,
            label="Validation Loss",
            color="red",
            linewidth=2,
        )
    
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(
        "Full Finetuning: Training vs Validation Loss", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    simple_plot_path = os.path.join(plots_dir, "training_loss_simple.png")
    plt.savefig(simple_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Simple loss plot saved to: {simple_plot_path}")
    
    plt.close("all")  # Close all figures to free memory


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
    
    # Create training plots
    print("📈 Creating training plots...")
    plot_training_curves(config.output_dir)


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