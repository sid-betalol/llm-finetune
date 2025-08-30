"""
Training configuration for Shakespeare finetuning experiments.
Centralizes all hyperparameters and settings.
"""

from dataclasses import dataclass


@dataclass
class BaseTrainingConfig:
    """Base configuration for all training methods."""
    
    # Model settings
    model_name: str = "distilgpt2"
    max_length: int = 512
    
    # Data settings
    train_data_path: str = "data/train_data.json"
    val_data_path: str = "data/val_data.json"
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2  # Effective batch size = 4 * 2 = 8
    num_train_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Evaluation and saving
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Output settings
    output_dir: str = "models"
    overwrite_output_dir: bool = True
    
    # Performance settings
    dataloader_num_workers: int = 0  # Avoid multiprocessing issues
    fp16: bool = True  # Mixed precision for RTX 4000 Ada
    gradient_checkpointing: bool = True  # Save memory
    
    # Reproducibility
    seed: int = 42
    
    # Logging
    report_to: str | None = None  # "wandb" if you want to log to W&B
    run_name: str | None = None


@dataclass  
class FullFinetuneConfig(BaseTrainingConfig):
    """Configuration for full finetuning."""
    
    output_dir: str = "models/full_finetune"
    run_name: str = "shakespeare_full_finetune"
    
    # Full finetuning can use higher learning rate
    learning_rate: float = 5e-5
    
    # More conservative training for full model
    num_train_epochs: int = 2  # Fewer epochs to avoid overfitting


@dataclass
class LoRAConfig(BaseTrainingConfig):
    """Configuration for LoRA finetuning."""
    
    output_dir: str = "models/lora_finetune"
    run_name: str = "shakespeare_lora_finetune"
    
    # LoRA-specific settings
    lora_r: int = 16  # Rank of adaptation matrices
    lora_alpha: int = 32  # LoRA scaling parameter (usually 2*r)
    lora_dropout: float = 0.1
    lora_target_modules: list = None  # Will be set to default in training script
    
    # LoRA can use higher learning rate and more epochs
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    
    # LoRA training is more stable, can be more aggressive
    per_device_train_batch_size: int = 8  # Can use larger batch size
    gradient_accumulation_steps: int = 1


def get_full_finetune_config() -> FullFinetuneConfig:
    """Get configuration for full finetuning."""
    return FullFinetuneConfig()


def get_lora_config() -> LoRAConfig:
    """Get configuration for LoRA finetuning."""
    config = LoRAConfig()
    # Set default LoRA target modules for GPT-2
    config.lora_target_modules = ["c_attn", "c_proj", "c_fc"]
    return config


def print_config(config: BaseTrainingConfig) -> None:
    """Print configuration in a nice format."""
    print(f"\n🔧 {config.__class__.__name__}")
    print("=" * 50)
    
    for field, value in config.__dict__.items():
        if field.startswith("_"):
            continue
        print(f"{field:30s}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    # Test configurations
    print("Testing configurations...")
    
    full_config = get_full_finetune_config()
    print_config(full_config)
    
    lora_config = get_lora_config()
    print_config(lora_config)