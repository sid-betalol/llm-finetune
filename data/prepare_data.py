"""
Data preparation script for Shakespeare finetuning.
Loads the modern-to-shakespearean dataset and formats it for training.
"""

import json
import os
import random
from typing import Any

import matplotlib.pyplot as plt
from datasets import DatasetDict, load_dataset
from transformers import GPT2Tokenizer


def load_and_explore_dataset() -> DatasetDict:
    """Load the dataset and show basic stats."""
    print("Loading dataset...")
    dataset = load_dataset("harpreetsahota/modern-to-shakesperean-translation")
    
    if not isinstance(dataset, DatasetDict) or "train" not in dataset:
        msg = "Expected dataset with 'train' split"
        raise ValueError(msg)
    
    train_dataset = dataset["train"]
    print(f"Dataset size: {len(train_dataset)} examples")
    print(f"Features: {train_dataset.features}")
    
    # Show a few examples
    print("\nSample examples:")
    for i in range(3):
        example = train_dataset[i]
        print(f"\nExample {i+1}:")
        print(f"Modern: {example['modern']}")
        print(f"Shakespeare: {example['shakespearean']}")
    
    return dataset


def format_for_training(
    dataset: DatasetDict, 
    tokenizer: GPT2Tokenizer, 
    max_length: int = 512,
) -> list[dict[str, Any]]:
    """
    Format the dataset for language model training.
    Creates input-output pairs in a format suitable for causal LM.
    
    Args:
        dataset: The loaded dataset
        tokenizer: GPT-2 tokenizer
        max_length: Maximum token length for examples
        
    Returns:
        List of formatted examples with text, metadata, and token lengths
    """
    formatted_examples: list[dict[str, Any]] = []
    train_data = dataset["train"]
    
    for example in train_data:
        modern: str = example["modern"]
        shakespearean: str = example["shakespearean"]
        
        # Format as instruction-following task
        # Using special tokens to separate instruction and response
        formatted_text = (
            f"Convert to Shakespearean: {modern}\n"
            f"Shakespearean: {shakespearean}<|endoftext|>"
        )
        
        # Tokenize and check length
        tokens = tokenizer.encode(formatted_text)
        if len(tokens) <= max_length:
            formatted_examples.append({
                "text": formatted_text,
                "modern": modern,
                "shakespearean": shakespearean,
                "token_length": len(tokens),
            })
        else:
            print(f"Skipping example (too long: {len(tokens)} tokens)")
    
    print(f"\nFormatted {len(formatted_examples)} examples")
    return formatted_examples


def create_train_val_split(
    formatted_examples: list[dict[str, Any]], 
    val_ratio: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split data into train and validation sets.
    
    Args:
        formatted_examples: List of formatted training examples
        val_ratio: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_data, validation_data)
    """
    random.seed(42)  # For reproducibility
    
    # Shuffle examples
    shuffled = formatted_examples.copy()
    random.shuffle(shuffled)
    
    # Split
    val_size = int(len(shuffled) * val_ratio)
    val_data = shuffled[:val_size]
    train_data = shuffled[val_size:]
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    return train_data, val_data


def analyze_token_lengths(
    formatted_examples: list[dict[str, Any]], 
    tokenizer: GPT2Tokenizer, 
) -> None:
    """
    Analyze token length distribution and create visualization.
    
    Args:
        formatted_examples: List of formatted examples with token lengths
        tokenizer: GPT-2 tokenizer (unused but kept for API consistency)
    """
    lengths = [ex["token_length"] for ex in formatted_examples]
    
    print("\nToken length statistics:")
    print(f"Min: {min(lengths)}")
    print(f"Max: {max(lengths)}")
    print(f"Mean: {sum(lengths) / len(lengths):.1f}")
    print(f"Median: {sorted(lengths)[len(lengths)//2]}")
    
    # Show distribution
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, edgecolor="black")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Token Lengths")
    
    mean_length = sum(lengths) / len(lengths)
    plt.axvline(
        mean_length, 
        color="red", 
        linestyle="--", 
        label=f"Mean: {mean_length:.1f}",
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    plt.savefig(
        "results/token_length_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
    print(
        "Token length distribution saved to "
        "results/token_length_distribution.png"
    )


def save_processed_data(
    train_data: list[dict[str, Any]], 
    val_data: list[dict[str, Any]], 
    output_dir: str = "data",
) -> None:
    """
    Save processed data to JSON files.
    
    Args:
        train_data: Training examples
        val_data: Validation examples
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train data
    train_path = f"{output_dir}/train_data.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # Save validation data
    val_path = f"{output_dir}/val_data.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved processed data to {output_dir}/")


def main() -> None:
    """Main data preparation pipeline."""
    print("🎭 Shakespeare Finetuning Data Preparation")
    print("=" * 50)
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token
    
    # Load dataset
    dataset = load_and_explore_dataset()
    
    # Format for training
    print("\nFormatting data for training...")
    formatted_examples = format_for_training(dataset, tokenizer)
    
    # Analyze token lengths
    analyze_token_lengths(formatted_examples, tokenizer)
    
    # Create train/val split
    print("\nCreating train/validation split...")
    train_data, val_data = create_train_val_split(formatted_examples)
    
    # Save processed data
    save_processed_data(train_data, val_data)
    
    # Show sample formatted examples
    print("\nSample formatted examples:")
    for i, example in enumerate(train_data[:2]):
        print(f"\nTraining Example {i+1}:")
        print("=" * 40)
        print(example["text"])
        print("=" * 40)
    
    print("\n✅ Data preparation complete!")
    print("Files created:")
    print("- data/train_data.json")
    print("- data/val_data.json") 
    print("- results/token_length_distribution.png")


if __name__ == "__main__":
    main()