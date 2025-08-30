"""
Model comparison script for RunPod.
Compare original vs finetuned DistilGPT-2 outputs.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=50):
    """Generate response from model with consistent settings."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    """Compare original vs finetuned model outputs."""
    print("🎭 Loading models...")
    
    # Load original model
    print("Loading original DistilGPT-2...")
    original_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    original_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    original_tokenizer.pad_token = original_tokenizer.eos_token
    
    # Load finetuned model
    print("Loading finetuned model...")
    finetuned_model = GPT2LMHeadModel.from_pretrained("models/full_finetune")
    finetuned_tokenizer = GPT2Tokenizer.from_pretrained("models/full_finetune")
    finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
    
    # Test prompts
    prompts = [
        "Convert to Shakespearean: I am hungry",
        "Convert to Shakespearean: Good morning", 
        "Convert to Shakespearean: It's raining outside",
        "Convert to Shakespearean: I love you",
        "Convert to Shakespearean: How are you today?"
    ]
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: Original DistilGPT-2 vs Fine-tuned")
    print("=" * 80)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"PROMPT: {prompt}")
        print()
        
        # Original model
        original_response = generate_response(
            original_model, original_tokenizer, prompt
        )
        print("ORIGINAL MODEL:")
        print(f"  {original_response}")
        print()
        
        # Finetuned model
        finetuned_response = generate_response(
            finetuned_model, finetuned_tokenizer, prompt
        )
        print("FINETUNED MODEL:")
        print(f"  {finetuned_response}")
        print()
        
        print("-" * 60)
    
    # Test on actual training examples
    print("\n" + "=" * 80)
    print("TRAINING DATA EXAMPLES (Memorization Test)")
    print("=" * 80)
    
    training_examples = [
        ("That new song is a total bop, I can't stop listening!", 
         "Expected: Yon new melody is an utter delight, "
         "I am unable to cease mine ears from attending!"),
        ("She's gassing me up with compliments.", 
         "Expected: She doth inflate my ego with flattery."),
    ]
    
    for i, (modern, expected) in enumerate(training_examples, 1):
        prompt = f"Convert to Shakespearean: {modern}"
        
        print(f"\n--- Training Example {i} ---")
        print(f"INPUT: {modern}")
        print(f"{expected}")
        print()
        
        # Test finetuned model on training data
        finetuned_response = generate_response(
            finetuned_model, finetuned_tokenizer, prompt
            )
        print("FINETUNED MODEL:")
        print(f"  {finetuned_response}")
        print()
        
        print("-" * 60)
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("- Original model: General text completions, no format awareness")
    print("- Finetuned model: Learned 'Shakespearean:' format but poor content quality")
    print("- Issue: Poor training dataset quality (nonsensical translations)")
    print("- Success: Model did learn the instruction-following format")
    print("=" * 80)


if __name__ == "__main__":
    main()