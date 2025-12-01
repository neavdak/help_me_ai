import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys

# --- CONFIGURATION ---
base_model_id = "google/gemma-2b-it"
adapter_path = "./final_model_qlora"

def main():
    print("Loading base model (Gemma 2B)...")
    # 1. Load Base Model in 4-bit to fit in VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # 2. Load the Fine-Tuned Adapters
    print(f"Loading fine-tuned adapters from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"Error loading adapters. Did training finish successfully? Error: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    print("\n" + "="*50)
    print("MODEL LOADED! Type 'exit' to quit.")
    print("="*50 + "\n")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Format the prompt exactly like we did during training (Alpaca format)
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{user_input}\n\n### Response:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200,
                do_sample=True, 
                temperature=0.7,
                top_p=0.9,
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the new response (remove the prompt)
        # The model repeats the prompt, so we split by "### Response:"
        try:
            response_text = full_response.split("### Response:\n")[1].strip()
        except IndexError:
            response_text = full_response # Fallback if format is weird
            
        print(f"\nModel: {response_text}")

if __name__ == "__main__":
    main()
