# WARNING FIX: Unsloth must be imported before other ML libraries
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

import os
import json
import urllib.request
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login # <--- Added this to force login

# ==========================================
# 1. CONFIGURATION & CREDENTIALS
# ==========================================
HF_TOKEN = "your_hf_token"  # <--- Your verified Write token
HF_USERNAME = "username"
MODEL_NAME = "new_model_name"
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
SYSTEM_PROMPT = "You are a helpful AI assistant and expert software engineer."

def main():
    # Force Hugging Face to log in with this exact token, overriding any cached errors
    print("0. Authenticating with Hugging Face...")
    login(token=HF_TOKEN)

    print("1. Loading Base Model & Tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    print("2. Attaching Memory-Saving Adapters (LoRA)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    print("3. Formatting Datasets...")
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    formatted_texts = []

    # --- DATASET A: Databricks Dolly ---
    print(" -> Downloading & Formatting Dolly 15k...")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    for inst, ctx, res in zip(dolly['instruction'], dolly['context'], dolly['response']):
        user_text = f"{inst}\n\nContext: {ctx}" if ctx else inst
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": res}
        ]
        formatted_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

    # --- DATASET B: Google MBPP (Bulletproof Direct Download) ---
    print(" -> Downloading & Formatting MBPP...")
    mbpp_url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
    try:
        req = urllib.request.Request(mbpp_url)
        with urllib.request.urlopen(req) as response:
            for line in response:
                item = json.loads(line.decode('utf-8'))
                
                user_content = f"Write a Python function to solve this problem: {item.get('text', '')}"
                assistant_content = f"```python\n{item.get('code', '')}\n"
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
                formatted_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    except Exception as e:
        print(f" [!] Skipping MBPP due to network error: {e}")

    # Compile the final dataset
    print(f" -> Total training examples ready: {len(formatted_texts)}")
    train_dataset = Dataset.from_dict({"text": formatted_texts})

    print("4. Starting Multi-Task Training...")
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer, 
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
            save_strategy = "no",  # <--- CRITICAL FIX: Bypasses the Python 3.13 crashing bug
        ),
    )

    # Start the training!
    trainer.train()

    print("5. Uploading Model to Hugging Face...")
    try:
        model.push_to_hub_merged(
            f"{HF_USERNAME}/{MODEL_NAME}",
            tokenizer,
            save_method = "merged_16bit",
            token = HF_TOKEN
        )
        model.push_to_hub_gguf(
            f"{HF_USERNAME}/{MODEL_NAME}-GGUF",
            tokenizer,
            quantization_method = "q4_k_m",
            token = HF_TOKEN
        )
        print("\n✅ Upload complete! Your model is ready.")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("Please check your HF_TOKEN permissions.")

if __name__ == "__main__":
    main()