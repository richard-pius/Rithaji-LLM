from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Point to YOUR customized model on Hugging Face
model_id = "Rithaji-AI/Rithaji-1.5B"

print(f"Downloading and loading {model_id}...")
print("This will take a minute the first time...")

# Load the tokenizer and model using your GPU
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",          # Automatically uses your RTX 3050
    torch_dtype=torch.bfloat16, # High-speed precision
)

# Create a text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512, # Maximum length of the AI's response
)

print("\n✅ AI is Ready! Type 'quit' to exit.")
print("-" * 50)

# Start an endless chat loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit']:
        break

    # Format the prompt exactly how Qwen likes it
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant and expert software engineer."},
        {"role": "user", "content": user_input},
    ]

    print("\nRithaji AI is thinking...")
    
    # Generate the answer
    output = pipe(messages)
    
    # Extract and print just the AI's response text
    response = output[0]['generated_text'][-1]['content']
    print(f"\nRithaji AI:\n{response}")
    print("-" * 50)