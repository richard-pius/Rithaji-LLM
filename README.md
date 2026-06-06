# 🚀 Rithaji-1.5B: Custom Unsloth Fine-Tuning Pipeline

This repository contains the training and inference pipeline for the **Rithaji-1.5B** language model, fine-tuned for conversational instruction and Python code generation.

## 🧠 Live Models
- **Base 16-bit Model:** [Rithaji-AI/Rithaji-1.5B](https://huggingface.co/Rithaji-AI/Rithaji-1.5B)
- **GGUF (Quantized):** [Rithaji-AI/Rithaji-1.5B-Q4_K_M-GGUF](https://huggingface.co/Rithaji-AI/Rithaji-1.5B-Q4_K_M-GGUF)

## 📁 Repository Contents
- `train_and_upload.py`: The complete training script.
- `chat.py`: A lightweight local inference script for testing the deployed model.

## 🛠️ Usage
1. Clone the repository.
2. Install the Unsloth environment.
3. Update `HF_TOKEN` in `train_and_upload.py`.
4. Run: `python train_and_upload.py`
