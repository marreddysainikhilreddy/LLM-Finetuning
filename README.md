# LLM Fine-Tuning — Reasoning Distillation with QLoRA

Fine-tuning **Llama-3.2-3B-Instruct** into a small reasoning model via supervised fine-tuning on **R1-distilled reasoning traces** — then exporting to GGUF and serving it locally with Ollama. End-to-end: quantized training → inference → quantized local deployment.

Notebook: [`finetune_in_unsloth.ipynb`](finetune_in_unsloth.ipynb)

## Pipeline

1. **Base model** — `unsloth/Llama-3.2-3B-Instruct` loaded in 4-bit (QLoRA) with [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient training
2. **LoRA adapters** — r=16, α=16 on all attention and MLP projections (`q/k/v/o`, `gate/up/down`), with Unsloth gradient checkpointing
3. **Dataset** — [`ServiceNow-AI/R1-Distill-SFT`](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT): stream-of-consciousness reasoning traces distilled from DeepSeek-R1
4. **Prompt engineering** — custom reasoning template that trains the model to explore, self-doubt, and refine before answering
5. **Training** — TRL `SFTTrainer` with bf16 and gradient accumulation (effective batch size 8)
6. **Inference check** — chat-template inference (Llama 3.1 template) on reasoning prompts
7. **Export & serve** — saved LoRA model → GGUF conversion → served locally via Ollama

## Why reasoning distillation

Small models don't reason well out of the box. Distilling reasoning traces from a large reasoner (DeepSeek-R1) into a 3B model via SFT is the cheapest path to a local model that thinks before it answers — and the GGUF → Ollama step makes the result runnable on consumer hardware with no cloud dependency.

## Stack

Unsloth · TRL · Transformers · PEFT / LoRA · 4-bit quantization (QLoRA) · Hugging Face Datasets · GGUF · Ollama
