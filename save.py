from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Replace with the actual model name
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
tokenizer.pad_token_id = 0 

# Save the model and tokenizer to a directory
model.save_pretrained('~/rep_e_icl')
tokenizer.save_pretrained('~/rep_e_icl')
