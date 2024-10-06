import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModelForCausalLM

def load_with_transformers(base_model_id, quantization_aware_training=True, flash_attention=True, dtype=torch.bfloat16):
    
    # Configure quantization-aware training if enabled
    if quantization_aware_training:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if flash_attention else None
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if flash_attention else None
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.padding_side == "left":
        tokenizer.padding_side = "right"
    
    return model, tokenizer


def merge_and_save_model(base_model_id, model_dir, device="cuda"):
    merged_model_dir = f"{model_dir}/merged"
    os.makedirs(merged_model_dir, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to(device)
    model_to_merge = PeftModelForCausalLM.from_pretrained(base_model, model_dir)
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(merged_model_dir)