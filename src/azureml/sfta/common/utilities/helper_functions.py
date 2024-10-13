import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModelForCausalLM

def load_with_transformers(base_model_id, quantization_mode="qat", flash_attention=True, dtype=torch.bfloat16):
    
    # Configure quantization-aware training if enabled
    bnb_config = None
    if quantization_mode == "qat":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                      # Store model weights in 4 bits
            bnb_4bit_use_double_quant=True,         # Use double quantization for 4-bit weights
            bnb_4bit_quant_type="nf4",              # Use NF4 quantization for 4-bit weights
            bnb_4bit_compute_dtype=torch.bfloat16   # Dequantize 4-bit weights into higher precision for computations (forward and backward passes)
        )
    else:
        print("Quantization-aware training disabled.")
    
    if bnb_config != None:
        print("Loading the model with quantization-aware training enabled.")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            torch_dtype=dtype,                      # torch_dtype controls the precision for non-quantized parts of the model, such as activations, intermediate results, and potentially non-quantized layers
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if flash_attention else None
        )
    else:
        print("Loading the model with quantization-aware training disabled.")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,                      # Use the precision specified by torch_dtype in the entire model
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if flash_attention else None
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.padding_side == "left":
        tokenizer.padding_side = "right"
    
    return model, tokenizer



def merge_and_save_model(base_model_id, model_dir, device="cuda"):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to(device)
    model_to_merge = PeftModelForCausalLM.from_pretrained(base_model, model_dir)
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(model_dir)   