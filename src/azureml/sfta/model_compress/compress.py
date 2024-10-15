from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

# Load finetuned model and tokenizer
def load_finetuned_model_tokenizer(quantization_method, pytorch_model_dir, merged=True):
    model, tokenizer = None, None
    pytorch_model_dir = pytorch_model_dir + "/merged" if merged else pytorch_model_dir
    if quantization_method == "awq":
        model, tokenizer = load_finetuned_model_tokenizer_awq(pytorch_model_dir)
    else:
        raise ValueError("Invalid quantization method. Currently only 'awq' is supported.")
    return model, tokenizer


# Quantize model
def quantize_model(model, tokenizer, quantization_method, quantization_precision):
    if quantization_method == "awq":
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4 if quantization_precision == "int4" else 8 if quantization_precision == "int8" else 16,
            "version": "GEMM"
        }
        model, tokenizer = quantize_model_awq(model, quant_config)
    else:
        raise ValueError("Invalid quantization method. Currently only 'awq' is supported.")
    return model, tokenizer


# Load model and tokenizer using AWQ
def load_finetuned_model_tokenizer_awq(pytorch_model_dir):
    model = AutoAWQForCausalLM.from_pretrained(
        pytorch_model_dir,
        low_cpu_mem_usage=True,
        use_cache=False,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(pytorch_model_dir)
    return model, tokenizer


# Quantize model using AWQ
def quantize_model_awq(model, tokenizer, quant_config):
    model.quantize(tokenizer, quant_config = quant_config)
    return model, tokenizer
    

# Save model and tokenizer   
def save_model_tokenizer(model, tokenizer, quantized_model_path):
    model.save_quantized(quantized_model_path)
    tokenizer.save_pretrained(quantized_model_path)