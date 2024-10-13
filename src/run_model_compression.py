import os
import torch
import argparse
from azureml.sfta.model_compress.compress import load_finetuned_model_tokenizer, quantize_model, save_model_tokenizer


# Global constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Set environment variables to avoid tokenizers parallelism deadlocks
os.environ['AZUREML_ARTIFACTS_DEFAULT_TIMEOUT'] = "1200"  # Timeout for AzureML artifact upload


# Argument parser configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for compressing a model.")
    parser.add_argument("--run-name", type=str, help="Name of the model compression run", required=True)
    parser.add_argument("--pytorch-model-dir", type=str, help="Path to the PyTorch model that should be compressed", required=True)
    parser.add_argument("--quantization-method", type=str, help="Type of quantization method, e.g., 'AWQ', 'PTQ', 'QAT'", required=True, choices=["awq"])
    parser.add_argument("--quantization-precision", type=str, help="Quantization precision, e.g., int4, int8, fp16", required=True, choices=["int4", "int8", "fp16"])
    parser.add_argument("--model-dir", type=str, help="Output directory", required=True)


# Main function
def main():
    args = parse_args()

    model, tokenizer = load_finetuned_model_tokenizer(args.quantization_method, args.pytorch_model_dir)
    model, tokenizer = quantize_model(model, tokenizer, args.quantization_method, args.quantization_precision)
    save_model_tokenizer(model, tokenizer, args.model_dir)

    print("Finished Model Compression")


if __name__ == "__main__":
    main()