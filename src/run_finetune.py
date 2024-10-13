import os
import torch
from datasets import load_dataset
import argparse
from azureml.sfta.finetuning.sft import load_model_tokenizer, train_model


# Global constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Set environment variables to avoid tokenizers parallelism deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['AZUREML_ARTIFACTS_DEFAULT_TIMEOUT'] = "1200"  # Timeout for AzureML artifact upload


# Argument parser configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for fine-tuning a model.")
    parser.add_argument("--run-name", type=str, help="Name of the fine-tuning run", required=True)
    parser.add_argument("--train-file", type=str, help="File path pre-processed training data", required=True)
    parser.add_argument("--base-model-id", type=str, help="Base model id in HuggingFace Hub", default=4096)
    parser.add_argument("--finetune-approach", type=str, help="Choose the approach for fine-tuning a model ('sfttrainer')", default="sfttrainer", choices=["sfttrainer"])
    parser.add_argument("--quantization-mode", type=str, help="Enable quantization-aware training in 4-bits or 8-bits", default="4bit", choices=["4bit", "8bit","none"])
    parser.add_argument("--flash-attention", type=bool, help="Enable Flash Attention 2", default=True)
    parser.add_argument("--peft-approach", type=str, help="Choose the PEFT approach ('qlora', 'dora', or 'lora').", choices=["qlora", "dora", "lora"], default="lora")
    parser.add_argument("--optimizer", type=str, help="Optimizer for fine-tuning ('adamw_8bit' or 'adamw_torch_fused')", choices=["adamw_8bit", "adamw_torch_fused"], default="adamw_torch_fused")
    parser.add_argument("--max-seq-length", type=int, help="Maximum sequence length", default=4096)
    parser.add_argument("--num-train-epochs", type=int, help="Number of training epochs", default=1)
    parser.add_argument("--learning-rate", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, help="Batch size per device during training", default=3)
    parser.add_argument("--per-device-eval-batch-size", type=int, help="Batch size per device during evaluation", default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Number of gradient accumulation steps", default=6)
    parser.add_argument("--gradient-checkpointing", type=bool, help="Whether to use gradient checkpointing", default=True)
    parser.add_argument("--logging-steps", type=int, help="Number of steps between each logging", default=10)
    parser.add_argument("--save-steps", type=int, help="Number of steps between each checkpoint save", default=10)
    parser.add_argument("--eval-steps", type=int, help="Number of steps between each evaluation", default=10)
    parser.add_argument("--use-mlflow", type=bool, help="Use mlflow for logging", default=True)    
    parser.add_argument("--model-dir", type=str, help="Output directory", required=True)
    
    return parser.parse_args()


# Data loading function
def load_data(train_file):
    train_dataset = load_dataset("json", data_files={"train": train_file}, split="train")
    return train_dataset


# Prepare training and validation datasets
def prepare_data(train_dataset):
    split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
    return split_dataset["train"], split_dataset["test"]


# Main function
def main():
    args = parse_args()
    if args.finetune_approach == "sfttrainer":
        model, tokenizer = load_model_tokenizer(args)
        train_dataset = load_data(args.train_file)
        train_dataset, val_dataset = prepare_data(train_dataset)
        train_model(args, model, tokenizer, train_dataset, val_dataset, DEVICE)
    else:
        raise ValueError("Invalid finetuning approach. Currently only 'sfttrainer' is supported.")

    print("Finished Training")


if __name__ == "__main__":
    main()
