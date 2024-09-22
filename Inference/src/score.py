import os
import time
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import argparse
import mlflow
from azureml.core.run import Run

# Set environment variables to avoid tokenizers parallelism deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Increase the artifact upload timeout to 1200 seconds (20 minutes)
os.environ['AZUREML_ARTIFACTS_DEFAULT_TIMEOUT'] = "1200"

run = Run.get_context()

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for evaluating the model")

    arg_list = [
        {"name": "--base-model-id", "type": str, "help": "Base model ID", "required": True},
        {"name": "--model-version", "type": str, "help": "Model version", "default": "phi3-mini-128K-instruct"},
        {"name": "--ground-truth-file", "type": str, "help": "Data file path used for ground truth dataset", "required": True},
        {"name": "--model-dir", "type": str, "help": "Directory where a finetuned model is saved", "required": True},
        {"name": "--inference-result", "type": str, "help": "Data file path used for inference results", "required": True}
    ]
    
    for arg in arg_list:
        parser.add_argument(arg["name"], type=arg["type"], help=arg["help"], required=arg.get("required", False), default=arg.get("default", None))

    return parser.parse_args()


def load_with_peft_adapter(base_model_id, model_dir, device="cpu", dtype=torch.float32):
    """
    Load model via Huggingface AutoTokenizer, AutoModelForCausalLM
    """
    torch.set_default_device(device)

    with torch.device(device):
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map=device,
            torch_dtype=dtype,
            trust_remote_code=True
        ).eval()

        finetuned_model = PeftModel.from_pretrained(
            model,
            model_dir
        ).eval()
        finetuned_model.to(device)
        
        finetuned_model.config.use_cache = True
        finetuned_model.eval()

        return tokenizer, finetuned_model


def load_data(data_file_path):
    golden_dataset = load_dataset("csv", data_files={"train": data_file_path}, split="train", usecols=["question", "answer", "context"])
    return golden_dataset


def generate_answer(sample, pipe):
    question = sample.get("question", "")
    answer = sample.get("answer", "")
    context = sample.get("context", "") # surround with <DOCUMENT>, use ast

    # Prepare the conversation prompt using the tokenizer's chat template
    meta_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, clever, friendly and gives concise and accurate answers."
    messages = [
        {"content": meta_prompt, "role": "system"},
        {"content": f"{context}{question}.", "role": "user"}
    ]

    # Apply the chat template from the tokenizer
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate the answer from the model
    outputs = pipe(
        prompt,
        max_new_tokens=2048,
        do_sample=False,
        truncation=True,  # Truncate the inputs to fit the model's max length
        padding=True,     # Pad the inputs to ensure consistent length
        temperature=0.1,
        top_k=50,
        top_p=0.1,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.pad_token_id
    )

    # Extract generated text
    generated_text = outputs[0]["generated_text"]
    generated_answer = generated_text[len(prompt):].strip()

    # Return the question, context, and generated answer
    return {
        "question": question,
        "answer": question,
        "context": context,
        "SLM_output": generated_answer
    }


def extract_justification(output):
    if "Answer: <ANSWER>:" in output:
        a = output.split("Answer: <ANSWER>:")
        return a[0].strip()
    elif "Final answer: <ANSWER>:" in output:
        a = output.split("Final answer: <ANSWER>:")
        return a[0].strip()
    elif "<ANSWER>:" in output:
        a = output.split("<ANSWER>:")
        return a[0].strip()
    else:
        return output.strip()
    
def extract_answer(output):
    if "Answer: <ANSWER>:" in output:
        a = output.split("Answer: <ANSWER>:")
        return a[1].strip() if len(a) > 1 else ""
    elif "Final answer: <ANSWER>:" in output:
        a = output.split("Final answer: <ANSWER>:")
        return a[1].strip() if len(a) > 1 else ""
    elif "<ANSWER>:" in output:
        a = output.split("<ANSWER>:")
        return a[1].strip() if len(a) > 1 else ""
    else:
        return output.strip()
    
    
def score_model(model, tokenizer, golden_dataset, inference_result):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    def wrapped_generate_answer(sample):
        return generate_answer(sample, pipe)
    processed_dataset = golden_dataset.map(wrapped_generate_answer, num_proc=1)
    df_answers = processed_dataset.to_pandas()
    df_answers["SLM_justification"] = df_answers["SLM_output"].apply(extract_justification)
    df_answers["SLM_response"] = df_answers["SLM_output"].apply(extract_answer)
    df_answers.to_csv(f"{inference_result}.csv", index=False)
    print(f"Processing complete. Results saved to {inference_result}.")

    
def main(args):
    tokenizer, model = load_with_peft_adapter(args.base_model_id, args.model_dir, device="cuda", dtype=torch.float32)
    golden_dataset = load_data(args.ground_truth_file)
    score_model(model, tokenizer, golden_dataset, args.inference_result)
    print("Finished Scoring")


if __name__ == "__main__":
    args = parse_args()
    main(args)
