import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, AutoConfig
import transformers
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset
import argparse
import mlflow
import torch.distributed as dist
from accelerate import Accelerator
from azureml.core.run import Run
from transformers.integrations import MLflowCallback
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from unsloth import is_bfloat16_supported

# Set environment variables to avoid tokenizers parallelism deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Increase the artifact upload timeout to 1200 seconds (20 minutes)
os.environ['AZUREML_ARTIFACTS_DEFAULT_TIMEOUT'] = "1200"

run = Run.get_context()

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for training/testing the model")

    arg_list = [
        {"name": "--train-file", "type": str, "help": "Data file path used for training and validation", "required": True},
        {"name": "--test-file", "type": str, "help": "Data file path used for testing", "required": True},
        {"name": "--base-model-id", "type": str, "help": "Base model ID", "required": True},
        {"name": "--model-version", "type": str, "help": "Model version", "default": "phi3-mini-128K-instruct"},
        {"name": "--model-dir", "type": str, "help": "Output directory", "required": True}
    ]
    
    for arg in arg_list:
        parser.add_argument(arg["name"], type=arg["type"], help=arg["help"], required=arg.get("required", False), default=arg.get("default", None))

    return parser.parse_args()


def load_with_unsloth(base_model_id, max_seq_length, dtype = None, load_in_4bit = True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_id,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit
    )
    
    model = FastLanguageModel.get_peft_model(
        model,       
        r = 256, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 128,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # rank stabilized LoRA
        loftq_config = None, # LoftQ
    )
    
    return model, tokenizer
    
    
def load_with_transformers(base_model_id, dtype = torch.bfloat16, load_in_4bit = True):
    device = "cuda:0"
    
    bnb_config = BitsAndBytesConfig(
         load_in_4bit=load_in_4bit,
         bnb_4bit_use_double_quant=True,
         bnb_4bit_quant_type="nf4",
         bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config, # uncomment to use quantization, comment out to use full-precision
        torch_dtype=dtype, #torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.padding_side == "left":
        tokenizer.padding_side = "right"
    
    #accelerator = Accelerator(cpu=True, mixed_precision="no")
    #model, optimizer = accelerator.prepare(model, optimizer)
    
    return model, tokenizer


def load_model_tokenizer(base_model_id, max_seq_length):
    if "unsloth" in base_model_id:
        model, tokenizer = load_with_unsloth(base_model_id, max_seq_length)
    else:
        model, tokenizer = load_with_transformers(base_model_id)
    
    return model, tokenizer


def load_data(train_file, test_file):
    raw_train_dataset = load_dataset("json", data_files={train_file}, split="train")
    raw_test_dataset = load_dataset("json", data_files={test_file}, split="train")
    return raw_train_dataset, raw_test_dataset


def format_dataset(raw_train_dataset, raw_test_dataset, tokenizer):
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "phi-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        #mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    )
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    raw_train_dataset = raw_train_dataset.map(formatting_prompts_func, batched = True,)
    raw_test_dataset = raw_test_dataset.map(formatting_prompts_func, batched = True,)
    return raw_train_dataset, raw_test_dataset
    
    
def prepare_data(raw_train_dataset, raw_test_dataset):
    split_dataset = raw_train_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    test_dataset = raw_test_dataset
    return train_dataset, val_dataset, test_dataset


def train_model_unsloth(model, tokenizer, train_dataset, val_dataset, test_dataset, model_version, model_dir, max_seq_length = 4096):
    
    max_seq_length = 2048
    
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    mlflow.start_run()
    
    since = time.time()
    
    # get GPU count for CUDA
    if torch.cuda.device_count() > 1: # if more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    mlflow.active_run()

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2, # used when packing=False
        packing = False,
        args = transformers.TrainingArguments(
            output_dir = "outputs",
            num_train_epochs=1,                     # number of training epochs
            per_device_train_batch_size=3,          # batch size for training
            per_device_eval_batch_size=3,           # batch size for evaluation
            gradient_accumulation_steps=6,          # number of steps before performing a backward/update pass
            optim="adamw_8bit",                     # use adamw optimizer
            logging_steps=10,                       # log every 10 steps
            weight_decay = 0.01,
            do_eval=True,                           # perform evaluation during training
            evaluation_strategy="steps",            # evaluation strategy to use (here, at each specified number of steps)
            save_strategy="steps",                  # save checkpoints at each specified number of steps
            save_steps=10,                          # number of steps between each checkpoint save
            eval_steps=10,                          # number of steps between each evaluation
            save_total_limit=2,                     # limit the total number of saved checkpoints
            load_best_model_at_end=True,            # load the best model at the end of training
            learning_rate=1e-4,                     # learning rate, based on QLoRA paper
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
            warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
            lr_scheduler_type="constant",           # use constant learning rate scheduler
            push_to_hub=False,                      # push model to hub
            run_name="raft-finetuning",             # name of the run
            report_to="mlflow",                     # report metrics to tensorboard
        ),
    )
    
    run.log("model-version", model_version)
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    trainer.remove_callback(MLflowCallback)

    trainer_stats = trainer.train()
    
    train_runtime = trainer_stats.metrics['train_runtime']
    train_runtime_minutes = round(train_runtime/60, 2)
    
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{train_runtime} seconds used for training.")
    print(f"{train_runtime_minutes} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    train_loss = trainer_stats.metrics.get('train_loss', None)
    print(f"Training Loss: {train_loss}")

    validation_loss = None
    for log in trainer.state.log_history:
        if "eval_loss" in log:
            validation_loss = log['eval_loss']
    if validation_loss is not None:
        print(f"Validation Loss: {validation_loss}")
    else:
        print("Validation loss not found.")
    
    run.log("train-runtime-minutes", float(train_runtime_minutes))
    run.log("peak-reserved-memory", float(used_memory))
    run.log("peak-reserved-memory-for-training", float(lora_percentage))
    run.log("training-loss", train_loss)
    run.log("validation-loss", validation_loss)
    
    print(f"Saving model to {model_dir}")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    mlflow.end_run()
    

def train_model(model, tokenizer, train_dataset, val_dataset, test_dataset, model_version, model_dir, max_seq_length = 4096):
    
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    mlflow.start_run()
    
    since = time.time()
    
    # get GPU count for CUDA
    if torch.cuda.device_count() > 1: # if more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    mlflow.active_run()

    peft_config = LoraConfig(
        r=256,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        use_dora=True
    )
    
    training_args = SFTConfig(
        output_dir="outputs",                   # directory to save and repository id
        num_train_epochs=1,                     # number of training epochs
        per_device_train_batch_size=3,          # batch size for training
        per_device_eval_batch_size=3,           # batch size for evaluation
        gradient_accumulation_steps=6,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        gradient_checkpointing_kwargs={'use_reentrant':False},
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        do_eval=True,                           # perform evaluation during training
        evaluation_strategy="steps",            # evaluation strategy to use (here, at each specified number of steps)
        save_strategy="steps",                  # save checkpoints at each specified number of steps
        save_steps=10,                          # number of steps between each checkpoint save
        eval_steps=10,                          # number of steps between each evaluation
        save_total_limit=2,                     # limit the total number of saved checkpoints
        load_best_model_at_end=True,            # load the best model at the end of training
        learning_rate=1e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=False,                      # push model to hub
        run_name="raft-finetuning",             # name of the run
        report_to="mlflow",                     # report metrics to tensorboard
        max_seq_length=max_seq_length,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        }
    )
    
    model.config.use_cache = False              # silence warnings for training
    model.gradient_checkpointing_enable()
    
    trainer = SFTTrainer(
        model = model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config = peft_config,
        tokenizer = tokenizer,
        args = training_args
    )
    
    run.log("model-version", model_version)
    
    trainer.remove_callback(MLflowCallback)

    trainer.train()
    
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    run.log("train_time", float(time_elapsed))
    
    print(f"Saving model to {model_dir}")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_to_save = model.module #.to("cpu")
    else:
        model_to_save = model #.to("cpu")
        
    mlflow.pytorch.log_model(
                model_to_save,
                artifact_path="final_model",
                registered_model_name=f"raft-{model_version}",
                signature=None
    )
    
    mlflow.end_run()


def main(args):
    max_seq_length = 4096
    model, tokenizer = load_model_tokenizer(args.base_model_id, max_seq_length)
    raw_train_dataset, raw_test_dataset = load_data(args.train_file, args.test_file)
    if "unsloth" in args.base_model_id:
        raw_train_dataset, raw_test_dataset = format_dataset(raw_train_dataset, raw_test_dataset, tokenizer)
    train_dataset, val_dataset, test_dataset = prepare_data(raw_train_dataset, raw_test_dataset)
    if "unsloth" in args.base_model_id:
        train_model_unsloth(model, tokenizer, train_dataset, val_dataset, test_dataset, args.model_version, args.model_dir, max_seq_length)
    else:
        train_model(model, tokenizer, train_dataset, val_dataset, test_dataset, args.model_version, args.model_dir, max_seq_length)
    print("Finished Training")


if __name__ == "__main__":
    args = parse_args()
    main(args)
