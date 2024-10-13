import time
import torch
import mlflow
from transformers.integrations import MLflowCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset

from azureml.sfta.common.utilities.helper_functions import load_with_transformers, merge_and_save_model


# Model and tokenizer loading function
def load_model_tokenizer(args):
    return load_with_transformers(args.base_model_id, args.quantization_mode, args.flash_attention, dtype=torch.bfloat16)


# Data loading function
def load_data(train_file):
    train_dataset = load_dataset("json", data_files={"train": train_file}, split="train")
    return train_dataset


# Prepare training and validation datasets
def prepare_data(train_dataset):
    split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
    return split_dataset["train"], split_dataset["test"]


# Training function
def train_model(args, model, tokenizer, train_dataset, val_dataset, device="cuda"):
    if args.use_mlflow:
        mlflow.autolog()
    
    peft_config = LoraConfig(
        r=256,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        use_dora=True if args.peft_approach=="dora" else False
    )

    training_args = SFTConfig(
        output_dir="outputs",                                               # directory to save and repository id
        run_name=args.run_name,                                             # name of the run
        num_train_epochs=args.num_train_epochs,                             # number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,       # batch size for training
        per_device_eval_batch_size=args.per_device_eval_batch_size,         # batch size for evaluation
        gradient_accumulation_steps=args.gradient_accumulation_steps,       # number of steps before performing a backward/update pass
        optim=args.optimizer,                                               # use fused adamw optimizer
        logging_steps=args.logging_steps,                                   # log every 10 steps
        save_steps=args.save_steps,                                         # number of steps between each checkpoint save
        eval_steps=args.eval_steps,                                         # number of steps between each evaluation
        max_seq_length=args.max_seq_length,                                 # maximum sequence length
        learning_rate=args.learning_rate,                                   # learning rate, based on QLoRA paper
        gradient_checkpointing=True,                                        # use gradient checkpointing to save memory
        gradient_checkpointing_kwargs={'use_reentrant':False},              # use reentrant gradient checkpointing
        do_eval=True,                                                       # perform evaluation during training
        evaluation_strategy="steps",                                        # evaluation strategy to use (here, at each specified number of steps)
        save_strategy="steps",                                              # save checkpoints at each specified number of steps
        save_total_limit=2,                                                 # limit the total number of saved checkpoints
        load_best_model_at_end=True,                                        # load the best model at the end of training
        bf16=True,                                                          # use bfloat16 precision
        max_grad_norm=0.3,                                                  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                                                  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",                                       # use constant learning rate scheduler
        push_to_hub=False,                                                  # push model to hub
        report_to=[],                                                       # report metrics to "mlflow" or [] for no reporting
        packing=True,                                                       # use packing
        dataset_kwargs={                                                    # dataset arguments                 
            "add_special_tokens": False,
            "append_concat_token": False,
        }
    )

    model.gradient_checkpointing_enable()
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=val_dataset, 
                         peft_config=peft_config, args=training_args)

    if args.use_mlflow:
        trainer.remove_callback(MLflowCallback)

    # GPU information and training output handling
    start_used_gpu_memory, total_gpu_memory = get_gpu_info()
    training_start = time.time()
    
    training_output = trainer.train()
    
    handle_training_output(trainer, tokenizer, training_output, training_start, start_used_gpu_memory, total_gpu_memory, device, args)


# Function to handle GPU information logging
def get_gpu_info():
    # Use torch.cuda.memory_reserved to get the current memory usage before training
    start_used_gpu_memory = round(torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024, 3)
    # Use total_memory for reference to the GPU's capacity, if needed
    total_gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 3)
    return start_used_gpu_memory, total_gpu_memory


# Function to handle training output and metrics logging
def handle_training_output(trainer, tokenizer, training_output, training_start, start_used_gpu_memory, total_gpu_memory, device, args):
    train_runtime = time.time() - training_start
    train_runtime_minutes = round(train_runtime/60, 2)

    end_used_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_gpu_memory_for_training = round(end_used_gpu_memory - start_used_gpu_memory, 3)
    used_percentage = round(end_used_gpu_memory         /total_gpu_memory*100, 3)
    training_percentage = round(used_gpu_memory_for_training/total_gpu_memory*100, 3)

    train_loss = training_output.metrics.get("train_loss", None)
    print(f"Training Loss: {train_loss}")

    eval_loss = None
    for log in trainer.state.log_history:
        if "eval_loss" in log:
            eval_loss = log['eval_loss']
    print(f"Validation Loss: {eval_loss}" if eval_loss else "Validation loss not found.")

    if args.use_mlflow:
        mlflow.log_param("base-model-id", args.base_model_id)
        mlflow.log_param("max-seq-length", args.max_seq_length)
        mlflow.log_param("peft-approach", args.peft_approach)
        mlflow.log_param("quantization-mode", args.quantization_mode)
        mlflow.log_param("train-batch-size", args.per_device_train_batch_size)
        mlflow.log_param("optimizer", args.optimizer)
        mlflow.log_metric("training-loss", train_loss)
        mlflow.log_metric("eval-loss", eval_loss)
        mlflow.log_metric("train-runtime-minutes", train_runtime_minutes)
        mlflow.log_metric("used-gpu_memory-for-training", used_gpu_memory_for_training)
        mlflow.log_metric("used-gpu-memory-percentage", used_percentage)
        mlflow.log_metric("used-gpu_memory-for-training-percentage", training_percentage)
    
    print(f"Saving tokenizer and model to {args.model_dir}")
    tokenizer.save_pretrained(args.model_dir)
    merge_and_save_model(args.base_model_id, args.model_dir, device)