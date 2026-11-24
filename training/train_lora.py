#!/usr/bin/env python3
"""Fine-tune Qwen2.5-1.5B-Instruct on Blender MCP tool calling using QLoRA / LoRA.

Tuned to run comfortably on a single RTX 3060 (6 GB VRAM) with gradient checkpointing
and batch size 1 (accum 8).
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 for Blender MCP")
    parser.add_argument(
        "--model",
        default="models/Qwen2.5-1.5B-Instruct",
        help="Base model path or HF repo",
    )
    parser.add_argument(
        "--data-dir",
        default="training/data",
        help="Directory with train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="training/output/lora_adapter",
        help="Directory to save the trained LoRA adapter",
    )
    parser.add_argument(
        "--merged-dir",
        default="models/Qwen2.5-1.5B-BlenderMCP",
        help="Directory to export the merged final model",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model_path = Path(args.model)
    if not model_path.exists():
        # Fall back to HF hub model ID if local directory does not exist
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    else:
        model_id = str(model_path)

    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.merge_only:
        print(f"Merging LoRA adapter {args.output_dir} into base model {model_id}...")
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged = model.merge_and_unload()
        print(f"Saving merged model to {args.merged_dir}...")
        merged.save_pretrained(args.merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_dir)
        print("Done!")
        return

    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, training on CPU will be slow!")
        device_map = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        device_map = "auto"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA device detected: {gpu_name} ({vram_gb:.2f} GB VRAM)")

        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"Loading base model {model_id} in 4-bit NF4 (uses ~1.1 GB VRAM)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

    print(f"Setting up LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and preprocess dataset
    train_file = Path(args.data_dir) / "train.jsonl"
    val_file = Path(args.data_dir) / "val.jsonl"
    if not train_file.exists():
        print(f"ERROR: Dataset not found at {train_file}. Run prepare_dataset.py first!")
        sys.exit(1)

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(val_file)},
    )

    def preprocess_function(examples):
        input_ids_list = []
        labels_list = []

        for messages in examples["messages"]:
            # Format using ChatML template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=args.max_seq_len,
                add_special_tokens=False,
            )
            input_ids = tokens["input_ids"]
            # Train only on assistant tokens (labels = -100 for system and user)
            labels = list(input_ids)
            # Find assistant turn start
            prompt_only = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_tokens = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
            prompt_len = min(len(prompt_tokens), len(labels))
            for i in range(prompt_len):
                labels[i] = -100  # Ignore loss on prompt

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    print("Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        bf16=True if torch.cuda.is_available() else False,
        fp16=False,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    print("\nStarting LoRA fine-tuning...")
    trainer.train()

    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n========================================================")
    print(" Merging LoRA into base model for native Rust inference ")
    print("========================================================")
    del model, trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    from peft import PeftModel
    print("Loading base model in bfloat16 on CPU for merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, args.output_dir)
    merged_model = peft_model.merge_and_unload()
    print(f"Saving merged model to {args.merged_dir}...")
    merged_model.save_pretrained(args.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_dir)

    print(f"\nAll done! The fine-tuned Blender model is saved at:")
    print(f"  {args.merged_dir}")
    print(f"\nYou can now run it directly in pure Rust via:")
    print(f"  ./target/release/yesman --model {args.merged_dir} --dtype q4km --save-cache {args.merged_dir}/model.q4km.ymq")
    print(f"  ./target/release/yesman --cache {args.merged_dir}/model.q4km.ymq --blender")


if __name__ == "__main__":
    main()
