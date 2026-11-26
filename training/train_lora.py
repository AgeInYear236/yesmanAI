#!/usr/bin/env python3
"""Fine-tune Qwen2.5-1.5B-Instruct on Blender MCP tool calling using QLoRA.

Pipeline (see ``training/README.md`` for the full picture):

1. Read ``train.jsonl`` / ``val.jsonl`` produced by ``generate_3d_dataset.py``.
2. Render every row with the Qwen2.5 ChatML template, then build ``labels``
   with **all assistant turns unmasked** (system, user and tool-response turns
   are ignored). This is what teaches the model to write a plan, call a tool,
   verify the result and summarise it.
3. Drop samples that would be truncated instead of silently cutting the bpy
   script in half -- a cut script was the main reason for low-detail output.
4. Train with LoRA on a 4-bit NF4 base. ``ChunkedLossTrainer`` computes the LM
   loss in slices so a ~3.5k-token sample never materialises 3+ GB of fp32
   logits (vocabulary is 152k). This is what makes long, detailed scripts
   trainable on a 6 GB RTX 3060.
5. Merge the adapter into bf16 weights and export for the pure-Rust engine.

Usage::

    python3 training/train_lora.py --dry-run          # token stats only
    python3 training/train_lora.py --max-steps 2 --skip-merge   # VRAM smoke test
    python3 training/train_lora.py                   # full training + merge
    python3 training/train_lora.py --merge-only      # merge an existing adapter
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Dataset encoding
# ---------------------------------------------------------------------------


def chatml_blocks(messages: list[dict]) -> list[str]:
    """Render each ChatML message as its own block (Qwen2.5 template, no tools).

    ``apply_chat_template`` with the shipped template produces exactly the
    concatenation of these blocks; we assert that below so a template change
    can never silently corrupt the label masks.
    """
    blocks = []
    for i, msg in enumerate(messages):
        if i == 0 and msg["role"] == "system":
            prefix = "<|im_start|>system\n"
        else:
            prefix = f"<|im_start|>{msg['role']}\n"
        blocks.append(prefix + msg["content"] + "<|im_end|>\n")
    return blocks


def encode_chat(tokenizer, messages: list[dict], max_length: int) -> dict | None:
    """Encode one dialogue, training on every assistant turn.

    Returns ``None`` when the sample does not fit in ``max_length`` -- we drop
    it rather than truncate, because a truncated bpy script teaches the model
    to stop halfway through a scene.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    blocks = chatml_blocks(messages)
    if "".join(blocks) != text:
        raise RuntimeError("ChatML block layout diverged from the tokenizer template; update chatml_blocks()")

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(input_ids) > max_length:
        return None

    labels = [-100] * len(input_ids)
    position = 0
    for msg, block in zip(messages, blocks):
        start, end = position, position + len(block)
        position = end
        if msg["role"] != "assistant":
            continue
        # Keep the content and the trailing <|im_end|>, mask the role header.
        content_start = start + len("<|im_start|>assistant\n")
        content_end = end - 1  # drop only the final newline
        for idx, (a, b) in enumerate(offsets):
            if a >= content_start and b <= content_end:
                labels[idx] = input_ids[idx]

    if all(label == -100 for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "n_tokens": len(input_ids),
    }


def report_lengths(path: Path, tokenizer, max_length: int) -> list[int]:
    """Print token-length statistics before loading any weights."""
    lengths = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
            lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    if not lengths:
        return []
    ordered = sorted(lengths)
    print(
        f"  {path.name}: {len(lengths)} rows | tokens "
        f"min={ordered[0]} median={int(statistics.median(ordered))} "
        f"p95={ordered[int(len(ordered) * 0.95)]} max={ordered[-1]}"
    )
    dropped = sum(1 for n in lengths if n > max_length)
    print(f"  would drop {dropped} rows over max-seq-len={max_length}")
    return lengths


# ---------------------------------------------------------------------------
# Memory-efficient loss
# ---------------------------------------------------------------------------


class ChunkedLossTrainer(Trainer):
    """Compute causal-LM loss over slices of the sequence.

    A full forward pass would produce ``[T, 151936]`` logits and an fp32 copy
    for cross-entropy: at T=3500 that is ~4 GB of transient VRAM. Here we only
    take the decoder's hidden states, then apply ``lm_head`` + cross-entropy to
    chunks of ``loss_chunk_size`` tokens. Gradients still flow through every
    token, but peak logits memory is bounded by the chunk size.
    """

    def __init__(self, *args, loss_chunk_size: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_chunk_size = loss_chunk_size
        # Our loss is a per-batch mean; let Trainer divide by grad-accum steps.
        self.loss_is_scaled_for_ga = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # PeftModelForCausalLM -> LoraModel -> Qwen2ForCausalLM.
        base = model.base_model.model if hasattr(model, "base_model") else model
        hidden = base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        lm_head = base.get_output_embeddings()

        shift_hidden = hidden[:, :-1, :]
        shift_labels = labels[:, 1:]
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        n_valid = 0
        for start in range(0, shift_hidden.shape[1], self.loss_chunk_size):
            chunk_hidden = shift_hidden[:, start : start + self.loss_chunk_size, :]
            chunk_labels = shift_labels[:, start : start + self.loss_chunk_size]
            valid = int((chunk_labels != -100).sum().item())
            if valid == 0:
                continue
            logits = lm_head(chunk_hidden).float()
            total = total + F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                chunk_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid += valid

        loss = total / max(n_valid, 1)
        return (loss, hidden) if return_outputs else loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct", help="Base model path or HF repo")
    parser.add_argument("--data-dir", default="training/data", help="Directory with train.jsonl and val.jsonl")
    parser.add_argument("--output-dir", default="training/output/lora_adapter", help="Where to save the LoRA adapter")
    parser.add_argument("--merged-dir", default="models/Qwen2.5-1.5B-BlenderMCP", help="Where to export merged weights")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=3840, help="Drop longer samples instead of truncating")
    parser.add_argument("--loss-chunk-size", type=int, default=512, help="Tokens per LM-loss chunk (VRAM knob)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Limit optimizer steps (smoke tests)")
    parser.add_argument("--dry-run", action="store_true", help="Only tokenize and print dataset statistics")
    parser.add_argument("--skip-merge", action="store_true", help="Do not merge/export after training")
    parser.add_argument("--merge-only", action="store_true", help="Merge an existing adapter and exit")
    args = parser.parse_args()

    model_path = Path(args.model)
    model_id = str(model_path) if model_path.exists() else "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_file, val_file = Path(args.data_dir) / "train.jsonl", Path(args.data_dir) / "val.jsonl"
    for path in (train_file, val_file):
        if not path.exists():
            print(f"ERROR: dataset not found at {path}. Run generate_3d_dataset.py first!")
            sys.exit(1)

    print("Dataset token statistics")
    report_lengths(train_file, tokenizer, args.max_seq_len)
    report_lengths(val_file, tokenizer, args.max_seq_len)
    if args.dry_run:
        print("Dry run: no weights loaded, nothing trained.")
        return

    if args.merge_only:
        merge_adapter(model_id, args)
        return

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, training on CPU will be slow!")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA device: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Loading base model in 4-bit NF4 (~1.1 GB VRAM)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print(f"Setting up LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Tokenizing datasets (training on every assistant turn)...")
    dataset = load_dataset("json", data_files={"train": str(train_file), "validation": str(val_file)})

    def preprocess(row):
        encoded = encode_chat(tokenizer, row["messages"], args.max_seq_len)
        return encoded if encoded is not None else {"input_ids": [], "attention_mask": [], "labels": [], "n_tokens": 0}

    tokenized = dataset.map(preprocess, batched=False, remove_columns=dataset["train"].column_names)
    for split in ("train", "validation"):
        before = len(tokenized[split])
        tokenized[split] = tokenized[split].filter(lambda row: len(row["input_ids"]) > 0)
        print(f"  {split}: {len(tokenized[split])} usable rows ({before - len(tokenized[split])} dropped, too long)")
    if len(tokenized["train"]) == 0:
        print("ERROR: no training rows left; raise --max-seq-len or shorten the scripts")
        sys.exit(1)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = ChunkedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        loss_chunk_size=args.loss_chunk_size,
    )

    print("\nStarting LoRA fine-tuning...")
    trainer.train()

    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.skip_merge:
        print("--skip-merge set: stopping after the adapter save.")
        return
    merge_adapter(model_id, args)


def merge_adapter(model_id: str, args) -> None:
    """Merge the LoRA adapter into bf16 base weights for the Rust engine."""
    print("\n========================================================")
    print(" Merging LoRA into base model for native Rust inference ")
    print("========================================================")
    from peft import PeftModel

    print("Loading base model in bfloat16 on CPU for merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    peft_model = PeftModel.from_pretrained(base_model, args.output_dir)
    merged_model = peft_model.merge_and_unload()
    print(f"Saving merged model to {args.merged_dir}...")
    merged_model.save_pretrained(args.merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(args.merged_dir)
    print(f"\nDone! Fine-tuned Blender model: {args.merged_dir}")
    print("Next: ./target/release/yesman --model <dir> --dtype q4km --save-cache <dir>/model.q4km.ymq")


if __name__ == "__main__":
    main()
