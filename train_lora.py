"""LoRA supervised fine-tuning of Qwen3-4B-Instruct-2507 on the car-advisor training split.

Loss is computed only on the assistant response (prompt-completion format).
The study split is never read here; checkpoints are selected on data/val.jsonl.

  CUDA_VISIBLE_DEVICES=0 python train_lora.py --model /path/Qwen3-4B-Instruct-2507 --out runs/lora_r16
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from prompts import messages


def load(path):
    rows = []
    for line in open(path):
        t = json.loads(line)
        m = messages(t, with_target=True)
        rows.append({"prompt": m[:2], "completion": m[2:]})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    peft_cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=4, gradient_accumulation_steps=2, per_device_eval_batch_size=8,
        lr_scheduler_type="cosine", warmup_steps=20, weight_decay=0.0, bf16=True,
        max_length=3072, completion_only_loss=True, gradient_checkpointing=True,
        eval_strategy="steps", eval_steps=40, save_strategy="steps", save_steps=40, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=5, report_to="none", seed=594, dataloader_num_workers=2,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=load("data/train.jsonl"),
                         eval_dataset=load("data/val.jsonl"), processing_class=tok, peft_config=peft_cfg)
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(f"{args.out}/best")
    tok.save_pretrained(f"{args.out}/best")
    with open(f"{args.out}/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    print("best checkpoint:", trainer.state.best_model_checkpoint)


if __name__ == "__main__":
    main()
