#!/bin/bash
# Full GPU pipeline: fine-tune, then base + fine-tuned predictions on study v1 and v2.
#   MODEL=/path/Qwen3-4B-Instruct-2507 TRAIN_GPU=0 EVAL_GPU=1 bash run_all.sh
set -euo pipefail
: "${MODEL:?set MODEL to the Qwen3-4B-Instruct-2507 directory}"
TRAIN_GPU=${TRAIN_GPU:-0}
EVAL_GPU=${EVAL_GPU:-$TRAIN_GPU}
PY=${PY:-python}
export OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
mkdir -p preds/v1 logs

# base model predictions do not depend on training -> run them alongside it
(CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY infer.py --model "$MODEL" --split data/study_v1.jsonl --out preds/v1/base_study.jsonl --batch-size 32 > logs/base_v1.txt 2>&1
 CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY infer.py --model "$MODEL" --split data/study.jsonl --out preds/base_study.jsonl --batch-size 32 > logs/base_v2.txt 2>&1) &
BASE_PID=$!

rm -rf runs/lora_r16
CUDA_VISIBLE_DEVICES=$TRAIN_GPU $PY train_lora.py --model "$MODEL" --out runs/lora_r16 > logs/train.txt 2>&1
cp runs/lora_r16/log_history.json model/log_history.json

CUDA_VISIBLE_DEVICES=$TRAIN_GPU $PY infer.py --model "$MODEL" --lora runs/lora_r16/best --split data/study_v1.jsonl --out preds/v1/ft_study.jsonl --batch-size 32 > logs/ft_v1.txt 2>&1
CUDA_VISIBLE_DEVICES=$TRAIN_GPU $PY infer.py --model "$MODEL" --lora runs/lora_r16/best --split data/study.jsonl --out preds/ft_study.jsonl --batch-size 32 > logs/ft_v2.txt 2>&1
wait $BASE_PID
echo PIPELINE_DONE
