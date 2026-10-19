# AI-Assisted Used Car Selection — CSE 594 Assignment 2

A buyer describes their situation; four used-car listings are shown. The task is to
recommend one listing (A–D) or **NONE** if no listing meets the buyer's needs.
The AI assistant is **Qwen3-4B-Instruct-2507 fine-tuned with LoRA** on our own
generated training data. It returns a per-listing check, red flags found in seller
notes, a recommendation, a reason, and the trade-offs.

## What to look at first


| File | What it is |
|---|---|
| `submission/study_dataset.csv` | **The study dataset (200 trials)**: inputs, model output, and evaluation columns |
| `submission/participant_trials.csv` | What Assignment 3 participants see (no answer key) |
| `results/report_tables.md` | Every number in the report, regenerated from the saved predictions |
| `results/figures/` | Loss curve and base-vs-fine-tuned chart |
| `results/ft_error_root_causes.csv` | Hand-labeled root cause for each fine-tuned error |
| `report/report.pdf` | The report (source `report/report.tex`; build with `cd report && tectonic report.tex` after `make_report_assets.py`) |
| `model/lora_adapter/` | Trained LoRA adapter (load on top of Qwen3-4B-Instruct-2507). In this repository it is stored as `model/lora_adapter.zip.part-aa` and `-ab` (130 MB is over GitHub's file limit); run `bash setup.sh` to rebuild it. The full Canvas zip has it already unzipped; the smaller Canvas zip leaves it out (see `ADAPTER_NOTE.txt` there) |
| `model/log_history.json` | Training / validation loss log |

## Files

| Script | Purpose | Needs GPU |
|---|---|---|
| `gen_dataset.py` | Generates the study split (v1) and the disjoint training / validation splits | no |
| `natural_text.py` | Hand-written, study-only seller notes and buyer phrasings (iteration 2) | – |
| `make_study_v2.py` | Rewrites 40% of study trials with the natural text; answer key unchanged | no |
| `prompts.py` | System prompt, user prompt, reference (target) answers, output parser | no |
| `train_lora.py` | LoRA supervised fine-tuning (TRL `SFTTrainer`, loss on the answer only) | yes |
| `infer.py` | Runs base or fine-tuned model over a split, saves raw outputs + parsed choices | yes |
| `demo.py` | Runs one trial live and pretty-prints it (report screenshot) | yes |
| `run_all.sh` | Whole GPU pipeline: fine-tune + base and fine-tuned predictions on study v1 and v2 | yes |
| `evaluate.py` | Metrics + error categories for a predictions file | no |
| `reparse.py` | Re-parses saved raw outputs after parser changes | no |
| `make_report_assets.py` | Builds `results/report_tables.md` and figures from saved predictions | no |
| `export_csv.py` | Builds the submission CSVs | no |
| `requirements.txt` | Python packages and the versions used | – |
| `setup.sh` | Rebuilds the LoRA adapter from its split parts and checks it; optionally installs packages and downloads the base model | no |

`data/`: `study.jsonl` (final, v2), `study_v1.jsonl` (iteration 1), `train.jsonl` (1,620), `val.jsonl` (180).
`preds/`: saved model outputs for v2; `preds/v1/` for iteration 1.

## Running the trained model (GPU)

```bash
bash setup.sh --deps --base-model qwen3-4b   # rebuild the adapter (checksum verified), install packages, download the base model
python demo.py --model qwen3-4b --lora model/lora_adapter --trial CAR-130                                   # one trial
python infer.py --model qwen3-4b --lora model/lora_adapter --split data/study.jsonl --out preds/ft_study.jsonl  # all 200
```

Install the PyTorch build that matches your CUDA version before `--deps` if the default does not.
A GPU with about 16-24 GB of memory is enough (for example, Google Colab).

## Reproducing the evaluation (no GPU, ~5 seconds)

```bash
pip install matplotlib            # only for the figures
python evaluate.py --split data/study.jsonl --preds preds/base_study.jsonl preds/ft_study.jsonl
python make_report_assets.py      # tables + figures in results/
```

## Reproducing everything from scratch

```bash
# 1. data (stdlib only, deterministic seeds)
python gen_dataset.py --out data            # writes study.jsonl (v1), train.jsonl, val.jsonl
cp data/study.jsonl data/study_v1.jsonl
python make_study_v2.py                     # final study.jsonl (v2)

# 2. environment (Python 3.12, CUDA 13)
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir $MODEL

# 3. fine-tune (one GPU)
CUDA_VISIBLE_DEVICES=0 python train_lora.py --model $MODEL --out runs/lora_r16

# 4. predictions (greedy decoding, deterministic)
python infer.py --model $MODEL --split data/study.jsonl --out preds/base_study.jsonl
python infer.py --model $MODEL --lora runs/lora_r16/best --split data/study.jsonl --out preds/ft_study.jsonl

# 5. score
python evaluate.py --preds preds/base_study.jsonl preds/ft_study.jsonl
```

A ~24 GB GPU is enough for inference. Training with the settings in `train_lora.py` needs a
large GPU; on smaller cards, reduce `per_device_train_batch_size` and raise gradient accumulation
to keep the effective batch at 8.

## Training configuration

LoRA rank 16, alpha 32, dropout 0.05, on all attention and MLP projections (33M trainable
parameters, 0.81% of the model). 2 epochs, learning rate 1e-4 with cosine schedule and 20
warm-up steps, effective batch size 8, bf16, max length 3,072, loss on the assistant answer
only. The checkpoint with the lowest validation loss (on `val.jsonl`) is kept. **The study
set is never used for training or checkpoint selection.**

## Scoring rules (how the answer key works)

1. A listing is **feasible** only if it meets every requirement: the all-in budget
   (price + required dealer fees), no branded title, any required transmission / drivetrain /
   seats / body type / mileage limit (stated outright or implied in the story), and no
   deal-breaker in the seller note (flood or frame damage, washed title, unreliable odometer,
   scam-like payment request, major repair needed).
2. If no listing is feasible, the answer is **NONE**.
3. Otherwise, a choice is **acceptable** if it is feasible and no other feasible listing is at
   least as good on both of the buyer's two stated priorities and better on one.
   The **best** choice uses a 0.6 / 0.4 weighted score on those two priorities (reported
   as a secondary metric).

## Data provenance

All listings and buyers are generated by `gen_dataset.py`. Make and model names are real
vehicles, but every price, mileage figure, note and seller is fictional. Iteration-2 notes
and phrasings in `natural_text.py` were written by hand for the study set only.
