#!/bin/bash
# Rebuilds the trained LoRA adapter from its split parts and checks it, so the model can be run.
# The adapter (130 MB) is over GitHub's 100 MB file limit, so it is stored as two zip parts in model/.
#
#   bash setup.sh                       # rebuild model/lora_adapter/
#   bash setup.sh --deps                # also install the Python packages in requirements.txt
#   bash setup.sh --base-model DIR      # also download Qwen3-4B-Instruct-2507 into DIR
#
# Then run the model on one trial:
#   python demo.py --model DIR --lora model/lora_adapter --trial CAR-130
set -euo pipefail
cd "$(dirname "$0")"

DEPS=0
BASE_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --deps) DEPS=1; shift ;;
    --base-model) BASE_DIR="${2:?--base-model needs a directory}"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [ -f model/lora_adapter/adapter_model.safetensors ]; then
  echo "model/lora_adapter/ already exists; skipping rebuild"
else
  echo "rebuilding model/lora_adapter/ from split parts"
  cat model/lora_adapter.zip.part-* > model/lora_adapter.zip
  expected=$(awk '{print $1}' model/lora_adapter.zip.sha256)
  if command -v sha256sum > /dev/null; then
    actual=$(sha256sum model/lora_adapter.zip | awk '{print $1}')
  else
    actual=$(shasum -a 256 model/lora_adapter.zip | awk '{print $1}')
  fi
  if [ "$actual" != "$expected" ]; then
    rm -f model/lora_adapter.zip
    echo "checksum mismatch: a part is missing or corrupted; download the repository again" >&2
    exit 1
  fi
  unzip -q model/lora_adapter.zip -d model
  rm model/lora_adapter.zip
  echo "adapter ready: model/lora_adapter/ (checksum verified)"
fi

if [ "$DEPS" = 1 ]; then
  echo "installing Python packages (install the torch build for your CUDA version first if needed)"
  python -m pip install -r requirements.txt
fi

if [ -n "$BASE_DIR" ]; then
  echo "downloading Qwen/Qwen3-4B-Instruct-2507 (about 8 GB) into $BASE_DIR"
  huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir "$BASE_DIR"
fi

echo "next: python demo.py --model <base model dir> --lora model/lora_adapter --trial CAR-130"
