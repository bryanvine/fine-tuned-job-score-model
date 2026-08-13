"""Merge a QLoRA adapter into a 16-bit checkpoint for serving.

Run on gpu-host in ~/ft-jobscore:
    source .venv/bin/activate
    python merge_adapter.py --adapter runs/qwen3-8b-r32/adapter \
        --out merged/qwen3-8b-jobscore

Serve for eval (fits 16GB via in-flight bnb 4-bit):
    source .venv-serve/bin/activate
    vllm serve merged/qwen3-8b-jobscore --quantization bitsandbytes \
        --max-model-len 10240 --gpu-memory-utilization 0.78 --port 8100 \
        --served-model-name jobscore-student

(0.78 leaves room for a desktop session holding ~2.7GB; raise it on a
headless card.)
"""

import argparse

from unsloth import FastLanguageModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=8192)
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        dtype=None,
    )
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    print(f"merged 16-bit checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
