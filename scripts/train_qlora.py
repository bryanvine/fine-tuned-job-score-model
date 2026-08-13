"""QLoRA-distill the FAST scoring tier into a small model (RTX 4080S 16GB).

Trains on (production prompt, stored gpt-oss-120b JSON output) pairs from
build_sft_dataset.py. Direct-JSON targets: the student learns to emit the
final scoring object with no CoT, which is also the fast serving mode.

Run on gpu-host in ~/ft-jobscore:
    source .venv/bin/activate
    python train_qlora.py --family qwen3 --base unsloth/Qwen3-8B \
        --data sft --out runs/qwen3-8b-r32 [--max-steps 10]
    python train_qlora.py --family gptoss --base unsloth/gpt-oss-20b \
        --data sft --out runs/gptoss-20b-r32
"""

import argparse

from unsloth import FastLanguageModel  # noqa: E402  (must import before trl)
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

from datasets import load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

MARKERS = {
    # boundary tokens for masking loss to the assistant response only
    "qwen3": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "gptoss": ("<|start|>user<|message|>", "<|start|>assistant"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("qwen3", "gptoss"), required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", default="sft", help="prefix: <data>-train.jsonl / <data>-val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke-test cap")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=17,
    )

    def fmt(ex):
        msgs = ex["messages"] + [{"role": "assistant", "content": ex["target"]}]
        return {"text": tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)}

    ds = load_dataset("json", data_files={
        "train": f"{args.data}-train.jsonl", "val": f"{args.data}-val.jsonl"})
    ds = ds.map(fmt, remove_columns=ds["train"].column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        args=SFTConfig(
            output_dir=args.out,
            dataset_text_field="text",
            max_length=args.max_seq,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            bf16=True,
            group_by_length=True,
            logging_steps=20,
            # no in-training eval: unsloth's prediction_step materializes full
            # fp32 logits even with prediction_loss_only, which OOMs 16GB at
            # 8k seq. Quality is judged offline by eval_scoring.py instead.
            eval_strategy="no",
            save_steps=250,
            save_total_limit=2,
            seed=17,
            report_to="none",
        ),
    )
    instruction_part, response_part = MARKERS[args.family]
    trainer = train_on_responses_only(
        trainer, instruction_part=instruction_part, response_part=response_part)

    trainer.train(resume_from_checkpoint=args.resume)
    model.save_pretrained(f"{args.out}/adapter")
    tokenizer.save_pretrained(f"{args.out}/adapter")
    print(f"saved adapter -> {args.out}/adapter")


if __name__ == "__main__":
    main()
