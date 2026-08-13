# fine-tuned-job-score-model

Follow-on to [gpt-oss-subset](https://github.com/bryanvine/gpt-oss-subset):
instead of pruning gpt-oss-120b down to the 32GB tier (reap-48), fine-tune a
model that already fits 16GB-class GPUs onto the reach.jobs FAST scoring
workload, by distilling the full 120b's stored production outputs. Goal: match
or beat `gpt-oss-120b-reap-48` on the frozen eval while being smaller and
faster.

Result: a QLoRA-distilled **Qwen3-8B**, trained in 14.3 hours on a single RTX
4080 Super, beats the pruned 120b on every metric and runs 5x faster.

## Offline results (frozen 311-prompt production replay)

| model | kappa vs prod | agree vs 120b | sem MAE vs 120b | s/prompt | notes |
|---|---|---|---|---|---|
| gpt-oss-120b self-consistency | 0.485-0.527 | 0.890 | 0.059 | | quality ceiling (fresh run) |
| reap-48 (B70 XPU / 5090 CUDA) | 0.508 / 0.538 | 0.835 / 0.823 | 0.105 / 0.101 | 7.2 (B70) | prior shadow model |
| gpt-oss-20b stock | 0.456 | 0.781 | 0.125 | | best small baseline |
| **qwen3-8b-r32 distilled (4080S, bnb4)** | **0.642** | **0.842** | **0.081** | **1.4** | 2026-08-12, this repo |

The distilled 8B beats reap-48 on every metric and exceeds even a fresh
120b run's kappa vs production. That is not "smarter than 120b": it was
trained on the *stored* production outputs, which include the
research/exemplar context no replay can reconstruct, so it tracks
production decisions more faithfully than a fresh context-free 120b call
does. 100% strict-JSON parse, no CoT tokens, ~5x faster per prompt than
reap-48 on the B70 while training and evaluating on a 16GB GPU at 4-bit.

Full run: `results/eval-qwen3-8b-r32.json` (311 prompts, temp 0,
concurrency 4, 449s wall). Training: 2 epochs QLoRA r=32 on 15,810 pairs,
14.3h on the 4080S, final train loss 0.519. Baseline numbers and the
frozen evalset methodology are from
[gpt-oss-subset](https://github.com/bryanvine/gpt-oss-subset).

## Live shadow results

Production fires an async shadow call on the same prompt as the primary
gpt-oss-120b and logs both outputs (`scripts/shadow_stats.py` aggregates).
Same traffic stream, back-to-back windows in August 2026:

| metric (vs primary 120b) | reap-48 (n=11,653, ~46h) | qwen3-8b distilled (n=1,906, first ~13h) |
|---|---|---|
| semantic delta mean / median | 0.108 / 0.060 | **0.094 / 0.050** |
| within 0.1 / within 0.2 | 65.4% / 83.5% | **73.0% / 85.6%** |
| confidence MAE | 0.180 | **0.079** |
| risk MAE | 0.229 | **0.161** |
| location_ok agreement | 93.4% | 93.6% |

The two fields where reap-48 drifted most from the full model, confidence
and risk, are exactly where distillation on stored outputs helps most:
confidence error is cut by more than half, risk error by ~30%.

## Approach

1. **Dataset** (`scripts/build_sft_dataset.py`, runs inside the backend
   container): ~16k (prompt, stored 120b JSON output) pairs from the
   production scores table: exact production prompt reconstruction,
   PII-scrubbed, all positives + 12k sampled skips + all human-voted rows,
   frozen evalset score_ids excluded. Targets are the strict-JSON scoring
   object rebuilt from stored columns; direct-JSON distillation, no CoT.
2. **Train** (`scripts/train_qlora.py`): QLoRA r=32 via unsloth on a
   16GB RTX 4080 Super. 4-bit base `unsloth/Qwen3-8B-unsloth-bnb-4bit`,
   seq 8192 (p99 prompt is 6.4k tokens), loss masked to the assistant
   response, 2 epochs, bf16, adamw_8bit. A `gptoss` family flag exists for
   distilling into gpt-oss-20b instead (harmony format markers); not run.
3. **Merge + serve** (`scripts/merge_adapter.py`): merge the adapter to
   16-bit, serve with vLLM. Eval serving on the 4080S uses
   `--quantization bitsandbytes`; production serving on an Intel Arc B70
   runs dense bf16 (`deploy/b70-docker-compose.yml`).
4. **Eval** (`scripts/eval_scoring.py`, from gpt-oss-subset): frozen
   311-prompt evalset + 120b reference run, so numbers are directly
   comparable to the table above.

## Gotchas worth knowing

- unsloth's `prediction_step` materializes full-vocab fp32 logits even with
  `prediction_loss_only`, which OOMs a 16GB card at 8k sequence length.
  In-training eval is disabled (`eval_strategy="no"`); quality is judged by
  the offline replay instead, which is the metric that matters anyway.
- The in-training eval ran *before* the checkpoint save at the same step, so
  when it OOMed it took 2h of training with it. If you re-enable eval, save
  first.
- Serving the merged model on a desktop card: leave headroom for the display
  server (`--gpu-memory-utilization 0.78` on a 16GB card driving monitors).

## What is deliberately not here

- **`data/` is git-ignored**: the SFT pairs, evalset, and reference outputs
  are real (PII-scrubbed) user scoring records. Same policy as
  gpt-oss-subset.
- **No model weights.** reap-48 was published because pruning a public
  model touches no user data. This model was *trained on* user resumes and
  stored decisions; scrubbed or not, an SFT'd 8B can memorize its training
  set, so the weights stay private.
- **The production prompt builder** (`build_messages`) lives in the private
  reach.jobs backend; `build_sft_dataset.py` imports it. The dataset script
  is published for the selection/target-reconstruction logic, which is
  where the distillation decisions live.

## Hardware

Everything ran on consumer/workstation gear already on my desk: RTX 4080
Super 16GB (training, merge, bnb-4bit eval serving), Intel Arc B70 32GB
(production bf16 serving via intel/vllm XPU). The full 120b teacher never
left production; the dataset is its stored outputs.

## License

Apache 2.0.
