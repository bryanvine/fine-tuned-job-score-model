"""Replay the frozen eval set against an OpenAI-compatible endpoint and score it.

Metrics per candidate:
  - Cohen's kappa of positive/negative decisions vs stored production decisions,
    where predicted positive = semantic_score >= the user's real semantic_threshold.
  - Pearson r of semantic_score vs the stored production semantic_score.
  - If --reference-results is given (a prior output file from this script, usually the
    full 120b), decision agreement %, kappa vs reference decisions, and semantic MAE
    vs reference scores. Model-vs-model agreement is less noisy than kappa vs prod
    (prod runs had research/exemplar context we do not replay).

Usage:
  python eval_scoring.py --evalset data/evalset.jsonl \
      --endpoint http://gpu-server:8100/v1 --model openai/gpt-oss-120b \
      --out results/eval-120b-reference.json [--concurrency 3] [--limit N] \
      [--reference-results results/eval-120b-reference.json]

Only stdlib + urllib (no httpx dependency; runs anywhere with python3).
"""

import argparse
import concurrent.futures as cf
import json
import math
import re
import time
import urllib.request

JSON_RE = re.compile(r"\{.*\}", re.S)
SAVE_RAW = False


def call_one(endpoint, model, rec, temp, max_retries=3):
    body = json.dumps({
        "model": model,
        "messages": rec["messages"],
        "temperature": temp,
        "max_tokens": 4096,
    }).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                endpoint.rstrip("/") + "/chat/completions", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                msg = json.load(r)["choices"][0]["message"]
            content = msg.get("content") or ""
            m = JSON_RE.search(content)
            scores = json.loads(m.group(0)) if m else {}
            out = {"score_id": rec["score_id"],
                   "semantic": scores.get("semantic_score"),
                   "risk": scores.get("risk_score"),
                   "raw_ok": m is not None and "semantic_score" in scores}
            if SAVE_RAW:
                out["content"] = content[:20000]
                out["reasoning"] = (msg.get("reasoning_content") or "")[:20000]
            return out
        except Exception as e:
            if attempt == max_retries - 1:
                return {"score_id": rec["score_id"], "semantic": None, "risk": None,
                        "raw_ok": False, "error": str(e)[:200]}
            time.sleep(2 * (attempt + 1))


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-results")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--save-raw", action="store_true",
                    help="store raw content and reasoning per result (for calibration corpora)")
    args = ap.parse_args()
    global SAVE_RAW
    SAVE_RAW = args.save_raw

    recs = [json.loads(l) for l in open(args.evalset)]
    if args.limit:
        recs = recs[: args.limit]
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        results = list(ex.map(lambda r: call_one(args.endpoint, args.model, r, args.temp), recs))
    wall = time.time() - t0

    by_id = {r["score_id"]: r for r in results}
    ok = [rec for rec in recs if by_id[rec["score_id"]]["raw_ok"]
          and by_id[rec["score_id"]]["semantic"] is not None]
    pred, stored, sem_pred, sem_stored = [], [], [], []
    for rec in ok:
        s = by_id[rec["score_id"]]["semantic"]
        pred.append(s >= rec.get("semantic_threshold", 0.75))
        stored.append(bool(rec["stored_pos"]) if "stored_pos" in rec
                      else rec["stored_decision"] in ("apply", "review"))
        sem_pred.append(float(s))
        sem_stored.append(float(rec["stored_semantic"]))

    summary = {
        "model": args.model, "endpoint": args.endpoint, "n_total": len(recs),
        "n_parsed": len(ok), "parse_rate": len(ok) / len(recs) if recs else 0,
        "wall_seconds": round(wall, 1),
        "kappa_vs_prod": kappa(pred, stored),
        "pearson_vs_prod": pearson(sem_pred, sem_stored),
    }

    if args.reference_results:
        ref = json.load(open(args.reference_results))
        ref_by_id = {r["score_id"]: r for r in ref["results"]}
        rp, rr, ms, mr = [], [], [], []
        for rec in ok:
            rf = ref_by_id.get(rec["score_id"])
            if not rf or rf.get("semantic") is None:
                continue
            s = by_id[rec["score_id"]]["semantic"]
            rp.append(s >= rec.get("semantic_threshold", 0.75))
            rr.append(rf["semantic"] >= rec.get("semantic_threshold", 0.75))
            ms.append(float(s))
            mr.append(float(rf["semantic"]))
        summary["n_vs_reference"] = len(rp)
        summary["agreement_vs_reference"] = (sum(x == y for x, y in zip(rp, rr)) / len(rp)) if rp else None
        summary["kappa_vs_reference"] = kappa(rp, rr)
        summary["semantic_mae_vs_reference"] = (sum(abs(a - b) for a, b in zip(ms, mr)) / len(ms)) if ms else None

    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
