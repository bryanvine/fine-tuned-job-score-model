"""Aggregate live shadow-compare stats per shadow model.

The FAST scoring tier fires an async shadow call on the same prompt as the
primary model and appends one JSON line per pair:

    {"shadow_model": ..., "primary": {semantic_score, confidence_score,
     risk_score, location_ok}, "shadow": {...}, "semantic_delta": ..., "at": ...}

Usage:
    python shadow_stats.py /path/to/shadow_score_compare.jsonl
"""

import json
import statistics
import sys
from collections import defaultdict


def main(path: str) -> None:
    groups = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("primary") and rec.get("shadow"):
                groups[rec.get("shadow_model", "?")].append(rec)

    for model, recs in sorted(groups.items()):
        sem = [abs(r["semantic_delta"]) for r in recs if r.get("semantic_delta") is not None]
        conf = [abs(r["primary"]["confidence_score"] - r["shadow"]["confidence_score"])
                for r in recs
                if r["primary"].get("confidence_score") is not None
                and r["shadow"].get("confidence_score") is not None]
        risk = [abs(r["primary"]["risk_score"] - r["shadow"]["risk_score"])
                for r in recs
                if r["primary"].get("risk_score") is not None
                and r["shadow"].get("risk_score") is not None]
        loc = [r["primary"]["location_ok"] == r["shadow"]["location_ok"]
               for r in recs
               if r["primary"].get("location_ok") is not None
               and r["shadow"].get("location_ok") is not None]
        times = sorted(r["at"] for r in recs if r.get("at"))

        print(f"== {model} (n={len(recs)}) ==")
        if times:
            print(f"  window: {times[0]} .. {times[-1]}")
        if sem:
            print(f"  semantic delta: mean {statistics.mean(sem):.3f}  "
                  f"median {statistics.median(sem):.3f}  "
                  f"stdev {statistics.stdev(sem):.3f}" if len(sem) > 1 else "")
            print(f"  within 0.1: {sum(d <= 0.1 for d in sem) / len(sem):.1%}   "
                  f"within 0.2: {sum(d <= 0.2 for d in sem) / len(sem):.1%}")
        if conf:
            print(f"  confidence MAE: {statistics.mean(conf):.3f}")
        if risk:
            print(f"  risk MAE: {statistics.mean(risk):.3f}")
        if loc:
            print(f"  location_ok agreement: {sum(loc) / len(loc):.1%}")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
