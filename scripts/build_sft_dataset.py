"""Build an SFT distillation dataset for the FAST job-scoring tier.

Runs INSIDE the reach backend container (same convention as backtest_scoring_small.py):

    docker cp scripts/build_sft_dataset.py reach-backend-1:/tmp/build_sft.py
    docker exec reach-backend-1 python /tmp/build_sft.py \
        --neg 12000 --exclude-ids /tmp/evalset_ids.txt --out /app/data/benchmarks/sft

Each record pairs the EXACT production scoring prompt (rebuilt via
backtest_scoring_small.build_messages, research/exemplar context empty, resume
PII-scrubbed) with the stored gpt-oss-120b output reconstructed as the strict
JSON object the system prompt demands. This distills the full 120b's behavior
into whatever student model we fine-tune.

Selection:
  - all apply/review rows (positives are scarce, ~7k)
  - --neg random skip rows
  - all human-voted rows
  - model = openai/gpt-oss-120b, rationale complete (incl. location_ok)
  - score_ids in --exclude-ids (the frozen 311-prompt evalset) are dropped

Output <out>-train.jsonl / <out>-val.jsonl:
  {score_id, persona, stored_decision, user_rating, semantic_threshold,
   messages: [system, user], target: "<json string>"}
"""

import argparse
import asyncio
import json
import random
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/scripts")

from sqlalchemy import select, text  # noqa: E402

from app.core.database import async_session  # noqa: E402
from app.models import Job, JobScore, Profile, User  # noqa: E402
from backtest_scoring_small import build_messages, scrub  # noqa: E402

TARGET_KEYS = ("semantic_score", "confidence_score", "risk_score",
               "location_ok", "reasoning", "key_matches", "gaps")


def build_target(sc, user_name: str) -> str | None:
    """Reconstruct the model's JSON output from stored columns + rationale."""
    r = sc.rationale or {}
    if r.get("location_ok") is None or sc.semantic_score is None or sc.confidence_score is None:
        return None
    if not isinstance(r.get("risk_score"), (int, float)):
        return None

    def clean_str(s):
        return scrub(s, user_name) if isinstance(s, str) else ""

    def clean_list(v):
        return [clean_str(x) for x in v if isinstance(x, str)] if isinstance(v, list) else []

    obj = {
        "semantic_score": float(sc.semantic_score),
        "confidence_score": float(sc.confidence_score),
        "risk_score": float(r["risk_score"]),
        "location_ok": bool(r["location_ok"]),
        "reasoning": clean_str(r.get("reasoning", "")),
        "key_matches": clean_list(r.get("key_matches")),
        "gaps": clean_list(r.get("gaps")),
    }
    return json.dumps(obj, ensure_ascii=False)


async def fetch(neg_limit: int):
    async with async_session() as session:
        async def grab(where, limit):
            q = (
                select(JobScore, Job, Profile, User)
                .join(Job, JobScore.job_id == Job.id)
                .join(Profile, JobScore.profile_id == Profile.id)
                .join(User, JobScore.user_id == User.id)
                .where(
                    JobScore.model == "openai/gpt-oss-120b",
                    JobScore.semantic_score.isnot(None),
                    JobScore.rationale.isnot(None),
                    *where,
                )
                .order_by(text("random()"))
            )
            if limit:
                q = q.limit(limit)
            return (await session.execute(q)).all()

        rows = await grab([JobScore.decision.in_(("apply", "review"))], 0)
        rows += await grab([JobScore.decision == "skip"], neg_limit)
        rows += await grab([JobScore.user_rating.isnot(None)], 0)
        return rows


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", type=int, default=12000)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--exclude-ids", help="file with one score_id per line to drop")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="/app/data/benchmarks/sft")
    args = ap.parse_args()

    exclude = set()
    if args.exclude_ids:
        exclude = {l.strip() for l in open(args.exclude_ids) if l.strip()}

    rows = await fetch(args.neg)
    records, seen, n_skip = [], set(), 0
    for sc, job, prof, user in rows:
        sid = str(sc.id)
        if sid in seen or sid in exclude:
            continue
        seen.add(sid)
        body = prof.body_of_work or {}
        resume = body.get("resume_text") or ""
        if not resume.strip():
            continue
        target = build_target(sc, user.name or "")
        if target is None:
            n_skip += 1
            continue
        skills = body.get("skills") or []
        item = {
            "score_id": sid,
            "title": job.title or "", "company": job.company or "",
            "location": job.location or "",
            "description": job.description or job.title or "",
            "resume_text": scrub(resume, user.name or ""),
            "skills_str": ", ".join(skills) if skills else "Not specified",
            "prefs": prof.search_prefs or {},
        }
        records.append({
            "score_id": sid,
            "persona": prof.name,
            "stored_decision": sc.decision,
            "user_rating": sc.user_rating,
            "semantic_threshold": (prof.search_prefs or {}).get("semantic_threshold", 0.75),
            "messages": build_messages(item),
            "target": target,
        })

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    splits = {"val": records[:n_val], "train": records[n_val:]}
    for name, recs in splits.items():
        path = f"{args.out}-{name}.jsonl"
        with open(path, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_pos = sum(r["stored_decision"] in ("apply", "review") for r in recs)
        print(f"{name}: {len(recs)} records ({n_pos} positive) -> {path}")
    print(f"skipped {n_skip} rows with incomplete rationale, excluded {len(exclude)} eval ids")


if __name__ == "__main__":
    asyncio.run(main())
