"""Compare semif-score --backend anthropic (Opus 5.5 via fcc) against the committed
Qwen3.5-4B direct predictions on the labeled authored144 set.

Run: .venv/bin/python benchmarks/opus_vs_committed.py OUTPUT.jsonl
Exits 1 if Opus scores below the committed Qwen baseline.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from semif_phase1 import anthropic_backend

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(line) for line in (ROOT / "benchmarks/data/authored144.jsonl").open()]
qwen = {r["id"]: r for r in map(json.loads, (ROOT / "results/raw/predictions/direct-authored144.jsonl").open())}
model, tokenizer, metadata = anthropic_backend.load_model(None)


def pick(probabilities):
    return max(range(len(probabilities)), key=probabilities.__getitem__)


with ThreadPoolExecutor(8) as pool:
    opus = list(pool.map(lambda r: anthropic_backend.score(
        model, tokenizer, {k: r[k] for k in ("id", "state", "question", "options")}, metadata), rows))
Path(sys.argv[1]).write_text("".join(json.dumps(o) + "\n" for o in opus))
opus_ok = sum(pick(o["probabilities"]) == r["label"] for o, r in zip(opus, rows))
qwen_ok = sum(pick(qwen[r["id"]]["probabilities"]) == r["label"] for r in rows)
print(f"authored144: opus {opus_ok}/{len(rows)}  committed qwen3.5-4B {qwen_ok}/{len(rows)}")
sys.exit(opus_ok < qwen_ok)
