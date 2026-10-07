#!/usr/bin/env python3
"""patch_dead_trees.py — regenerate the 10 exhausted leaves until each ships.

Reuses build_dataset_v2.attempt() verbatim (same model, samplers, strip, gates).
Seeds CONTINUE the v2 stream: BASE_SEED + idx*10 + att, att>=60 (v2 burned 0..59).
Append-only output: blandv2_pull/sharded10/patch_dead10.jsonl (shards stay frozen).

ENV: BLAND_URL (Mac llama-server), MG_LEN_MAX=700 required,
     PATCH_MAX (extra attempts/leaf, default 140), PATCH_WORKERS (default 3).
"""
import json, os, sys, time, threading
import concurrent.futures as cf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset_v2 as v          # attempt() machinery; main() is guarded

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "blandv2_pull", "sharded10", "patch_dead10.jsonl")
SHARDS = [os.path.join(HERE, "blandv2_pull", "sharded10",
                       f"bland_prose_stopped.shard{i}of10") for i in range(10)]
DEAD = ["Lục Bảo Huyền Ký", "Pathwise Fantasy", "Phantom Resistance",
        "GARLIC WESTERN FICTION HERMETIC ALCHEMISTRY", "MYSTIC MOTION",
        "MYSTERY MINDFUL VELOCITY", "BLOOM SHADOW", "Epic Monachism",
        "Magical Quest", "GRAVITY WAYFARER"]
ATT0 = int(os.environ.get("PATCH_ATT0", "60"))          # v2 used 0..59
MAXX = int(os.environ.get("PATCH_MAX", "140"))          # extra attempts per leaf
WORKERS = int(os.environ.get("PATCH_WORKERS", "3"))

assert os.environ.get("MG_LEN_MAX") == "700", "MG_LEN_MAX=700 required (v2 band)"

leaves = [(r["n"], r["p"]) for r in map(json.loads, open(v.LEAVES))]  # exact v2 order
targets = [(i, n, p) for i, (n, p) in enumerate(leaves) if n in set(DEAD)]
assert len(targets) == 10, f"expected 10 dead leaves, matched {len(targets)}"

seen = set()
for f in SHARDS:
    with open(f) as fh:
        for line in fh:
            seen.add(json.loads(line)["k"])

print(f"CONFIG leaves=197006 targets=10 seen_k={len(seen)} "
      f"att={ATT0}..{ATT0 + MAXX - 1} workers={WORKERS} "
      f"url={v.URLS[0]} out={OUT}", flush=True)

lock = threading.Lock()
done_ok = 0

def run_leaf(t):
    global done_ok
    idx, n, p = t
    for att in range(ATT0, ATT0 + MAXX):
        row = v.attempt(n, idx, att, p)
        if row["verdict"] in ("pass", "style") and row["k"] not in seen:
            row["parent"] = p
            row["draws"] = att
            with lock:
                seen.add(row["k"])
                with open(OUT, "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done_ok += 1
                ok_now = done_ok
            print(f"[{time.time() - t0:6.0f}s] {n[:40]:40s} seed={row['seed']} "
                  f"att={att} verdict={row['verdict']} -> SHIPPED ({ok_now}/10)",
                  flush=True)
            return True
    print(f"[{time.time() - t0:6.0f}s] {n[:40]:40s} exhausted at att={ATT0 + MAXX - 1}",
          flush=True)
    return False

t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
    results = list(ex.map(run_leaf, targets))
print(f"DONE shipped={sum(results)}/10 elapsed={time.time() - t0:.0f}s", flush=True)
