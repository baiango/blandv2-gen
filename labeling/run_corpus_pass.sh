#!/bin/bash
# One labeling pass with retry sweeps. Usage: run_corpus_pass.sh <1|2|3>
# Sweeps: run driver -> strip rows that failed parsing/HTTP from OUT -> re-run
# (driver resume skips any k present in OUT, so failed rows MUST be stripped
#  between attempts or they become permanent holes).
PASS=$1
cd /Users/arin/pctransfer/writing_storm/genre_classify || exit 1
OUT=corpus_ling_r${PASS}.jsonl
LOG=corpus_ling_r${PASS}.log
echo "=== pass ${PASS} start $(date) ===" >> "$LOG"
for sweep in 1 2 3 4 5; do
  # strip failed rows (bad world/energy) so resume retries them
  python3 - "$OUT" <<'PYEOF'
import json, sys, os
out = sys.argv[1]
if not os.path.exists(out):
    sys.exit(0)
ok, bad = [], 0
with open(out) as f:
    for l in f:
        try:
            r = json.loads(l)
        except Exception:
            bad += 1; continue
        if r["world"] in {"Contemporary","Fantasy & Supernatural","Science Fiction","Historical & Western"} \
           and r["energy"] in {"External","Relational"}:
            ok.append(l)
        else:
            bad += 1
if bad:
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(ok)
    os.replace(tmp, out)
print(f"[sweep-strip] kept {len(ok)} rows, stripped {bad}", flush=True)
PYEOF
  python3 classify_prod_v3.py \
    URL=https://openrouter.ai/api/v1/chat/completions \
    MODEL=inclusionai/ling-3.0-flash \
    IN=/Users/arin/pctransfer/writing_storm/skipgram_calib/blandv2_pull/sharded10/full_corpus.jsonl \
    WORKERS=6 \
    OUT=$OUT MISSES=corpus_ling_r${PASS}_miss.jsonl >> "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then echo "[sweep ${sweep}] DRIVER CRASHED rc=$RC, aborting pass" >> "$LOG"; break; fi
  LAST=$(grep '^DONE' "$LOG" | tail -1)
  MISS=$(echo "$LAST" | sed -n 's/.*miss=\([0-9]*\).*/\1/p')
  echo "[sweep ${sweep}] ${LAST}" >> "$LOG"
  if [ "$MISS" = "0" ]; then break; fi
done
echo "=== pass ${PASS} finished $(date) ===" >> "$LOG"
