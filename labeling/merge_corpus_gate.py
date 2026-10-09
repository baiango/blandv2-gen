#!/usr/bin/env python3
"""merge_corpus_gate.py — add cfgate gate keys onto corpus_labeled.jsonl.

Reads corpus_labeled.jsonl + batch_gate.py outputs (results + optional
.rejects.jsonl), writes a NEW file with 4 optional keys appended per row
(canonical order, see row_order.CANON):

    gate_decision   PASS | CONDITIONAL PASS | FAIL
    gate_score      0-100 float (server-computed, never the judge's own sum)
    gate_fails      sorted false hard-gate keys (the decision floors)
    gate_defaulted  judge keys default-filled (non-empty = degraded decode)

Rows WITHOUT a gate result must appear in the rejects file (verbatim reason);
they keep no gate keys. Coverage invariant: result_ks ∪ reject_ks == corpus_ks
(disjoint), asserted before writing. Content invariant: canon_md5 of every row
projected onto non-gate keys is identical before/after, asserted per row.

Usage:
    python3 merge_corpus_gate.py \
        --corpus corpus_labeled.jsonl \
        --gate gate_results.jsonl [more_results.jsonl ...] \
        [--rejects gate_results.jsonl.rejects.jsonl] \
        --out corpus_labeled_gate.jsonl [--dry-run]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from row_order import canon_md5, reorder_row  # noqa: E402

GATE_KEYS = ["gate_decision", "gate_score", "gate_fails", "gate_defaulted"]


def load_gate(paths):
    res, dups = {}, []
    for path in paths:
        for l in open(path):
            r = json.loads(l)
            k = r["k"]
            if k in res:
                dups.append(k)
                continue
            res[k] = r
    return res, dups


def load_rejects(path):
    rej = {}
    if path and os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            rej[r["k"]] = r.get("reason", "?")
    return rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--rejects", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    gate_paths = [p for pat in a.gate for p in glob.glob(pat)]
    if not gate_paths:
        sys.exit("no gate result files matched")
    res, dups = load_gate(gate_paths)
    rej = load_rejects(a.rejects)
    if dups:
        sys.exit(f"FATAL: {len(dups)} duplicate gate ks, first: {dups[:5]}")
    overlap = set(res) & set(rej)
    if overlap:
        sys.exit(f"FATAL: {len(overlap)} ks in BOTH results and rejects")

    covered, missing = 0, []
    n_rows = 0
    with open(a.corpus) as f:
        for l in f:
            n_rows += 1
            k = json.loads(l)["k"]
            if k in res or k in rej:
                covered += 1
            else:
                missing.append(k)
    if missing:
        sys.exit(f"FATAL: {len(missing)} corpus rows have no gate result and "
                 f"no reject entry, first: {missing[:5]}")
    print(f"corpus={n_rows} gate_results={len(res)} rejects={len(rej)} "
          f"covered={covered} ({100*covered/max(1,n_rows):.2f}%)")
    print("decision dist: ", end="")
    from collections import Counter
    print(dict(Counter(r["gate_decision"] for r in res.values())))

    if a.dry_run:
        print("DRY RUN — no output written")
        return
    if os.path.abspath(a.out) == os.path.abspath(a.corpus):
        sys.exit("refusing to overwrite corpus in place; pick another --out")

    tmp = a.out + ".tmp"
    n_bad = 0
    with open(a.corpus) as fin, open(tmp, "w") as fout:
        for l in fin:
            row = json.loads(l)
            base_md5 = canon_md5({k: v for k, v in row.items()
                                  if k not in GATE_KEYS})
            rec = res.get(row["k"])
            if rec is not None:
                row = {k: v for k, v in row.items() if k not in GATE_KEYS}
                row.update(rec)
            merged = reorder_row(row)
            if canon_md5({k: v for k, v in merged.items()
                          if k not in GATE_KEYS}) != base_md5:
                n_bad += 1
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
    if n_bad:
        os.remove(tmp)
        sys.exit(f"FATAL: {n_bad} rows mutated non-gate content; nothing written")
    os.replace(tmp, a.out)
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB, "
          f"{n_rows} rows, gate keys on {len(res)})")


if __name__ == "__main__":
    main()
