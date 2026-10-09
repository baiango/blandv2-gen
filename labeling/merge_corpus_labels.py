#!/usr/bin/env python3
"""Merge 3-vote genre labels into the full corpus.

Reads full_corpus.jsonl + corpus_ling_r{1,2,3}.jsonl, writes corpus_labeled.jsonl:
each original row VERBATIM + added fields world/energy/tags/confidence.
  confidence: 'unanimous' | '2-1' | 'tie'  (energy vote split; ties fall back to r1)
  tags: union of tags present in >=2 of 3 votes
Stdlib only. Usage: python3 merge_corpus_labels.py [DRY_RUN=1]
"""
import json, os
from row_order import reorder_row
from collections import Counter

BASE = os.path.expanduser("~/pctransfer/writing_storm/genre_classify")
IN   = os.path.expanduser("~/pctransfer/writing_storm/skipgram_calib/blandv2_pull/sharded10/full_corpus.jsonl")
OUT  = os.path.join(BASE, "corpus_labeled.jsonl")
VOTES = [os.path.join(BASE, f"corpus_ling_r{i}.jsonl") for i in (1, 2, 3)]
TAGS = ["Drama / Slice of Life", "Romance", "Mystery & Crime",
        "Action & Adventure", "Horror"]
WORLDS = {"Contemporary", "Fantasy & Supernatural", "Science Fiction",
          "Historical & Western"}
ENERGIES = {"External", "Relational"}

def majority(vals):
    c = Counter(v for v in vals if v is not None)
    if not c:
        return None, "missing"
    top, n = c.most_common(1)[0]
    if len(c) == 1:
        return top, "unanimous"
    if n * 2 > sum(c.values()):
        return top, "2-1"
    return vals[0], "tie"

def main():
    votes = {}  # k -> [row_r1, row_r2, row_r3]
    for p in VOTES:
        with open(p) as f:
            for l in f:
                r = json.loads(l)
                if r["world"] in WORLDS and r["energy"] in ENERGIES:
                    votes.setdefault(r["k"], []).append(r)
    n_in = n_lab = 0
    conf = Counter(); world_conf = Counter()
    tmp = OUT + ".tmp"
    with open(IN) as f, open(tmp, "w") as g:
        for l in f:
            row = json.loads(l)
            n_in += 1
            vs = votes.get(row["k"], [])
            if vs:
                w, wc = majority([v["world"] for v in vs])
                e, ec = majority([v["energy"] for v in vs])
                tcount = Counter(t for v in vs for t in set(v["tags"]))
                tags = [t for t in TAGS if tcount[t] >= 2]
                row["world"] = w
                row["energy"] = e
                row["tags"] = tags
                row["confidence"] = ec
                conf[ec] += 1; world_conf[wc] += 1
                n_lab += 1
            g.write(json.dumps(reorder_row(row), ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)
    print(f"rows in: {n_in}, labeled: {n_lab} ({n_lab/max(n_in,1):.1%}), missing votes: {n_in-n_lab}")
    print(f"energy confidence: {dict(conf)}")
    print(f"world confidence:  {dict(world_conf)}")

if __name__ == "__main__":
    main()
