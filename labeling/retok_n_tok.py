#!/usr/bin/env python3
"""Recompute n_tok as subword tokens (Qwen/Qwen3-0.6B-Base tokenizer).

Rewrites corpus_labeled.jsonl in place (atomic tmp+replace):
  n_tok  = Qwen3-0.6B-Base subword count (add_special_tokens=False)
  n_word = OLD regex word count, preserved verbatim
Not resumable — single atomic pass; if interrupted, delete .tmp and rerun.
"""
import json, os, time
from row_order import reorder_row
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from transformers import AutoTokenizer

PATH = os.path.expanduser("~/pctransfer/writing_storm/genre_classify/corpus_labeled.jsonl")
TMP = PATH + ".tmp"
BATCH = 512

def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    t0 = time.time()
    stats = {"n": 0, "tok": 0}

    def flush(batch):
        texts = [(r.get("text") or r.get("prose") or "") for r in batch]
        encs = tok(texts, add_special_tokens=False)["input_ids"]
        with open(TMP, "a") as g:
            for r, ids in zip(batch, encs):
                r["n_word"] = r.get("n_tok")      # preserve old word count
                r["n_tok"] = len(ids)             # Qwen3 subword count
                g.write(json.dumps(reorder_row(r), ensure_ascii=False) + "\n")
        stats["n"] += len(batch)
        stats["tok"] += sum(len(e) for e in encs)
        if stats["n"] % 2048 < BATCH:
            el = time.time() - t0
            print(f"{stats['n']} rows, {stats['tok']:,} tok, "
                  f"{stats['n']/el:.0f} rows/s, "
                  f"eta {(197016-stats['n'])/max(stats['n']/el,.01)/60:.0f} min", flush=True)

    batch = []
    with open(PATH) as f:
        for l in f:
            batch.append(json.loads(l))
            if len(batch) >= BATCH:
                flush(batch); batch = []
    if batch:
        flush(batch)
    os.replace(TMP, PATH)
    print(f"DONE rows={stats['n']} total_qwen_tokens={stats['tok']:,} "
          f"avg={stats['tok']/max(stats['n'],1):.0f} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
