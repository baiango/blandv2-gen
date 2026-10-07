#!/usr/bin/env python3
"""Offline repair + schema migration. ONE content key: "text" = VERBATIM model
output (newlines kept); "prose" = derived clean copy; n_tok/k set at build time.

All post-processing (style strip, gates, review flags) is a pure function of
verbatim text, so a bug in any of it NEVER means a wipe — re-run this script
instead. Wipes are only for generation-affecting changes
(prompt/sampler/server/seeds).

Legacy rows (old schema: whitespace-normalized "text" + separate "raw") are
MIGRATED in the same pass: text=raw, "raw" key dropped. New-schema rows repair
in place. Normalize idiom " ".join(text.split()) is idempotent, so rows from
every era repair identically.

Usage: python3 restrip.py FILE.jsonl [FILE2.jsonl ...]
       in-place via .tmp + rename.
Note : recomputed verdicts are honest — a row whose repaired text now fails a
       gate is marked failed+gates (offline; no auto-redraw). Keep only if you
       agree, or re-draw those leaves separately.
Note : "ends" is deliberately NOT recomputed (target-derived at build time; 10-09
       measured 0 flips over 17,544 rows, and failed rows are dropped anyway).
Note : NEVER run this on a file a LIVE driver holds open — the driver keeps
       writing to the unlinked old inode (open(OUT, "a") at startup) and every
       row written afterwards is lost. Stop the drivers first.
"""
import json, os, sys
import build_dataset as bd

def repair_row(r):
    src = r.get("raw") or r.get("text")           # verbatim, either schema era
    if src is None:
        return None
    prose, n = bd._style_strip_n(src)
    flat = " ".join(prose.split()) if prose is not None else None   # gates want \s-free
    target = flat if flat is not None else " ".join(src.split())
    why, soft = bd.mg.gates_all(target, raw=src)
    if n >= 2:
        soft.append("style-heavy")
    if bd.mg.echo_cuts:
        soft.append("echo-lead")                  # 10-07: echo/spec/ident lead cut
    if prose is not None and prose[:1].islower():
        soft.append("lc-start")                   # stripper may have eaten opening
    out = dict(r)
    for k in ("prose", "review", "gates"):
        out.pop(k, None)
    out["text"] = src                             # migrate: verbatim under one key
    out.pop("raw", None)
    if soft:
        out["review"] = soft
    if why:
        out.update(verdict="failed", gates=why)
    elif prose is not None:
        out.update(prose=prose, verdict="style")
    else:
        out["verdict"] = "pass"
    return out

def main():
    for path in sys.argv[1:]:
        rows, changed, skipped = [], 0, 0
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            nr = repair_row(r)
            if nr is None:
                skipped += 1
                nr = r
            else:
                changed += (nr != r)
            rows.append(nr)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        print(f"{path}: {len(rows)} rows, {changed} repaired/migrated, "
              f"{skipped} skipped", flush=True)

if __name__ == "__main__":
    main()
