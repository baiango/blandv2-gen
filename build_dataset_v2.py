#!/usr/bin/env python3
"""bland v2 (MODEL-STOPS spec): the ceiling must never be the author.

Why v2 exists: v1 (cap 240) shipped 94% mid-sentence. Postmortem (790-draw
probe + 480-leaf box calibration) showed cap-240/300 ceilings were doing the
trimming AND the len gate (40..240 words) was calibrated to that cap — the
cap was the author of row length. User directive (10-06): LET THE MODEL STOP.

v2 = v1 machinery imported UNTOUCHED (style strip, gate glue, resume, shard,
k-dedupe) + four pinned changes:
  CEILING  CAP=800 srv tokens (790-draw probe parity; natural-ending p95 was
           780-790). A draw may run there, but nothing ships BECAUSE of it:
  STOP     a row ships only if finish_reason == "stop" — the model ended its
           own text. finish_reason is recorded on EVERY row as "stop" field;
           ceiling-cut draws fail with gates=["no-stop"] (+any gate whys).
  END      the shipped text must end in terminal punctuation (END_RE on the
           text that would ship); else gates=["no-end"] -> retry.
  BAND     words band widened via MG_LEN_MAX=700 (set in the launch env; the
           make_generic default stays 240 = v1 semantics). 700 is an
           anti-dump sanity bound, not a length target; upper len-edge soft
           flag tracks it.
Prompt  = F3 label form (format bake-off winner; 42% natural endings, 15%
          parent-leak): parent in prompt for proofread context,
            parent != leaf: "Write a story in English. Genre: {p}. Subgenre: {n}.\n\n"
            parent == leaf: "Write a story in English. Genre: {n}.\n\n" (no dup)
Sampler = pinned in the request body (samplers [top_k,top_p,min_p,temperature],
          k40 p95 t1.0 minp0.1) — server defaults inert; rep-pen banned.
Rows    = v1 fields + "ends" (bool) + "stop" (finish_reason) on EVERY row;
          "prompt" is the v2 F3 text. Verdicts: pass|style|failed|dupe|exhausted.
Retry   = <=ATTEMPTS fresh-seed draws until gates pass AND stop=stop AND ends.
          Calibration: ~10%/draw accept at ceiling ~300 (cap-240 arm);
          ATTEMPTS=60 => P(exhaust) < 0.5% at p>=0.08.
Seeds   = BASE_SEED + idx*10 + att; BASE_SEED=911000000 (v1 was 777000000).
Leaves  = leaves.jsonl (mg.load_leaves logic dumped; same sorted order => same
          idx/shard map as v1); avoids pandas on the box.
"""
import json, os, re, time, urllib.request, itertools, concurrent.futures as cf
from collections import Counter
import ceiling_probe as cp
import make_generic as mg
import build_dataset as bd          # v1 machinery: strip, glue — UNTOUCHED

# ---- CONFIG (v2 operating point; grep-able) ----
URL     = os.environ.get("BLAND_URL", "http://localhost:8937/v1/completions")
# 8-endpoint lever (10-06): BLAND_URLS = comma-separated list of ARG-IDENTICAL
# server clones. Round-robin per request. Same model + sampled params + seeds =>
# the draw distribution is unchanged; only WHICH clone answers changes.
URLS    = [u.strip() for u in os.environ.get("BLAND_URLS", "").split(",") if u.strip()] or [URL]
_rr     = itertools.count()
def pick_url():
    return URLS[next(_rr) % len(URLS)]
OUT     = os.environ.get("BLAND_OUT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "battery_data_v2", "bland_prose_stopped.jsonl")
WORKERS   = int(os.environ.get("BLAND_WORKERS", "8"))
SHARD     = os.environ.get("BLAND_SHARD", "")            # "i/n"
ATTEMPTS  = int(os.environ.get("BLAND_ATTEMPTS", "60"))
CAP       = int(os.environ.get("BLAND_CAP", "800"))      # ceiling; NEVER the author
BASE_SEED = 911000000
LEAVES    = os.environ.get("BLAND_LEAVES") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "leaves.jsonl")

bd.URL = URLS[0]                   # bd.gen resolves URL from ITS global at call time

END_RE = re.compile(r'[.!?…]["\')\]]?$')

FAILHIST = Counter()   # (stop, first-gate) per discarded draw — real retry census
ERR = Counter()        # client anomalies: http retries, empty first draws

def gen2(prompt, seed, mx=CAP):
    # v1 chain pinned EXPLICITLY (top_k/top_p/min_p/temperature; k40 p95 t1.0
    # minp0.1; no rep-pen) + finish_reason captured (the model-stops audit).
    body = json.dumps({"model": "q", "prompt": prompt, "max_tokens": mx,
                       "samplers": ["top_k", "top_p", "min_p", "temperature"],
                       "top_k": 40, "top_p": 0.95, "temperature": 1.0, "min_p": 0.1,
                       "seed": seed}).encode()
    req = urllib.request.Request(pick_url(), body, {"Content-Type": "application/json"})
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
            return r["choices"][0]["text"], r["choices"][0].get("finish_reason", "?")
        except Exception:
            ERR["http"] += 1
            time.sleep(2)
    return "", "error"

def prompt_for(leaf, par):
    if par == leaf:
        return f"Write a story in English. Genre: {leaf}.\n\n"
    return f"Write a story in English. Genre: {par}. Subgenre: {leaf}.\n\n"

def attempt(leaf, idx, att, par):
    prompt = prompt_for(leaf, par)
    text, stop = gen2(prompt, BASE_SEED + idx * 10 + att, mx=CAP)   # VERBATIM,
    if len(cp.toks(text)) < 30:                                     # non-rederivable
        ERR["empty1"] += 1
        text, stop = gen2(prompt, BASE_SEED + idx * 10 + att + 555555, mx=CAP)
    norm = " ".join(text.split())                    # gates/simhash want \s-free
    row = {"genre": leaf, "prompt": prompt, "text": text,
           "n_tok": len(cp.toks(norm)), "k": mg.sl.h(norm),
           "seed": BASE_SEED + idx * 10 + att,
           "stop": stop}                             # audit field: EVERY row
    prose, n_stripped = bd._style_strip_n(text)      # strip on VERBATIM (v1)
    flat = " ".join(prose.split()) if prose is not None else None
    target = flat if flat is not None else norm
    row["ends"] = bool(END_RE.search(target))        # audited on EVERY row
    why, soft = mg.gates_all(target, raw=text)
    if stop != "stop" and "no-stop" not in why:      # ceiling cut => never ship
        why = why + ["no-stop"]
    if not why and not row["ends"]:                  # model stopped mid-line
        why = ["no-end"]
    if why:
        row.update(verdict="failed", gates=why)
        return row
    if n_stripped >= 2: soft.append("style-heavy")   # v1 review flags, verbatim
    if prose is not None and prose[:1].islower():
        soft.append("lc-start")
    if mg.echo_cuts:
        soft.append("echo-lead")
    if soft: row["review"] = soft
    if prose is not None:
        row.update(prose=prose, verdict="style")
    else:
        row["verdict"] = "pass"
    return row

def main():
    global OUT
    si = sn = None
    if SHARD:
        si, sn = map(int, SHARD.split("/"))
        OUT = f"{OUT}.shard{si}of{sn}"
    leaves = [(r["n"], r["p"]) for r in map(json.loads, open(LEAVES))]
    done, seen_k0 = set(), set()
    if os.path.exists(OUT):
        for l in open(OUT):                          # resume: genres AND dedupe keys
            try:
                r = json.loads(l)
            except ValueError:                       # torn tail line (kill mid-flush)
                print("resume: skipping torn line", flush=True)
                continue
            done.add(r["genre"]); seen_k0.add(r["k"])
    pending = [(i, n, p) for i, (n, p) in enumerate(leaves) if n not in done]
    if SHARD:
        pending = [(i, n, p) for i, n, p in pending if i % sn == si]
    print(f"leaves={len(leaves)} done={len(done)} pending={len(pending)} "
          f"workers={WORKERS} shard={SHARD or '-'} urls={','.join(URLS)} cap={CAP} "
          f"attempts={ATTEMPTS} base_seed={BASE_SEED} "
          f"len_band=40..{mg.MG_LEN_MAX} ship_rule=gates+stop=stop+ends out={OUT}",
          flush=True)
    seen_k, n_ship, n_att, n_nostop = seen_k0, 0, 0, 0
    t0 = time.time()
    lock_out = open(OUT, "a")
    def work(arg):
        idx, leaf, par = arg
        row = attempt(leaf, idx, 0, par)
        att = 1
        while att < ATTEMPTS and (row["verdict"] == "failed" or row["k"] in seen_k):
            FAILHIST[(row.get("stop", "?"),
                      (row.get("gates") or ["dupe"])[0])] += 1
            row = attempt(leaf, idx, att, par); att += 1   # fresh seed every draw
        if row["verdict"] == "failed":
            row["verdict"], row["att"] = "exhausted", att
        return {"parent": par, "draws": att, **row}
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for n, row in enumerate(ex.map(work, pending), 1):
            n_att += row.get("draws", 1)
            if "no-stop" in (row.get("gates") or []): n_nostop += 1
            v = row["verdict"]
            if v in ("pass", "style") and row["k"] in seen_k:
                row["verdict"] = "dupe"
            if row["verdict"] in ("pass", "style"):
                seen_k.add(row["k"]); n_ship += 1
            lock_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 500 == 0:
                lock_out.flush()
                el = time.time() - t0
                print(f"  {n}/{len(pending)} ship={n_ship} ({n_ship/n:.1%}) "
                      f"draws/leaf={n_att/n:.1f} rate={n/el:.2f}/s "
                      f"eta_h={(len(pending)-n)/(n/el)/3600:.1f} err={dict(ERR)} "
                      f"fails={FAILHIST.most_common(6)}", flush=True)
    lock_out.flush(); lock_out.close()
    print(f"DONE rows={len(pending)} ship={n_ship} draws/leaf={n_att/max(len(pending),1):.1f} "
          f"elapsed={(time.time()-t0)/3600:.2f}h -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
