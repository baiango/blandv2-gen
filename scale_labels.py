#!/usr/bin/env python3
"""Scale the judge-labeled pool from 730 -> ~3000 paragraphs for ranker distillation
(students: Veyra2-Blueberry-5M, Veyra2-Apricot-50M; reference: ling30-m4r16 judge).

Design notes:
- Labels keyed by sha1(text)[:16] in battery_data/ranker_labels.jsonl (source-tagged),
  unlike the old index-keyed ceiling_judge_labels.jsonl. Old 730 labels are PORTED
  (no re-judging) by reconstructing the exact paras order of ceiling_probe.distill().
- New positives: cliche = 9 rotations of the 20 gold sentences in pairs (:8937 maker);
  babble = 12 openers x 68 seeds. Verify with the SAME gates (short/loop/format/lowtics).
- join = random 3-sentence joins of the 298 verified generic sentences (no generation).
- fresh extras: laguna 3turn [300:800], kimi genre500 [300:500] — slices chosen by
  replaying build()'s rng(7) sequence so they are disjoint from the original 300/300.
- fresh_ul: REAL merged-ladder bench outputs (ul5/ul6/dag2/grpo3 bench_results jsonl in
  ~/backups/box_final_2026-10-07) — the exact distribution the ranker gates at train
  time, incl. gate-FAILED rows (true glitches, honestly sourced). No live sampling needed.
Stages: port | gen | verify | bench | label | status   ('all' = gen+verify+bench+label)
"""
import hashlib, json, os, random, re, sys, urllib.request
import ceiling_probe as cp
import dataset_battery as db

W, D = cp.W, cp.D
CK = D + "ranker_labels.jsonl"
GK = D + "scale_gen.jsonl"
PK = D + "scale_positives.json"
URL_G = "http://localhost:8937/v1/completions"   # Qwen3-0.6B-Base maker
BENCH_GLOB = os.path.expanduser(
    "~/backups/box_final_2026-10-07/ladder/sft_long/minicpm_ladder/bench_results_*.jsonl")

def h(t): return hashlib.sha1(t.encode()).hexdigest()[:16]

def _post(url, payload):
    body = json.dumps(payload).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(
        url, body, {"Content-Type": "application/json"}), timeout=240))

def _gen(url, prompt, seed, mx):
    r = _post(url, {"prompt": prompt, "temperature": 1.0, "min_p": 0.1,
                    "samplers": ["min_p", "temperature"],   # pin FULL chain (server defaults add top_k/top_p)
                    "top_k": 0, "top_p": 1.0,
                    "max_tokens": mx, "stop": ["\n\n"], "seed": seed})
    return r["choices"][0]["text"].strip()

# ---------- port old 730 labels ----------
def port():
    s = json.load(open(D + "ceiling_sets.json"))
    pos = json.load(open(D + "ceiling_positives.json"))
    paras = ([p["text"] for p in pos if p["kind"] == "cliche"] +
             [p["text"] for p in pos if p["kind"] == "babble"] +
             s["fresh_kimi"] + s["fresh_laguna"])
    src = (["cliche"] * sum(1 for p in pos if p["kind"] == "cliche") +
           ["babble"] * sum(1 for p in pos if p["kind"] == "babble") +
           ["fresh_kimi"] * len(s["fresh_kimi"]) + ["fresh_laguna"] * len(s["fresh_laguna"]))
    assert len(paras) == len(src)
    n = 0
    with open(CK, "w") as f:
        for l in open(D + "ceiling_judge_labels.jsonl"):
            r = json.loads(l)
            if r["score"] is None: continue
            f.write(json.dumps({"k": h(paras[r["i"]]), "s": r["score"], "src": src[r["i"]]}) + "\n")
            n += 1
    print(f"ported {n} labels -> {CK}")

# ---------- generate new positives (:8937) ----------
def gen():
    gold = cp.gold_cliches()
    jobs = []
    for off in range(9):                                   # 540 cliche-seeded
        for i in range(60):
            st = (2 * i + off) % len(gold)
            pair = gold[st] + " " + gold[(st + 1) % len(gold)]
            jobs.append(("cliche", pair, 990000 + off * 1000 + i))
    for off in range(4):                                   # 816 babble
        for oi, op in enumerate(cp.OPENERS):
            for i in range(17):
                jobs.append(("babble", op, 970000 + off * 3000 + oi * 100 + i))
    rng = random.Random(99); rng.shuffle(jobs)
    done = set()
    if os.path.exists(GK):
        for l in open(GK): done.add(json.loads(l)["j"])
    ok = 0; n = 0
    with open(GK, "a") as f:
        for j, (kind, opener, sd) in enumerate(jobs):
            if j in done: continue
            t = _gen(URL_G, opener + " ", sd, 220)
            if len(cp.toks(t)) < 40: t = _gen(URL_G, opener + " ", sd + 555555, 220)
            t = cp.first_words((opener + " " + t).strip(), cp.NWIN)
            f.write(json.dumps({"j": j, "kind": kind, "opener": opener, "text": t}) + "\n")
            ok += bool(len(cp.toks(t)) >= 40); n += 1
            if n % 50 == 0: f.flush(); print(f"  gen {j}/{len(jobs)} ok={ok}", flush=True)
    print(f"generated ok={ok}/{n} -> {GK}")

# ---------- verify + joins ----------
def verify():
    rows = [json.loads(l) for l in open(GK)]
    keep = []
    for r in rows:
        t = r["text"]
        tics = sum(len(rx.findall(t)) for _, rx in db.CLICHE)
        why = None
        if len(cp.toks(t)) < 40: why = "short"
        elif cp.has_loop(t): why = "loop"
        elif cp.FORMAT_LEAK.search(t): why = "format"
        elif r["kind"] == "cliche" and tics < 2: why = "lowtics"
        elif re.search(r"\d", t): why = "digits"
        if not why:
            keep.append({"k": h(t), "kind": r["kind"], "text": t, "tics": tics})
    pool = list(json.load(open(W + "generic_corpus.json")))
    rng = random.Random(4242)
    for j in range(200):                                   # join positives, no GPU
        t = " ".join(rng.sample(pool, 3))
        keep.append({"k": h(t), "kind": "join", "text": t,
                     "tics": sum(len(rx.findall(t)) for _, rx in db.CLICHE)})
    from collections import Counter
    print("kept:", Counter(r["kind"] for r in keep))
    json.dump(keep, open(PK, "w"))

# ---------- fresh extras (disjoint by rng replay) ----------
def fresh_sets():
    rng = random.Random(7)                     # replay ceiling_probe.build() order
    fk, fl = [], []
    for f in (D + "source/genre500_uscrub_train.jsonl", D + "source/genre500_uscrub_val.jsonl"):
        for line in open(f): fk.append(cp.first_words(json.loads(line)["text"], cp.NWIN))
    for line in open(D + "train_3turn.jsonl"):
        fl.append(cp.first_words("\n\n".join(json.loads(line)["pieces"]), cp.NWIN))
    rng.shuffle(fk); rng.shuffle(fl)
    return fk[300:500], fl[300:800]

# ---------- real merged-ladder bench outputs ----------
def bench():
    import glob
    out = D + "scale_ul_bench.jsonl"
    have = set()
    if os.path.exists(out):
        for l in open(out): have.add(json.loads(l)["k"])
    n = 0
    with open(out, "a") as f:
        for path in sorted(glob.glob(BENCH_GLOB)):
            tag = os.path.basename(path).replace("bench_results_", "").replace(".jsonl", "")
            for l in open(path):
                r = json.loads(l)
                t = r.get("text") or ""
                if not t: continue
                t = cp.first_words(" ".join(t.split()), cp.NWIN)   # same 110-word cap
                k = h(t)
                if k in have or len(cp.toks(t)) < 40: continue
                f.write(json.dumps({"k": k, "text": t, "tag": tag,
                                    "ok": bool(r.get("ok")),
                                    "fail": r.get("fail") or r.get("fail_raw")}) + "\n")
                n += 1
            print(f"  {tag}", flush=True)
    print(f"bench outputs harvested n={n} ->", out)

# ---------- judge-label everything (:8936) ----------
def _judge_para(t):
    ss = [x for x in cp.split_re.split(t) if 8 <= len(cp.toks(x)) <= 40] or [t]
    v = [db.judge_sentence(x) for x in ss]
    v = [x for x in v if x is not None]
    return float(cp.np.mean(v)) if v else None

def label():
    ek, el = fresh_sets()
    bp = D + "scale_ul_bench.jsonl"
    bench_rows = ([json.loads(l) for l in open(bp)] if os.path.exists(bp) else [])
    btext = {r["k"]: r["text"] for r in bench_rows}
    pool = ([(r["k"], r["kind"], r["text"]) for r in json.load(open(PK))] +
            [(h(t), "fresh_kimi", t) for t in ek] +
            [(h(t), "fresh_laguna", t) for t in el] +
            [(r["k"], "fresh_ul", btext[r["k"]]) for r in bench_rows])
    have = set()
    if os.path.exists(CK):
        for l in open(CK): have.add(json.loads(l)["k"])
    todo = [p for p in pool if p[0] not in have]
    print(f"pool={len(pool)} labeled={len(have)} todo={len(todo)}")
    n = 0
    with open(CK, "a") as f:
        for k, src, t in todo:
            s = _judge_para(t)
            if s is not None:
                f.write(json.dumps({"k": k, "s": round(s, 4), "src": src}) + "\n")
            n += 1
            if n % 50 == 0: f.flush(); print(f"  label {n}/{len(todo)}", flush=True)
    print("labeled ->", CK)

# ---------- final dataset: text+src+score joined ----------
def materialize():
    s = json.load(open(D + "ceiling_sets.json"))
    pos = json.load(open(D + "ceiling_positives.json"))
    nc = sum(1 for p in pos if p["kind"] == "cliche")
    nb = sum(1 for p in pos if p["kind"] == "babble")
    paras = ([p["text"] for p in pos if p["kind"] == "cliche"] +
             [p["text"] for p in pos if p["kind"] == "babble"] +
             s["fresh_kimi"] + s["fresh_laguna"])
    srcs = (["old_cliche"] * nc + ["old_babble"] * nb +
            ["old_fresh_kimi"] * len(s["fresh_kimi"]) +
            ["old_fresh_laguna"] * len(s["fresh_laguna"]))
    text = {h(t): (t, sc) for t, sc in zip(paras, srcs)}
    for r in json.load(open(PK)):
        text[r["k"]] = (r["text"], "new_" + r["kind"])
    ek, el = fresh_sets()
    for t in ek: text[h(t)] = (t, "fresh_kimi_x")
    for t in el: text[h(t)] = (t, "fresh_laguna_x")
    for r in [json.loads(l) for l in open(D + "scale_ul_bench.jsonl")]:
        text[r["k"]] = (r["text"], "fresh_ul")
    extra = {r["k"]: {"ok": r.get("ok"), "fail": r.get("fail")}
             for r in [json.loads(l) for l in open(D + "scale_ul_bench.jsonl")]}
    lab = {json.loads(l)["k"]: json.loads(l)["s"] for l in open(CK)}
    n = miss = 0
    with open(D + "ranker_pool.jsonl", "w") as f:
        for k, sc in sorted(lab.items()):
            if k not in text: miss += 1; continue
            t, sc2 = text[k]
            row = {"k": k, "text": t, "src": sc2, "s": sc}
            row.update(extra.get(k, {}))
            f.write(json.dumps(row) + "\n")
            n += 1
    print(f"materialized {n} (missing {miss}) -> ranker_pool.jsonl")


def status():
    from collections import Counter
    import numpy as np
    rows = [json.loads(l) for l in open(CK)]
    by = Counter(r["src"] for r in rows)
    print("n =", len(rows), dict(by))
    for src in by:
        v = [r["s"] for r in rows if r["src"] == src]
        print(f"  {src:14s} n={len(v):4d} med={np.median(v):.3f} p90={np.percentile(v,90):.3f}")

if __name__ == "__main__":
    a = sys.argv[1]
    if a == "all":
        gen(); verify(); bench(); label(); materialize(); status()
    else:
        {"port": port, "gen": gen, "verify": verify, "fresh_sets": fresh_sets,
         "bench": bench, "label": label, "materialize": materialize,
         "status": status}[a]()
