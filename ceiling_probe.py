#!/usr/bin/env python3
"""Ceiling probe: how much cliché/bland signal is linearly readable from a FROZEN
embedding (Octen-Embedding-0.6B, :8935) with shallow heads?

FINDINGS SHAPING THE DESIGN:
- 16,829 windows mined from ALL raw teacher shards (kimi, deepseek-v4, minimax,
  mistral, qwen3) contain ZERO census-cliché-dense windows — the cliché class is
  extinct in our holdings. Positives must be manufactured.
- ling30 (kimi-anchored judge) REFUSES to write generic prose even when asked —
  its RLHF rewards non-generic style. Use Qwen3-0.6B-Base (:8937, the same maker
  that built the 298-sentence generic cloud) for synthetic positives.

Positives:
- kind=babble: Qwen3-0.6B continuing plain story openers -> generic/bland paragraphs
- kind=cliche: Qwen3-0.6B continuing pairs of the 20 HAND-WRITTEN gold cliché
  sentences -> worn paragraphs (census-verified >=2 tics)
Honesty anchor: the 20 gold sentences held out — sentence-level head trained on
generated positives must still rank gold clichés above fresh.

Servers: :8935 Octen embeddings, :8937 Qwen3-0.6B-Base generation (both --embedding/-ngl 99).
Stages: build | genpara | verify | embed | probe | distill | report
"""
import ast, json, os, random, re, sys, urllib.request
import numpy as np

W = os.path.dirname(os.path.abspath(__file__)) + "/"
D = W + "battery_data/"
SEED, NWIN = 7, 110
URL_E = "http://localhost:8935/v1/embeddings"
URL_G = "http://localhost:8937/v1/completions"
INSTR = "Instruct: Retrieve story sentences that are generic or cliched, similar in meaning to the query\nQuery: "
word_re = re.compile(r"[a-z0-9']+"); split_re = re.compile(r"(?<=[.!?])\s+|\n+")
toks = lambda s: word_re.findall(s.lower())
first_words = lambda t, n: " ".join(t.split()[:n])

OPENERS = [
    "The lighthouse keeper had not seen a ship in three weeks.",
    "She found the letter behind the bookshelf, yellowed and unopened.",
    "Once upon a time, in a village by the sea, there lived a fisherman who",
    "Detective Ruiz knew the case was closed, but something still bothered him.",
    "When the power went out, the whole street went dark.",
    "The old dog waited by the gate every afternoon.",
    "In the year 2147, the last library on Earth",
    "He counted the money twice and hid it under the floorboards.",
    "The market opened at dawn, as it had for two hundred years.",
    "Nobody in the town remembered when the fountain had stopped working.",
    "The train was late again, and she waited on the empty platform.",
    "He inherited the farm from an uncle he had never met.",
]

def gold_cliches():
    src = open(W + "embedding_gate.py").read()
    m = re.search(r"CLICHES = (\[.*?\])\n", src, re.S)
    return ast.literal_eval(m.group(1)) if m else []

# ---------- build ----------
def build():
    rng = random.Random(SEED)
    fresh_k, fresh_l = [], []
    for f in (D + "source/genre500_uscrub_train.jsonl", D + "source/genre500_uscrub_val.jsonl"):
        for line in open(f): fresh_k.append(first_words(json.loads(line)["text"], NWIN))
    for line in open(D + "train_3turn.jsonl"):
        fresh_l.append(first_words("\n\n".join(json.loads(line)["pieces"]), NWIN))
    rng.shuffle(fresh_k); rng.shuffle(fresh_l)
    fresh_k, fresh_l = fresh_k[:300], fresh_l[:300]
    bland = json.load(open(W + "generic_corpus.json"))
    fsents = []
    for t in fresh_k + fresh_l:
        fsents += [s.strip() for s in split_re.split(t) if 8 <= len(toks(s)) <= 40]
    rng.shuffle(fsents); fsents = fsents[:600]
    json.dump({"fresh_kimi": fresh_k, "fresh_laguna": fresh_l, "bland_sents": bland,
               "fresh_sents": fsents, "gold_cliches": gold_cliches()},
              open(D + "ceiling_sets.json", "w"))
    print(f"fresh_kimi={len(fresh_k)} fresh_laguna={len(fresh_l)} bland={len(bland)} "
          f"fresh_sents={len(fsents)} gold={len(gold_cliches())}")

# ---------- genpara (:8937) ----------
CKG = D + "ceiling_genpara.jsonl"
def _gen1(prompt, seed, mx=220):
    body = json.dumps({"prompt": prompt, "temperature": 1.0, "min_p": 0.1,
                       "max_tokens": mx, "stop": ["\n\n"], "seed": seed}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL_G, body, {"Content-Type": "application/json"}), timeout=120))
    return first_words(r["choices"][0]["text"].strip(), NWIN)

def genpara(n_babble=160, n_cliche=200):
    rng = random.Random(SEED); gold = gold_cliches()
    jobs = []
    for i in range(n_babble):
        jobs.append(("babble", OPENERS[i % len(OPENERS)], SEED * 1000 + i))
    for i in range(n_cliche):
        s = " ".join(gold[(2 * i) % len(gold):(2 * i) % len(gold) + 2])
        jobs.append(("cliche", s, SEED * 2000 + i))
    done = set()
    if os.path.exists(CKG):
        for l in open(CKG): done.add(json.loads(l)["i"])
    ok = 0
    with open(CKG, "a") as f:
        for i, (kind, opener, sd) in enumerate(jobs):
            if i in done: continue
            t = _gen1(opener + " ", sd)
            if len(toks(t)) < 40: t = _gen1(opener + " ", sd + 777777)  # retry
            f.write(json.dumps({"i": i, "kind": kind, "opener": opener, "text": t}) + "\n")
            ok += bool(len(toks(t)) >= 40)
            if i % 40 == 0: f.flush(); print(f"  {i}/{len(jobs)} ok={ok}")
    print(f"generated ok={ok}/{len(jobs)} ->", CKG)

# ---------- verify (positives must PROVE their class) ----------
def has_loop(text, n=8):
    ws = toks(text); grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n)]
    return len(grams) != len(set(grams))

FORMAT_LEAK = re.compile(r"Given the context|Possible answers|Answer:|Question:|\bStep \d|Note:|Here (is|are) ", re.I)

def verify():
    import dataset_battery as db
    rows = [json.loads(l) for l in open(CKG)]
    for r in rows:
        r["text"] = first_words((r["opener"] + " " + r["text"]).strip(), NWIN)  # cap = fresh cap
        r["tics"] = sum(len(rx.findall(r["text"])) for _, rx in db.CLICHE)
        r["words"] = len(r["text"].split())
    bad = []
    for r in rows:
        why = None
        if r["words"] < 40: why = "short"
        elif has_loop(r["text"]): why = "loop"
        elif FORMAT_LEAK.search(r["text"]): why = "format"
        elif r["kind"] == "cliche" and r["tics"] < 2: why = "lowtics"
        if why: r["drop"] = why; bad.append((r["i"], why))
    keep = [r for r in rows if "drop" not in r and r["kind"] == "cliche"]
    # babble positives: joins of the 298 VERIFIED generic sentences (5/para).
    # The 0.6B firehose leaks QA/math registers too aggressively; only its
    # census-verified cliché continuations are trusted here.
    rngj = random.Random(SEED)
    sent_pool = list(json.load(open(W + "generic_corpus.json")))
    rngj.shuffle(sent_pool)
    for j in range(len(sent_pool) // 5):
        t = " ".join(sent_pool[j * 5:(j + 1) * 5])
        keep.append({"i": 10000 + j, "kind": "babble", "opener": "generic_join", "text": t,
                     "words": len(t.split()),
                     "tics": sum(len(rx.findall(t)) for _, rx in db.CLICHE)})
    for k in ("babble", "cliche"):
        rr = [r for r in keep if r["kind"] == k]
        if not rr: print(f"{k}: EMPTY"); continue
        print(f"{k}: n={len(rr)} words med={sorted(r['words'] for r in rr)[len(rr)//2]} "
              f"tics med={sorted(r['tics'] for r in rr)[len(rr)//2]}")
    from collections import Counter
    print(f"dropped {len(bad)}:", Counter(w for _, w in bad))
    json.dump(keep, open(D + "ceiling_positives.json", "w"))
    print("sample cliché:", [r for r in keep if r['kind'] == 'cliche'][0]["text"][:250])
    print("sample babble:", [r for r in keep if r['kind'] == 'babble'][0]["text"][:250])

# ---------- embed ----------
def embed(texts):
    out = []
    for i in range(0, len(texts), 32):
        body = json.dumps({"input": [INSTR + t for t in texts[i:i + 32]]}).encode()
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            URL_E, body, {"Content-Type": "application/json"}), timeout=180))
        out += [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
    E = np.array(out, dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def embed_stage():
    s = json.load(open(D + "ceiling_sets.json"))
    pos = json.load(open(D + "ceiling_positives.json"))
    bab = [p["text"] for p in pos if p["kind"] == "babble"]
    clh = [p["text"] for p in pos if p["kind"] == "cliche"]
    gs = [x.strip() for t in clh for x in split_re.split(t) if 8 <= len(toks(x)) <= 40]
    rng = random.Random(SEED); rng.shuffle(gs); gs = gs[:600]
    E = {"fresh_kimi": embed(s["fresh_kimi"]), "fresh_laguna": embed(s["fresh_laguna"]),
         "babble_para": embed(bab), "cliche_para": embed(clh), "cliche_sents": embed(gs),
         "bland_sents": embed(s["bland_sents"]), "fresh_sents": embed(s["fresh_sents"]),
         "gold_cliches": embed(s["gold_cliches"])}
    np.savez_compressed(D + "ceiling_emb.npz", **E)
    json.dump(gs, open(D + "ceiling_cliche_sents_texts.json", "w"))
    print("embedded:", {k: v.shape for k, v in E.items()})

# ---------- probe (Part A) ----------
def probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    E = np.load(D + "ceiling_emb.npz")
    heads = {"logreg": LogisticRegression(max_iter=2000),
             "mlp64": MLPClassifier((64,), max_iter=800, random_state=SEED),
             "gbm": GradientBoostingClassifier(random_state=SEED)}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = {}
    def auc(X, y, name):
        row = {}
        for hn, h in heads.items():
            sc = cross_val_score(h, X, y, cv=cv, scoring="roc_auc")
            row[hn] = round(float(sc.mean()), 3)
        print(f"{name:40s} " + "  ".join(f"{k}={v}" for k, v in row.items()))
        res[name] = row
    print("== paragraph level (5-fold CV AUC) ==")
    for tag, fk, pk in (("cliche-seeded-vs-freshKIMI", "fresh_kimi", "cliche_para"),
                        ("cliche-seeded-vs-freshLAGUNA", "fresh_laguna", "cliche_para"),
                        ("generic-join-vs-freshKIMI", "fresh_kimi", "babble_para"),
                        ("generic-join-vs-freshLAGUNA", "fresh_laguna", "babble_para")):
        X = np.vstack([E[pk], E[fk]]); y = [1] * len(E[pk]) + [0] * len(E[fk])
        auc(X, y, tag)
    print("== sentence level ==")
    # clean sentence pool: drop generated sentences that ARE (prefixes of) the gold 20
    gold_norm = {tuple(toks(g)[:8]) for g in json.load(open(D + "ceiling_sets.json"))["gold_cliches"]}
    sent_texts = json.load(open(D + "ceiling_cliche_sents_texts.json"))
    mask = [tuple(toks(t)[:8]) not in gold_norm for t in sent_texts]
    Xs = E["cliche_sents"][np.array(mask)]
    X = np.vstack([Xs, E["fresh_sents"]]); y = [1] * len(Xs) + [0] * len(E["fresh_sents"])
    print(f"   (dropped {len(mask) - sum(mask)} verbatim-gold sentences from train)")
    auc(X, y, "genClicheSents-vs-freshSents")
    Xb = np.vstack([E["bland_sents"], E["fresh_sents"]]); yb = [1] * len(E["bland_sents"]) + [0] * len(E["fresh_sents"])
    auc(Xb, yb, "generic298-vs-freshSents")
    Xc = np.vstack([E["fresh_kimi"], E["fresh_laguna"]]); yc = [1] * len(E["fresh_kimi"]) + [0] * len(E["fresh_laguna"])
    auc(Xc, yc, "CONTROL freshKIMI-vs-freshLAGUNA")
    # within-generator control: Qwen cliché-seeded vs Qwen non-cliché babble (same model!)
    qb = []
    for l in open(CKG):
        r = json.loads(l)
        if r["kind"] != "babble": continue
        t = first_words((r["opener"] + " " + r["text"]).strip(), NWIN)
        if len(toks(t)) >= 40 and not has_loop(t) and not FORMAT_LEAK.search(t) and not re.search(r"\d", t):
            qb.append(t)
    qb = qb[:71]
    Xq = np.vstack([E["cliche_para"], embed(qb)]); yq = [1] * len(E["cliche_para"]) + [0] * len(qb)
    auc(Xq, yq, f"CONTROL cliche-vs-QwenBabble (same gen, n={len(qb)})")
    # length control: gold vs SHORT fresh sentences (<=15 tokens)
    sets = json.load(open(D + "ceiling_sets.json"))
    short_f = [x for x in sets["fresh_sents"] if len(toks(x)) <= 15][:20]
    Xg2 = np.vstack([E["gold_cliches"], embed(short_f)]); yg2 = [1] * 20 + [0] * len(short_f)
    auc(Xg2, yg2, "CONTROL gold-vs-LENGTH-MATCHED-fresh")
    print("== GOLD holdout (train on generated sents -> test 20 hand-written) ==")
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    ng = len(E["gold_cliches"])
    Xg = np.vstack([E["gold_cliches"], E["fresh_sents"][:ng]]); yg = [1] * ng + [0] * ng
    g = round(float(roc_auc_score(yg, lr.predict_proba(Xg)[:, 1])), 3)
    print(f"{'gold-vs-fresh(logreg, held-out class)':40s} auc={g}")
    res["gold_holdout_logreg"] = g
    json.dump(res, open(D + "ceiling_probeA.json", "w"), indent=1)

# ---------- distill (Part B, :8936 judge) ----------
CK = D + "ceiling_judge_labels.jsonl"
def distill():
    s = json.load(open(D + "ceiling_sets.json"))
    import dataset_battery as db
    pos = json.load(open(D + "ceiling_positives.json"))
    paras = ([p["text"] for p in pos if p["kind"] == "cliche"] +
             [p["text"] for p in pos if p["kind"] == "babble"] +
             s["fresh_kimi"] + s["fresh_laguna"])
    def judge_para(t):
        ss = [x for x in split_re.split(t) if 8 <= len(toks(x)) <= 40] or [t]
        v = [db.judge_sentence(x) for x in ss]; v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None
    done = set()
    if os.path.exists(CK):
        for l in open(CK): done.add(json.loads(l)["i"])
    with open(CK, "a") as f:
        n = 0
        for i, t in enumerate(paras):
            if i in done: continue
            f.write(json.dumps({"i": i, "score": judge_para(t)}) + "\n")
            n += 1
            if n % 100 == 0: f.flush(); print(f"  {i}/{len(paras)}")
    print("labeled ->", CK)

def report():
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from scipy.stats import spearmanr
    s = json.load(open(D + "ceiling_sets.json")); E = np.load(D + "ceiling_emb.npz")
    labels = {json.loads(l)["i"]: json.loads(l)["score"] for l in open(CK)}
    pos = json.load(open(D + "ceiling_positives.json"))
    clh = [p["text"] for p in pos if p["kind"] == "cliche"]
    bab = [p["text"] for p in pos if p["kind"] == "babble"]
    X = np.vstack([E["cliche_para"], E["babble_para"], E["fresh_kimi"], E["fresh_laguna"]])
    src = (["clh"] * len(E["cliche_para"]) + ["bab"] * len(E["babble_para"]) +
           ["fk"] * len(E["fresh_kimi"]) + ["fl"] * len(E["fresh_laguna"]))
    idx = [i for i in range(len(src)) if i in labels and labels[i] is not None]
    X, y, src = X[idx], np.array([labels[i] for i in idx]), [src[i] for i in idx]
    rng = np.random.default_rng(SEED); perm = list(rng.permutation(len(y)))
    tr, te = perm[:int(.7 * len(y))], perm[int(.7 * len(y)):]
    out = {"n": len(y), "score_range": [round(float(y.min()), 3), round(float(y.max()), 3)],
           "mean_by_src": {k: round(float(np.mean([y[i] for i in range(len(src)) if src[i] == k])), 3)
                           for k in ("clh", "bab", "fk", "fl")}}
    for name, m in (("ridge", Ridge(1.0)), ("mlp64", MLPRegressor(hidden_layer_sizes=(64,), max_iter=1500, random_state=SEED))):
        m.fit(X[tr], y[tr]); p = m.predict(X[te])
        out[name] = {"heldout_spearman": round(float(spearmanr(y[te], p).statistic), 3),
                     "mae": round(float(np.abs(y[te] - p).mean()), 4)}
        print(name, out[name])
    json.dump(out, open(D + "ceiling_probeB.json", "w"), indent=1)

if __name__ == "__main__":
    {"build": build, "genpara": genpara, "verify": verify, "embed": embed_stage,
     "probe": probe, "distill": distill, "report": report}[sys.argv[1]]()
