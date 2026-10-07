#!/usr/bin/env python3
"""Dataset battery: genre500 vs minicpm5-sft3-3turn — census-v2 tics, diversity, judge.

Stages (run separately; server only needed for `judge`):
    python3 dataset_battery.py census   # tic census-v2 + diversity stats
    python3 dataset_battery.py judge    # LLM judge over stratified sentence samples
    python3 dataset_battery.py report   # table + md

Instrument notes (honesty block, also copied into the report):
- census-v2 is a RECONSTRUCTED instrument (original /tmp/uncanny_census.py lived on a
  recycled box). It is applied identically to both corpora, so the COMPARISON is valid;
  absolute rates are NOT comparable to the historical 1.16/1.41/1kw numbers.
- judge = arin/ling30-m4r16-ud-q4km.gguf, sentence-level pG/(pG+pS), few-shot prefix
  copied verbatim from judge_probe.py (thr 0.12 convention). genre500 is the judge's
  own training positive class -> its score is circular, reported for reference only.
- judge sample: 300 docs/corpus, evenly spaced over doc order, 1 random eligible
  sentence (8-40 tokens) per doc, seed 7. Covers the full range, no cherry-picking.
"""
import hashlib, json, os, random, re, sys, urllib.request, zlib

W = os.path.dirname(os.path.abspath(__file__)) + "/"
D = W + "battery_data/"
SEED, N_JUDGE_DOCS = 7, 300
THR = 0.12
word_re = re.compile(r"[a-z0-9']+")
split_re = re.compile(r"(?<=[.!?])\s+|\n+")
toks = lambda s: word_re.findall(s.lower())

CORPORA = {
    "genre500": [D + "source/genre500_uscrub_train.jsonl", D + "source/genre500_uscrub_val.jsonl"],
    "sft3":     [D + "train_3turn.jsonl", D + "valid_3turn.jsonl"],
}

# ---------------- census-v2 (reconstructed tic list, quote-stripped text) ----------------
QUOTED = re.compile(r'["\u201c][^"\u201d]*["\u201d]')
def strip_q(t): return QUOTED.sub(" ", t)

def P(pat): return (pat, re.compile(pat, re.I))
CLICHE = [
    P(r"\bstood sentinel\b"), P(r"\bheart (pounded|raced|hammered|thudded) in (her|his|my|their) chest\b"),
    P(r"\bbreath (he|she|they|I) didn't know (he|she|they|I) was holding\b"), P(r"\blet out a breath\b"),
    P(r"\bchill ran down\b"), P(r"\bshivers? ran down\b"), P(r"\bsilence was deafening\b"),
    P(r"\blittle did (he|she|they|I)\b"), P(r"\btime (seemed to stand still|stood still)\b"),
    P(r"\beyes met across\b"), P(r"\bsun dipped below the horizon\b"), P(r"\bnever be the same\b"),
    P(r"\blong shadows across\b"), P(r"\brain fell softly\b"), P(r"\btwinkled like stars\b"),
    P(r"\bwind whisper(ed|ing)\b"), P(r"\bfull of promise\b"), P(r"\ba testament to\b"),
    P(r"\bthe air was thick with\b"), P(r"\bhung (heavy|thick) in the air\b"),
    P(r"\bdanced (in|on|through) the\b"), P(r"\bpalpable\b"),
]
UNCANNY = [
    P(r"\bsomething (shifted|stirred|watched|hung|crackled)\b"),
    P(r"\bthe air (hummed|crackled|buzzed|shimmered|thickened|tasted)\b"),
    P(r"\bas if the (world|universe|house|room|forest) itself\b"),
    P(r"\breality (bent|frayed|thinned|warped|blurred)\b"),
    P(r"\bunnerv(ing|ed)\b"), P(r"\bwrongness\b"), P(r"\bprickl(e|ed|ing)\b"),
    P(r"\bwatched (her|him|them) from\b"), P(r"\btoo (quiet|still|perfect)\b"),
    P(r"\bhummed with\b"), P(r"\b(silence|stillness) pressed\b"),
]
TIC_FAMILIES = {"cliche": CLICHE, "uncanny": UNCANNY}

# ---------------- loading ----------------
def load(name):
    docs = []
    for f in CORPORA[name]:
        for line in open(f):
            r = json.loads(line)
            docs.append(r["text"] if name == "genre500" else "\n\n".join(r["pieces"]))
    return docs

def sha8(f): return hashlib.sha256(open(f, "rb").read()).hexdigest()[:8]

# ---------------- census + diversity ----------------
def census(name):
    docs = load(name)
    rows = []
    for t in docs:
        t = strip_q(t)
        w = max(len(toks(t)), 1)
        fam = {f: sum(len(rx.findall(t)) for pat, rx in pats) for f, pats in TIC_FAMILIES.items()}
        rows.append({"w": w, **fam})
    tot_w = sum(r["w"] for r in rows)
    pooled = {f: 1000 * sum(r[f] for r in rows) / tot_w for f in TIC_FAMILIES}
    per_doc = {f: 1000 * sum(r[f] for r in rows) / len(rows) / (tot_w / len(rows)) for f in TIC_FAMILIES}
    ws = sorted(r["w"] for r in rows)
    return {
        "docs": len(rows), "words_total": tot_w,
        "words_mean": round(tot_w / len(rows)), "words_median": ws[len(ws) // 2],
        "tic_rate_pooled_per_1kw": {k: round(v, 2) for k, v in pooled.items()},
        "tic_rate_meandoc_per_1kw": {k: round(v, 2) for k, v in per_doc.items()},
        "docs_with_zero_tics": sum(1 for r in rows if sum(r[f] for f in TIC_FAMILIES) == 0),
    }

def diversity(name):
    docs = load(name)
    words_docs = [toks(t) for t in docs]
    # per-doc distinct-2/3
    def dn(ws, n):
        gs = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
        return len(set(gs)) / len(gs) if gs else 1.0
    d2 = sum(dn(w, 2) for w in words_docs) / len(words_docs)
    d3 = sum(dn(w, 3) for w in words_docs) / len(words_docs)
    # cross-doc duplicate 8-grams
    seen, dup = set(), 0
    total8 = 0
    for w in words_docs:
        gs = set(tuple(w[i:i + 8]) for i in range(len(w) - 7))
        for g in gs:
            total8 += 1
            if g in seen: dup += 1
            seen.add(g)
    blob = " ".join(docs)
    comp = len(zlib.compress(blob.encode(), 9) ) * 8 / max(len(blob), 1)
    return {"distinct2_meandoc": round(d2, 3), "distinct3_meandoc": round(d3, 3),
            "doc8gram_dup_pct": round(100 * dup / total8, 2), "unique_doc8grams": len(seen),
            "zlib_bits_per_char": round(comp, 3),
            "corpus_tokens": sum(len(w) for w in words_docs),
            "corpus_unique_tokens": len(set(t for w in words_docs for t in w))}

# ---------------- judge ----------------
PRE = ("You classify story sentences. G = GENERIC, overused cliche phrasing. "
       "S = SPECIFIC, fresh concrete detail.\n\n"
       "Sentence: The sun dipped below the horizon, casting long shadows across the quiet town.\n"
       "Answer: G\n\n"
       "Sentence: The ticket punch bit through the wet cardboard with a sound like a knuckle cracking.\n"
       "Answer: S\n\n")
URL = "http://localhost:8936/v1/chat/completions"

def judge_sentence(s):
    body = json.dumps({
        "messages": [{"role": "user", "content": PRE + f"Sentence: {s}\nAnswer:"}],
        "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
    }).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, body, {"Content-Type": "application/json"}), timeout=120))
    lp = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    import numpy as np
    pg = sum(np.exp(t["logprob"]) for t in lp if t["token"].strip().upper().startswith("G"))
    ps = sum(np.exp(t["logprob"]) for t in lp if t["token"].strip().upper().startswith("S"))
    return pg / (pg + ps) if pg + ps > 1e-9 else None

def sample_sentences(name, n_docs=N_JUDGE_DOCS):
    docs = load(name)
    rng = random.Random(SEED)
    idxs = [round(i * (len(docs) - 1) / (n_docs - 1)) for i in range(n_docs)]  # even coverage
    out = []
    for i in idxs:
        sents = [s for s in split_re.split(docs[i]) if 8 <= len(toks(s)) <= 40]
        if sents:
            out.append({"doc": i, "sentence": rng.choice(sents)})
    return out

CKPT = D + "judge_scores.jsonl"
def run_judge():
    done = set()
    if os.path.exists(CKPT):
        for l in open(CKPT):
            r = json.loads(l); done.add((r["corpus"], r["doc"]))
    jobs = [(c, j) for c in CORPORA for j in sample_sentences(c) if (c, j["doc"]) not in done]
    print(f"to score: {len(jobs)} (already done: {len(done)})")
    with open(CKPT, "a") as f:
        for n, (c, j) in enumerate(jobs):
            sc = judge_sentence(j["sentence"])
            f.write(json.dumps({"corpus": c, "doc": j["doc"], "score": sc}) + "\n")
            if n % 50 == 0: f.flush(); print(f"  {n}/{len(jobs)}")
    print("judge done ->", CKPT)

# ---------------- report ----------------
def report():
    res = {c: {"census": census(c), "diversity": diversity(c)} for c in CORPORA}
    scores = {c: [] for c in CORPORA}
    for l in open(CKPT):
        r = json.loads(l)
        if r["score"] is not None: scores[r["corpus"]].append(r["score"])
    import statistics as st
    def boot_ci(xs, B=2000):
        rng = random.Random(SEED); ms = []
        for _ in range(B):
            ms.append(st.mean(rng.choice(xs) for _ in xs))
        ms.sort(); return ms[int(.025 * B)], ms[int(.975 * B)]
    js = {}
    for c in CORPORA:
        xs = scores[c]
        lo, hi = boot_ci(xs)
        js[c] = {"n": len(xs), "mean": round(st.mean(xs), 3), "ci95": [round(lo, 3), round(hi, 3)],
                 "flag@0.12": round(100 * sum(x >= THR for x in xs) / len(xs), 1),
                 "misses": sum(1 for l in open(CKPT) if json.loads(l)["corpus"] == c and json.loads(l)["score"] is None)}
    out = {"files": {c: {os.path.basename(f): sha8(f) for f in CORPORA[c]} for c in CORPORA},
           "census": {c: res[c]["census"] for c in CORPORA},
           "diversity": {c: res[c]["diversity"] for c in CORPORA},
           "judge": js,
           "instrument_notes": __doc__.split('"""')[0]}
    json.dump(out, open(W + "battery_results.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    {"census": lambda: [print(c, json.dumps(census(c), indent=1)) for c in CORPORA],
     "judge": run_judge,
     "report": report}[sys.argv[1]]()
