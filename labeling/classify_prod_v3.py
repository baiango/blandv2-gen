#!/usr/bin/env python3
"""Production genre labeling v3 — WORLD + ENERGY + TAGS in one direct prompt.

ENERGY asked directly (not collapsed from 5-way votes) — the missing test.
Output schema: {"k","world","energy","tags":[...],"raw"}; "?" = unmappable.
Resumable by k. Stdlib only.

Usage:  python3 classify_prod_v3.py [KEY=VALUE ...] [DRY_RUN=1]
"""
import json, os, re, sys, time, urllib.request, urllib.error, concurrent.futures as cf

CFG = {
    "URL":      "http://127.0.0.1:8899/v1/chat/completions",
    "IN":       "audit100_full.jsonl",
    "OUT":      "audit100_prod_r1.jsonl",
    "MISSES":   "audit100_prod_r1_miss.jsonl",
    "LIMIT":    0,
    "WORKERS":  2,
    "MAX_CHARS": 8000,
    "TEMP":     1.0,      # PLAIN sampling, standing rule: temp1.0 + min_p0.1
    "MIN_P":    0.1,
    "MAXTOK":   45,
    "MODEL":    "ornith",
    "REASONING": 0,
}
WORLD  = ["Contemporary", "Fantasy & Supernatural", "Science Fiction", "Historical & Western"]
ENERGY = ["External", "Relational"]
TAGS   = ["Drama / Slice of Life", "Romance", "Mystery & Crime", "Action & Adventure", "Horror"]
WORLD_KW = {
    "Contemporary":           ["contemporary", "modern day", "real world", "present day", "realistic"],
    "Fantasy & Supernatural": ["fantasy", "fairy tale", "folklore", "magical realism",
                               "paranormal", "supernatural", "magic"],
    "Science Fiction":        ["science fiction", "sci-fi", "scifi", "cyberpunk",
                               "post-apocalyptic", "post apocalyptic", "dystopian", "space",
                               "futuristic"],
    "Historical & Western":   ["historical", "western", "wild west", "period", "medieval",
                               "victorian", "ancient"],
}
ENERGY_KW = {
    "External":   ["external", "danger", "crime", "survival", "mission", "thrill"],
    "Relational": ["relational", "relationship", "love", "family", "inner",
                   "everyday", "personal"],
}
TAG_KW = {
    "Drama / Slice of Life":  ["drama", "slice of life", "slice-of-life", "literary",
                               "coming-of-age", "everyday", "character study", "comedy"],
    "Romance":                ["romance", "romantic", "love story"],
    "Mystery & Crime":        ["mystery", "noir", "crime", "thriller", "detective", "heist",
                               "suspense", "investigation"],
    "Action & Adventure":     ["action", "adventure", "quest"],
    "Horror":                 ["horror", "slasher", "menace", "terror"],
}
WORLD_RE   = re.compile(r"WORLD\s*[:\-]\s*(.+)", re.I)
ENERGY_RE  = re.compile(r"ENERGY\s*[:\-]\s*(.+)", re.I)
TAGS_RE    = re.compile(r"TAGS\s*[:\-]\s*(.+)", re.I)
THINK_RE   = re.compile(r"<think>.*?</think>", re.S)

PROMPT = ("Read the story and fill in the slots.\n\n"
          "WORLD - where the story takes place, exactly one of:\n" +
          "\n".join(WORLD) + "\n\n"
          "ENERGY - what kind of pressure moves the story, exactly one of:\n"
          "External - the plot moves on danger, crime, survival, a mission, or a chase: "
          "something out there to solve, escape, or survive.\n"
          "Relational - the plot moves on love, family, belonging, or inner change: "
          "people and their bonds.\n"
          "(ENERGY must be exactly the word External or the word Relational - "
          "never a genre name.)\n\n"
          "TAGS - every genre element that is clearly present, comma-separated from this list "
          "(or none):\n" +
          "\n".join(f"{t} - " + d for t, d in [
              ("Romance", "part of the plot moves on whether two people end up together"),
              ("Mystery & Crime", "part of the plot moves on an unanswered question: who did it, what happened, how to pull it off"),
              ("Action & Adventure", "part of the plot moves on a physical mission, journey, or escape"),
              ("Horror", "part of the plot moves on an escalating threat; survival at stake"),
              ("Drama / Slice of Life", "inner life, relationships, everyday pressure; no external engine"),
          ]) + "\n\n"
          "Rules:\n"
          "- Judge the story as a whole. If the era isn't clearly past or future, "
          "the world is Contemporary.\n"
          "- Tag an element if it is clearly present, even when it is not the main pressure.\n\n"
          "Answer in exactly this format and nothing else:\n"
          "WORLD: <label>\nENERGY: <External or Relational>\nTAGS: <comma-separated labels or none>\n\n"
          "Story:\n\"\"\"\n{story}\n\"\"\"\n\nAnswer:")

def canon_one(txt, table, labels):
    t = txt.strip().strip('*').strip('"').strip("'").strip('.').strip()
    low = t.casefold()
    if not t or low in {"none", "n/a", "na", "-", "null", "unknown"}:
        return "none"
    for lab in labels:
        if low == lab.casefold():
            return lab
    best, blen = None, 0
    for lab, kws in table.items():
        for kw in kws:
            if kw in low and len(kw) > blen:
                best, blen = lab, len(kw)
    return best

def parse_tags(txt):
    out, seen = [], set()
    for part in re.split(r"[,;]", txt):
        lab = canon_one(part, TAG_KW, TAGS)
        if lab and lab != "none" and lab not in seen:
            seen.add(lab); out.append(lab)
    return out

def call(story, url, key):
    local = "127.0.0.1" in url or "localhost" in url
    body = {"model": CFG["MODEL"], "temperature": CFG["TEMP"], "min_p": CFG["MIN_P"],
            "max_tokens": CFG["MAXTOK"],
            "messages": [{"role": "user", "content": PROMPT.format(story=story)}]}
    if local:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["reasoning"] = {"enabled": bool(CFG["REASONING"])}
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=payload, headers=headers)
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                last = e; time.sleep([5, 15, 30][attempt]); continue
            raise
    raise last

def classify(row, key):
    story = (row.get("prose") or row.get("text") or "")[:CFG["MAX_CHARS"]]
    if not story.strip():
        return {"k": row["k"], "world": "?", "energy": "?", "tags": [],
                "raw": "<empty>", "ok": False}
    try:
        resp = call(story, CFG["URL"], key)
        raw = resp["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return {"k": row["k"], "world": f"<err {e}>", "energy": "?", "tags": [],
                "raw": "", "ok": False}
    raw = THINK_RE.sub("", raw)
    mw, me, mt = WORLD_RE.search(raw), ENERGY_RE.search(raw), TAGS_RE.search(raw)
    w = canon_one(mw.group(1), WORLD_KW, WORLD) if mw else None
    e = canon_one(me.group(1), ENERGY_KW, ENERGY) if me else None
    tags = parse_tags(mt.group(1)) if mt else []
    ok = (w in WORLD) and (e in ENERGY)
    return {"k": row["k"], "world": w if w else "?", "energy": e if e else "?",
            "tags": tags, "raw": raw, "ok": ok}

def main():
    for a in sys.argv[1:]:
        if "=" in a and a != "DRY_RUN=1":
            k, v = a.split("=", 1)
            old = CFG.get(k)
            if isinstance(old, int): v = int(v)
            elif isinstance(old, float): v = float(v)
            CFG[k] = v
    if "DRY_RUN=1" in sys.argv or os.environ.get("DRY_RUN") == "1":
        print(f"CONFIG {json.dumps({k: CFG[k] for k in sorted(CFG)})}")
        print("DRY_RUN prompt:\n" + PROMPT.format(story="STORY HERE"))
        return
    key = None
    if "127.0.0.1" not in CFG["URL"] and "localhost" not in CFG["URL"]:
        envp = os.path.expanduser("~/.pi/agent/openrouter.env")
        if os.path.exists(envp):
            for l in open(envp):
                if l.startswith("OPENROUTER_API_KEY="):
                    key = l.split("=", 1)[1].strip()
        if not key:
            key = json.load(open(os.path.expanduser("~/.pi/agent/auth.json")))["openrouter"]["key"]
        assert key and key.startswith("sk-or-"), "no openrouter key"
    rows = [json.loads(l) for l in open(CFG["IN"])]
    done = set()
    if os.path.exists(CFG["OUT"]):
        for l in open(CFG["OUT"]):
            try: done.add(json.loads(l)["k"])
            except Exception: pass
    todo = [r for r in rows if r["k"] not in done]
    if CFG["LIMIT"]: todo = todo[:CFG["LIMIT"]]
    print(f"model={CFG['MODEL']} url={'local' if key is None else 'openrouter'} "
          f"todo={len(todo)} done_before={len(done)}", flush=True)
    out = open(CFG["OUT"], "a"); mis = open(CFG["MISSES"], "a")
    t0 = time.time(); n = miss = 0
    with cf.ThreadPoolExecutor(CFG["WORKERS"]) as ex:
        for row, res in zip(todo, ex.map(lambda r: classify(r, key), todo)):
            out.write(json.dumps({k: res[k] for k in ("k", "world", "energy", "tags", "raw")},
                                 ensure_ascii=False) + "\n")
            if not res["ok"]:
                mis.write(json.dumps({"k": res["k"], "world": res["world"], "raw": res["raw"]}) + "\n")
                miss += 1
            out.flush(); mis.flush()
            n += 1
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"{n}/{len(todo)} miss={miss} rate={n/el:.2f}/s eta={(len(todo)-n)/max(n/el,.01)/60:.1f}min", flush=True)
    print(f"DONE n={n} miss={miss} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
