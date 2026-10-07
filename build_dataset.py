#!/usr/bin/env python3
"""Full-scale bland-prose dataset: 1 row per taxonomy leaf, >=1 pass per genre.

Source : baiango/genre-taxonomy-sfw (~197k genre+subgenre leaves; parent genre
         kept per row). Prompt asks "in English" so foreign-named genres yield
         English output (0.6B otherwise mirrors the genre name's language).
Model  : Qwen3-0.6B-Base.Q5_K_S via llama-server (URL overridable for GPU box)
Frame  : "Write a {name} story.\n\n"  (prose register; bare openers -> math reg)
Sampling: plain temp 1.0 / min_p 0.1, per-leaf deterministic seeds (no rep-pen)
Row    : {parent, genre, prompt, text, n_tok, k, seed, verdict[, gates, review]}
         "text" = VERBATIM model output incl. newlines — the single source of
         truth (the only field that cannot be re-derived). "prose" = derived
         clean copy (strip + whitespace-normalize); all post-processing is a
         pure function of text, so any bug there is repaired offline
         (restrip.py) — NEVER wipe for a post-processing bug; wipes are only
         for generation-affecting changes (prompt/sampler/server/seeds).
         Normalize idiom: " ".join(text.split()) — idempotent, so rows from
         every schema era repair identically.
Verdict: pass          -> text usable as-is (no style wrapper)
         style         -> leading title/byline/separator stripped; cleaned copy
                          in "prose", original "text" kept verbatim
         failed + gates -> rejected; list of ALL failed gates, evaluated on the
                          text that would be used ("prose" if present else "text")
         dupe          -> simhash-k duplicate of an earlier row this run
         exhausted     -> still failed after ATTEMPTS fresh-seed draws (poison
                          leaves); "att" = draws used; cleanup pass can retry
Review : rows that PASS all gates but sit near a threshold get "review":
         [coherence-low|english-low|diacritics|len-edge|loop-ish|style-heavy|echo]
         [lc-start] = stripped prose starts lowercase -> stripper may have eaten
         — shipped, flagged for the post-hoc eyeball pass, never rejected.
Policy : retry <=4 attempts until pass/style; else record last attempt.

Resume: rows for genres already in OUT are skipped; k-set restored from OUT so
         dedupe survives restarts. Progress every 500.
"""
import json, os, re, sys, time, random, urllib.request, concurrent.futures as cf
import ceiling_probe as cp
import make_generic as mg

URL = os.environ.get("BLAND_URL", "http://localhost:8937/v1/completions")
OUT = os.environ.get("BLAND_OUT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "battery_data", "bland_prose_full.jsonl")
WORKERS = int(os.environ.get("BLAND_WORKERS", "4"))
SHARD = os.environ.get("BLAND_SHARD", "")          # "i/n" -> this process takes idx%n==i
ATTEMPTS = int(os.environ.get("BLAND_ATTEMPTS", "40"))  # retry-until-pass: most stop at 1-2; tail
     # draws fresh seeds (cost only where needed); cap = poison-leaf valve.
BASE_SEED = 777000000

# --- style wrapper stripping --------------------------------------------
# Leading markdown-ish title/byline/separator block = STYLE issue, not prose
# failure. Strip it into "prose"; keep original "text" verbatim.
# (10-02: 120/161 pass rows carried ** artifacts before this existed.)
_TAIL = (r"(?:\s*(?:\*(?:\s*\*){2,}|-{3,}|_{3,})"            # * * * / --- / ___
         r"|\s*\*{0,2}\s*(?:By|Author)\s*:?\s*(?:\*{0,2})[A-Z][^*:\n]{1,60}\*{0,2})*")  # byline/author: By:/By X/*By X*/Author:
# metadata label lines the model emits after the title (Setting:/Tags:/Author: ...)
# \*{0,2} on both sides: labels arrive plain, *Label:* and **Label:** alike
_LABEL = (r"\*{0,2}\s*(?:Setting|Author|Tags?|Genre|Rating|Word Count|Prompt|Summary|Plot Summary|"
          r"Characters?|Theme|Tone|Style|Date|(?:A\s+|An\s+)?Brief Introduction)\s*:\*{0,2}[ \t]*[^\n]{0,100}(?:\n|$)")
                                # 10-07: 'Brief Introduction:' census shape — strip-
                                # side only, deliberately NOT in _TEMPLATE_LBL
                                # (asymmetric doctrine: strip=cleanup, gate=judgment)
# structural heading lines: "Chapter 1: The Whispering Woods", "Part 2", bare
# "Prologue"/"Epilogue: The Beginning". Trailing \n REQUIRED: prose sentences
# ("Chapter 1 covered the war.", "Prologue written, she sealed…") never match —
# they don't end at a newline.

# 10-03 Fix G (fix5 run6 section 6): >100-char label values must not survive
# as "Setting: ..." prose (template-gate tripwire). STYLE_LEAD is ^-anchored,
# so this lane only fires on a LEADING label: strip the prefix, keep value.
_LBL_PREFIX = '\\*{0,2}\\s*(?:Setting|Author|Tags?|Genre|Rating|Word Count|Prompt|Summary|Plot Summary|Characters?|Theme|Tone|Style|Date|(?:A\\s+|An\\s+)?Brief Introduction)\\s*:\\*{0,2}[ \\t]*'
_HEAD = (r"(?:(?i:(?:Chapter|Part|Scene))\s+(?:\d+|(?i:one|two|three|four|five|six|seven|eight|nine|ten|"
         r"eleven|twelve))|(?:Prologue|Epilogue))(?:\s*:[^\n]{0,100})?\n")  # (?i:): census
                                  # caught 'Chapter One: The…' x6 — display was
                                  # lowercased, patterns were not
# standalone ALL-CAPS italic line right after a title (*ABYSS VOCAL* = repeated
# section title); gated on n>0 in the loop so it can NEVER hit raw prose openers
_CAPS_LINE = re.compile(r"\s*\*[A-Z0-9][A-Z0-9\s'&\-]{2,60}\*\s*\n")
_EMB_HEAD = re.compile(   # mid-text section-heading line ('**The Founding of
    r"(?m)^[ \t]*\*\*(?![^*\n]{0,78}:[ \t]*\*\*)"   # AirGate**') glues into prose
    r"[^*\n]{1,80}\*\*[ \t]*\n?")  # on collapse; cut only after a wrapper (n>0),
     # same doctrine as _CAPS_LINE (never raw openers). Negative lookahead keeps
     # '**Plot:**'/'**Themes:**' LABEL lines (colon-terminated bold) OUT of the
     # cut — they are the template gate's evidence; eating them re-opens ALASKA.
# 10-09 heading lane (user bug, box 54420385 row seed 911000200): _HEAD was only
# ever applied at the CURRENT START of t2 (leading chain + blocked-chain sweep),
# so 'Title: The Silent Scream / Chapter 1: ...' shipped with Chapters 2..8
# still inside `prose`. This lane is LINE-anchored and position-independent:
# a standalone 'Chapter N[: title]' line anywhere is a wrapper, not prose.
# False-positive guard = the separator: the title tail is only legal after
# [.:-–—], so 'Part 2 of the plan was simple.' (no separator) can never match;
# 'Chapter 2 was the worst year.' likewise. Roman numerals included (STYLE_LEAD's
# heading alt only had arabic+words, so 'Chapter II: ...' led nowhere).
_HBODY = (r"(?:(?i:(?:Chapter|Part|Scene|Episode|Act))\s+(?:\d+|[IVXLC]+|"
          r"(?i:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve))"
          r"|(?i:Prologue|Epilogue))")
_HEAD_LINE = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?\*{0,2}[ \t]*" + _HBODY +
    r"[ \t]*\*{0,2}(?:[ \t]*[.:\-\u2013\u2014][ \t]*\*{0,2}[ \t]*[^\n]{0,100})?"
    r"[ \t]*\*{0,2}[ \t]*(?:\n|$)"          # \*{0,2} after the sep: the
    #                                           # '**Epilogue:** <prose>' form
    # glued-prose form: 'Epilogue: The Chronoscape, now a beacon ... continues'
    # = marker + a whole paragraph on ONE line (tail > 100 chars => not a title).
    # Cut the MARKER ONLY, keep the prose (alt 1 would eat the prose as a title).
    r"|^[ \t]*(?:#{1,6}[ \t]*)?\*{0,2}[ \t]*" + _HBODY +
    r"[ \t]*\*{0,2}[:.][ \t]*\*{0,2}[ \t]*(?=[^\n]{101,})")  # (?m) at pos 0
# 10-03 heading lane (fix5 run6 section 6): the legacy title alt missed
# label-only bolds, quoted/bold chapter forms and "Chapter N: ..." heading
# lines; the label alt now requires a line end (was \n? -> mid-word prefix
# cuts) and _LBL_PREFIX handles over-long label values. Readable
# construction lives in fix5_dry.py section 6; port_verify.py asserts this
# value byte-equal to the harness result.
STYLE_LEAD = re.compile('^\\s*(?:\\*\\*[A-Za-z][A-Za-z0-9\' \\u2019&.\\-]{0,30}:\\s*\\*\\*[ \\t]*|Title\\s*:\\s*(?=\\*\\*)|\\*\\*(?!(?!(?:(?:Setting|Author|Tags?|Genre|Rating|Word Count|Prompt|Summary|Plot Summary|Characters?|Theme|Tone|Style|Date|Track Title|(?:A\\s+|An\\s+)?Brief Introduction))\\s*:)[A-Za-z\' -]{1,25}:\\s*\\*\\*)(?:Title\\s*:\\s*)?[^\\n]{1,160}?\\*\\*|\\*\\*Title\\s*:\\s*\\"[^\\"\\n]{1,120}\\"\\s*|\\*\\*Title\\s*:\\s*[^*\\"\\n]{1,120}?(?=\\s+(?:In|On|At|Once|One|There|When|As|After|Before|The|A|An)\\b)|\\*{0,2}(?:Chapter|Part|Scene|Episode|Act)\\s*:\\s*[^\\n]{1,120}?\\*{0,2}[ \\t]*\\n|Title\\s*:\\s*[^\\n]*(?=\\n|$)|#{1,6}\\s*[^\\n]{1,120}\\n|(?:(?i:(?:Chapter|Part|Scene))\\s+(?:\\d+|(?i:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve))|(?:Prologue|Epilogue))(?:\\s*:[^\\n]{0,100})?\\n|[A-Z][a-z]+(?:,\\s*[A-Z][a-z]+){1,5}\\s*\\n|(?i:A\\b[^.!?\\n]{0,120}?\\b(?:story|tale)\\b[^.!?\\n]{0,120}\\n(?=\\s*\\*{0,2}\\s*(?:By|Author)\\b))|\\*{0,2}\\s*(?:Setting|Author|Tags?|Genre|Rating|Word Count|Prompt|Summary|Plot Summary|Characters?|Theme|Tone|Style|Date|(?:A\\s+|An\\s+)?Brief Introduction)\\s*:\\*{0,2}[ \\t]*[^\\n]{0,100}(?:\\n|$)|\\*{0,2}\\s*(?:Setting|Author|Tags?|Genre|Rating|Word Count|Prompt|Summary|Plot Summary|Characters?|Theme|Tone|Style|Date|(?:A\\s+|An\\s+)?Brief Introduction)\\s*:\\*{0,2}[ \\t]*)(?:\\s*(?:\\*(?:\\s*\\*){2,}|-{3,}|_{3,})|\\s*\\*{0,2}\\s*(?:By|Author)\\s*:?\\s*(?:\\*{0,2})[A-Z][^*:\\n]{1,60}\\*{0,2})*\\s*')

def style_strip(t):
    return _style_strip_n(t)[0]

def _style_strip_n(t):
    """(prose|None, blocks_stripped) — blocks can chain (ToC-style openers)."""
    mg.echo_cuts = 0                 # per-strip ledger: echo/spec/ident cuts
    t2, n = t, 0                     #   this pass flagged via mg (10-07)
    t2, n_head = _HEAD_LINE.subn("", t2)  # 10-09 standalone heading lines ANYWHERE
    n += n_head                      # (was: only at the current start of t2, so
                                     #  'Chapter 2..8' survived into `prose`);
                                     #  NOT named nh: the loop below rebinds nh
    for _ in range(5):                  # chained wrappers: title+byline, tags, Setting:
        m = STYLE_LEAD.match(t2)
        if not m and n:
            m = _CAPS_LINE.match(t2)    # all-caps repeat line only AFTER a title
        if not m:
            if n:
                t2, nh = _EMB_HEAD.subn(" ", t2)   # mid-text section headings
                n += nh                            # (verbatim lines survive here;
                                                   #   collapse would glue them)
            t3, nc = mg.cut_meta_leak(t2)  # meta-essay preamble; salvageable when
            if nc:                         # a story follows (10-06 ABANDON NOIR)
                t2, n = t3, n + nc
                continue
            break
        t2 = t2[len(m.group(0)):]
        n += 1
    if n:                               # blocked chain: a non-title line (numbered
        _blk = (re.compile(r"\s*" + _LABEL),  # beat, dateline) before metadata;
                re.compile(r"\s*" + _HEAD))   # sweep standalone label/heading
        while True:                     # lines — gated on n>0 so raw prose
            m = next((x.match(t2) for x in _blk if x.match(t2)), None)
            if not m: break
            t2 = t2[m.end():]
            n += 1
    if not n: return None, 0
    if not t2.strip():
        # 10-09: empty because the HEADING lane ate everything = a heading-only
        # text (outline), NOT the pure-meta case. "" (not None) so the caller
        # gates the empty target (fails `len`) instead of falling back to raw,
        # which would ship the outline with verdict=pass.
        return ("", n) if n_head else (None, n)
    t2 = re.sub(r"\*(?:\s*\*){2,}", " ", t2)      # spaced * * * scene breaks
    t2 = re.sub(r"-{3,}|_{3,}", " ", t2)
    t2 = t2.replace("**", "").replace("*", "")     # residual bold/italic marks
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", t2)]
    return "\n\n".join(p for p in paras if p), n

def gen(prompt, seed, mx=240):
    # Chain pinned EXPLICITLY (no server defaults): top_k/top_p truncation is load-bearing
    # for the 0.6B (truly-plain chain collapsed pass to 1.4% on the box). temp 1.0 + min_p 0.1
    # per plain-sampling rule; rep_pen absent from chain = 1.0 no-op. Deterministic seed.
    body = json.dumps({"model": "q", "prompt": prompt, "max_tokens": mx,
                       "samplers": ["top_k", "top_p", "min_p", "temperature"],
                       "top_k": 40, "top_p": 0.95, "temperature": 1.0, "min_p": 0.1,
                       "seed": seed}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    for _ in range(3):
        try:
            r = urllib.request.urlopen(req, timeout=180)
            return json.load(r)["choices"][0]["text"]
        except Exception:
            time.sleep(2)
    return ""

def attempt(leaf, idx, att):
    prompt = f"Write a {leaf} story in English.\n\n"
    text = gen(prompt, BASE_SEED + idx * 10 + att)   # VERBATIM, newlines kept —
    if len(cp.toks(text)) < 30:                      #   the only non-rederivable
        text = gen(prompt, BASE_SEED + idx * 10 + att + 555555)
    norm = " ".join(text.split())                    # gates/simhash want \s-free
    row = {"genre": leaf, "prompt": prompt, "text": text,
           "n_tok": len(cp.toks(norm)), "k": mg.sl.h(norm),
           "seed": BASE_SEED + idx * 10 + att}
    prose, n_stripped = _style_strip_n(text)         # strip on VERBATIM: the model's
    flat = " ".join(prose.split()) if prose is not None else None   # gates want \s-free
    target = flat if flat is not None else norm      # text the greedy title pattern
    why, soft = mg.gates_all(target, raw=text)       #   eats ~100 chars of opening
    if why:
        row.update(verdict="failed", gates=why)      # saved, failed-marked
        return row
    if n_stripped >= 2: soft.append("style-heavy")   # model was in card/listicle mode
    if prose is not None and prose[:1].islower():
        soft.append("lc-start")                      # stripper may have eaten the opening
    if mg.echo_cuts:
        soft.append("echo-lead")                     # 10-07: echo/spec/ident lead cut
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
        OUT = f"{OUT}.shard{si}of{sn}"      # shard path FIRST so resume reads the per-shard file
    leaves = mg.load_leaves()                        # sorted [(leaf, parent_genre)]
    done, seen_k0 = set(), set()
    if os.path.exists(OUT):
        for l in open(OUT):                          # resume: genres AND dedupe keys
            r = json.loads(l)
            done.add(r["genre"]); seen_k0.add(r["k"])
    pending = [(i, n, p) for i, (n, p) in enumerate(leaves) if n not in done]
    if SHARD:
        pending = [(i, n, p) for i, n, p in pending if i % sn == si]
    print(f"leaves={len(leaves)} done={len(done)} pending={len(pending)} "
          f"workers={WORKERS} shard={SHARD or '-'} url={URL} out={OUT}", flush=True)
    seen_k, n_pass, n_att = seen_k0, 0, 0
    t0 = time.time()
    lock_out = open(OUT, "a")
    def work(arg):
        idx, leaf, par = arg
        row = attempt(leaf, idx, 0)
        att = 1
        while att < ATTEMPTS and (row["verdict"] == "failed" or row["k"] in seen_k):
            row = attempt(leaf, idx, att); att += 1   # fresh seed every draw
        if row["verdict"] == "failed":
            row["verdict"], row["att"] = "exhausted", att
        return {"parent": par, **row}      # parent merged LAST: retry loop replaces row
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for n, row in enumerate(ex.map(work, pending), 1):
            v = row["verdict"]
            if v in ("pass", "style") and row["k"] in seen_k:
                row["verdict"] = "dupe"
            if row["verdict"] in ("pass", "style"):
                seen_k.add(row["k"]); n_pass += 1
            lock_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 500 == 0:
                lock_out.flush()
                print(f"  {n}/{len(pending)} pass={n_pass} "
                      f"rate={n/(time.time()-t0):.2f}/s", flush=True)
    lock_out.flush(); lock_out.close()
    print(f"DONE rows={len(pending)} pass={n_pass} "
          f"elapsed={(time.time()-t0)/3600:.2f}h -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
