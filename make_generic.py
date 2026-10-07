#!/usr/bin/env python3
"""Manufacture the FLUENT-GENERIC class the pool is missing.

User insight: Qwen3-0.6B free generation IS bland prose for free -- that is
literally the provenance of the old 816-babble class. But sample showed much of
it is INCOHERENT salad (non-sequiturs, blanks) -- detectable by incoherence, not
genericity. So: keep free generation, ADD a coherence gate (adjacent-sentence
content-word cohesion) to harvest only the fluent-coherent subset. Rewrite
twins (same paragraph, specifics stripped) stay as a small MATCHED eval
instrument, not pool filler.

Stages:
  free   (:8937) mass bland generation + standard gates + coherence gate
  peek   eyeball 5 kept samples
  pairs  (:8937) rewrite twins for paired eval (~150, held-out instrument)
  report (:8936) TEACHER-FIRST test -- run AFTER scale_labels label finishes:
         free: AUC(judge s of bland_free vs fresh_laguna_x)
         pairs: % of twins ranked generic > fresh by the judge
         If the judge can't separate either, distillation is hopeless: STOP.
"""
import json, os, random, re, sys
import ceiling_probe as cp
import dataset_battery as db
import scale_labels as sl

# words-band upper bound (gates_all "len" gate). Default 240 = v1 semantics,
# byte-identical to the 10-02 calibration. bland-v2 (model-stops spec) sets
# MG_LEN_MAX=700 in its launch env — an anti-dump sanity bound, not a cap.
MG_LEN_MAX = int(os.environ.get("MG_LEN_MAX", "240"))

W, D = cp.W, sl.D
FK_OUT = D + "bland_free.jsonl"
PKP = D + "generic_pairs.jsonl"

# continuation prompts, story register, generic settings (bland-inducing)
SEEDS = ["The town had", "It was the kind of place where", "In the years after the war, the",
         "Every summer the family", "The road out of the valley", "When she was young, the",
         "The factory at the edge of", "By then the village", "The house on the hill had",
         "For as long as anyone remembered, the", "The winter that year",
         "Morning came to the harbor", "The fields behind the church",
         "Nobody in the village", "The train arrived, as it always did,",
         "The river ran through the town"]

_STOP = frozenset(
    "the a an and or but of to in on at for with by from as is are was were be been being "
    "it its this that these those he she they them his her their we you i not no so if "
    "then than there here what which who whom whose when where why how all any both each "
    "few more most other some such only own same too very can will just don should now "
    "has have had do does did into out up down over under again once".split())

def _content(t):
    return set(w[:4] for w in cp.toks(t) if len(w) >= 3 and w.lower() not in _STOP)

# ---------- taxonomy names -> prose-register prompts ----------
# User insight: Qwen3-0.6B free prose is bland MOST of the time by nature
# ("no human would want to read it -- that's what bland is"). The taxonomy
# (baiango/genre-taxonomy-sfw, ~197k genre+subgenre leaves) supplies diverse
# topics; "Write a {name} story.\n\n" keeps the 0.6B in prose register
# (probed 2026-10-02: format A -> quiz meta, C -> headers, B -> prose). Bare
# openers leak the QA/math register -- never generate without a taxonomy name.
PARQUET = os.environ.get("TAXO_PARQUET", "/Users/arin/.cache/huggingface/hub/datasets--baiango--genre-taxonomy-sfw/snapshots/8ab030d56433de108d802af0e8c0578925b5c2e9/data/train-00000-of-00001.parquet")

def load_names():
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    names = set(df.genre.astype(str))
    for subs in df.subgenres: names.update(map(str, subs))
    names = sorted(n.strip() for n in names if 3 <= len(n.strip()) <= 60)
    return names

def load_leaves():
    """Sorted unique [(leaf, parent_genre)] — parent = the row whose subgenres
    contain the leaf, or itself when the leaf IS a top-level genre."""
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    parent = {}
    for g, subs in zip(df.genre.astype(str), df.subgenres):
        parent.setdefault(str(g).strip(), str(g).strip())
        for s in subs:
            parent.setdefault(str(s).strip(), str(g).strip())
    return sorted((n, p) for n, p in parent.items() if 3 <= len(n) <= 60)

# 10-03 recalibrated (user: "loosen the check so they pass"): a sentence
# pair is cohesive iff it shares >=1 content word (was >=15% of the smaller
# set); rows fail only below 0.10 (was 0.20). Anchors: GOLD bland_free
# 129/129 pass, ADVERSARY babble_ref 7/60 rejected, and 42/43 live
# incoherent new-fails recovered (Arcane Encounter = 2-sentence strip
# remainder, true degenerate, stays failed via the <3-sentence clause).
def _coh_frac(t):
    ss = [x for x in cp.split_re.split(t) if len(cp.toks(x)) >= 5]
    if len(ss) < 3:
        return 0.0
    pairs = list(zip(ss, ss[1:]))
    ok = 0
    for a, b in pairs:
        A, B = _content(a), _content(b)
        if A and B and (A & B):
            ok += 1
    return ok / len(pairs)

# Top English function words — ASCII-only foreign languages (Indonesian, Malay,
# ...) share almost none of these, real English prose is 40%+ from this list.
_EN_FUNC = set(("the of and to a in that it is was he she his her they i we you not "
    "with as for on at but had have be this from or an which were by so one all "
    "no there their if out up about who me him them my into over then than when "
    "what how said says would could will can did do been its our your has are am "
    "her hers himself herself itself myself don't can't didn't it's he's she's "
    "i'm they're we're you're that's there's what's very just now here where why "
    "again more most some such only own same too also because while during before "
    "after above below between through both each few other any nor until off down "
    "does doesn't didn't won't wouldn't couldn't shouldn't wasn't weren't").split())

_TYPO = set("’‘“”—–…")          # typographic chars: NOT language evidence (friend’s)

def _eng(t):
    """(is_english, soft_flag_or_None). Diacritic-heavy (Turkish/French/German...)
    and ASCII-script foreign prose rejected; em-dash/curly quotes are free."""
    na = sum(1 for c in t if ord(c) > 127 and c not in _TYPO)
    if na > max(8, 0.01 * len(t)): return False, None
    if na: return True, "diacritics"          # some é/ü/ğ — allow, flag for review
    toks = re.findall(r"[a-z']+", t.lower())
    if not toks: return False, None
    fw = sum(1 for w in toks if w in _EN_FUNC) / len(toks)
    if fw < 0.15: return False, None
    return True, ("english-low" if fw < 0.22 else None)

def english(t):
    return _eng(t)[0]

# --- template/meta-speech gates (10-06 census: 24 meta + 6 outline + 25 chapter
# heads in a 1057-row corpus, near-all verdict=pass). The loop gate counts
# REPEATED n-grams; description-speak repeats nothing, so it is structurally
# blind to this class (user catch). All anchored: mid-prose mentions never fire.
# Patterns run on WHITESPACE-NORMALIZED targets (attempt() gates norm/prose),
# so ^ = string start and there are no mid-string newlines.
# 10-03 widened echo clause (A|The + richer tail verbs), spec-preamble
# alternatives (Finding 18) and model-identity alts (fix5 run6 section 1).
ECHO_ALT = '(?:A|The)\\b([^.!?\\n]{0,60}?)\\b(?:story|tale)\\b[\\s,]+(?:about|of|in|which|that|where|who|is|was|were|are|named|called|set|tells?|told|unfolds?|begins?|continues?|ends?|describes|features|follows|revolves|centers?|explores|delves?|captures|reveals)\\b'
_META_BODY = (                    # shared by the start-gate and the sentence cutter
    "The (?:story|tale) (?:is|should|will|must|would|takes|began|begins|start(?:s|ed)?|opens|follows|revolves|centers?|explores|delves|unfolds|describes|tells|features)|This story\\b|This (?:is|was) the (?:beginning|start|end) of the (?:story|tale)\\b|This (?:legend|tale) is not (?:just|merely|only)\\b|This (?:problem|challenge) (?:would|will|could)\\b|I (?:have|had|need|will|must) to write (?:a|an|the|this|one|you)\\b|Here(?:'s| is) (?:a|an|the) story|In this story\\b|"
    + ECHO_ALT +
    "|It should (?:also )?(?:have|be|contain|include|incorporate|involve|focus on|follow|explore|use|employ|take)\\b(?!\\s+been)[^.!?]*\\b(?:story|characters?|plots?|settings?|action|scenes?|words?|pages?|length|tenses?|persons?|narrators?|formats?|focus|chapters?|dialogue|beginning)\\b|It should have a beginning\\b|You (?:should|must|will|need to) (?:write|have|make|create|include|be)\\b[^.!?]*\\b(?:story|characters?|words?|pages?|plots?|settings?|length|scenes?|formats?)\\b|Each (?:character|paragraph|page|section|chapter) (?:should|must|will|has|have)\\b|The (?:first|second|third|final|last) (?:page|paragraph|chapter|part|section) (?:should|must|will)\\b|I (?:am|'m) not (?:a human|an AI|a language model)\\b|I can be programmed to\\b|not limited to any specific\\b|Use (?:descriptive|vivid|simple|clear|sensory|rich|evocative|plain|engaging)\\b|It (?:is|was) (?:a|an|the) (?:story|tale)\\b|The (?:beginning|middle|end) of the story is\\b|The (?:title|name) of (?:this|the) story is\\b|Let's create (?:a|an|the) story\\b|Here(?:'s| is) (?:a|an|the) (?:detailed|short|engaging|complete|version|example|summary)\\b|The author (?:wants|wishes|wanted) to\\b|The protagonist of (?:the|this) (?:story|tale)\\b|It (?:will|would|shall) (?:be|become) (?:a|an|the)\\b[^.!?]{0,60}?\\b(?:story|tale|narrative|plot|scene|chapter|book)\\b|The (?:main )?(?:plot|story|tale|narrative) (?:will|would|should|must)\\b")
_META_OPEN = re.compile(r"^\s*(?:" + _META_BODY + r")", re.I)
_META_SENT = re.compile(r"\s*(?:" + _META_BODY + r")[^.!?]*[.!?]\s*", re.I | re.S)
_META_FRAG = re.compile(r"\s*(?:(?:" + _META_BODY + r")[^.!?]*[.!?]|(?:" + _META_BODY
                        + r")[^.!?]*$)\s*", re.I | re.S)   # pass-4 only: tail alt
                        # covers UNTERMINATED meta paras (ARCHICORN 10-06: final
                        # para 'The story follows … code' with no final period)
# 10-07 ABNORGANIC: the [^.!?]*[.!?] scan stops at TITLE-ABBREV periods
# ("…would lead Dr."), so a full frag/sentence cut amputates mid-name
# ("Vasquez into the unknown depths…"). Trap ⇒ excise ONLY the opener phrase
# and keep+capitalize the in-world remainder (user directive 10-07).
_META_LEAD = re.compile(r"\s*(?:" + _META_BODY
                        + r")(?:\s+(?:with|in|about|of|around|through))?",
                        re.I | re.S)   # eats the frame connective too — leaving
                        # it caps "With a series…" (10-07 boundary bug)
_ABBR_DOT = re.compile(r"(?:Dr|Mr|Mrs|Ms|St|Sr|Jr|Prof|vs|etc|e\.g|i\.e)\.\s*$", re.I)

# 10-07 ECHOLEAD guard: the new echo alt is shape-hungry, so cutter/gate ask
# _echo_fp before trusting a hit. An echo TAGLINE is a verb-less NP ("A light
# and fluffy story in English."); a NARRATIVE subject carries a finite verb
# outside its relative clauses ("A tale of woe swept through the village.").
# Relative-clause verbs don't disqualify an echo ("A story of a hero who
# saves a country" is still an NP), and an echo-shaped hit longer than 12
# words is a fused run-on (echo glued onto story text with no boundary —
# same doctrine as the _ABBR_DOT trap: excising the "sentence" would amputate
# the story, so keep it intact). Over-keeping = status quo; over-cutting =
# damage, so the guard errs toward keep.
_ECHO_NP = re.compile('^\\s*' + ECHO_ALT, re.I)
_ECHO_VB = re.compile('\\b(?:is|are|was|were|be|been|being|am|has|have|had|will|would|could|can|must|shall|should|do|does|did|begins?|began|starts?|started|ends?|ended|comes?|came|goes?|went|unfolds?|unfolded|tells?|told|sweeps?|swept|lives?|lived|dies|died|returns?|returned|takes?|took|spans?|spanned|carries?|carried|changed|shapes?|shaped|haunts?|haunted|awaits?|awaited|remains?|remained|reveals?|revealed|captures?|captured|echoes?|echoed|continues?|might|may|needs?|sees?|saw|meets?|met|becomes?|became)\\b', re.I)   # 10-03 widened tail verbs

# 10-03 split (fix5 run6 section 2): STRIP path (_echo_core, via _mhit) has
# NO length guard -- a long period-terminated echo lead is still CUT; GATE
# path (echo_keep) keeps >12-word hits = fused run-ons (ANCIENT CHAMBER).
_ECHO_PROT = re.compile('\\b(?:told|recounted|sung|whispered|spoken|written|shared|passed|handed|echoed|heard)\\b\\s*(?:by|in|through|among|to|with|down|out|around|across|from|between)\\b', re.I)
_BLANK_COP = re.compile('\\b(?:is|was|were|are|be|been|being)\\s+(?:about|of|set|inspired|based|adapted|written|created|crafted|centered|centred|focused|a blend|a mix|a combination|a story|a tale|a narrative|named|called|a classic|a well-known|a familiar|a popular)\\b', re.I)
_REL = re.compile('\\b(?:who|that|which)\\b[^.!?]*', re.I)
_SPEC_ART = re.compile('\\s*(?:story|tale)\\b\\s+(?:will|would|is|was|were|are|has|had|can|could|shall|should|must|may|might)\\b', re.I)

def _echo_core(s):
    """STRIP path (newline-preserved text): echo-shape test WITHOUT the length
    guard — a long hit here is a genuine long echo sentence and must be CUT
    (the ECHOLEAD class)."""
    m = _ECHO_NP.match(s)
    if not m:
        return False
    # Spec preamble, not narration (Finding 18, 10-03): "The story will be a
    # short, fast-paced tale of..." — the widened ECHO_ALT (A|The) now matches
    # it, and the "will" in group(1) would otherwise mark it narratival.
    if _SPEC_ART.match(m.group(1)):
        return False
    if _ECHO_VB.search(m.group(1)):
        return True
    body = _BLANK_COP.sub(" ", _REL.sub(" ", s))
    return bool(_ECHO_PROT.search(body))

def echo_keep(s):
    """GATE path (flattened text, gates_all): >12 words = fused run-on — echo
    glued onto story text with no boundary, cutting the "sentence" amputates
    the story (ANCIENT CHAMBER 10-02: two unpunctuated title lines + story glue
    into one flat 'A … tale of …' span). SCOPE: gate only — the strip path uses
    _echo_core, so long period-terminated echo leads are still cut."""
    if not _ECHO_NP.match(s):
        return False
    if len(s.split()) > 12:
        return True
    return _echo_core(s)

_echo_fp = echo_keep            # gates_all meta check (gate path)

def _mhit(s, frag=False):
    m = (_META_FRAG if frag else _META_SENT).match(s)
    if m is None:
        return None
    hit = m.group(0)
    if _ECHO_NP.match(hit):
        if frag:
            return None if _echo_core(hit) else m   # keep only narratival echo
        if _echo_core(hit):
            return None          # narratival → keep
    return m

# 10-07 ECHOLEAD ledger: _style_strip_n resets it; cut_meta_leak accumulates
# every echo-shaped sentence/line cut; attempt()/restrip surface it as the
# review flag "echo-lead" so no cut is ever silent.
echo_cuts = 0

def _excise_lead(f):
    """Drop just the meta opener from a frag/sentence; None if nothing remains."""
    m = _META_LEAD.match(f)
    if not m: return None
    rest = f[m.end():].lstrip()
    return rest[:1].upper() + rest[1:] if rest else None
_PARA_HEAD = re.compile(r"\s*(.*?)(?:\n\s*\n|\Z)", re.S)   # first paragraph

# 10-03 lead lane (fix5 run6 section 3): AI/refusal/instruction speech that
# survived the pattern cuts (narrow: unambiguous model/author voice).
_LEAD_LEAK = re.compile("\\s*(?:I (?:am|'m) not (?:an? )?(?:expert|able|capable|allowed|programmed)\\b|I (?:am|'m) (?:an? )?(?:AI|language model)\\b|As an AI\\b|I (?:can|will|'ll) (?:help you )?(?:generate|write|create|craft)\\b|Please (?:generate|write|create|craft) (?:a|an|the|me|it)\\b|The first (?:question|chapter|part|section|one)\\b|I thought for a second\\b|I decided to write\\b|It (?:is|was) told through\\b|The (?:name|title) of (?:the|this) story is\\b)(?:[^.!?]*[.!?])\\s*", re.I)

# 10-03 fused salvage (section 4): echo head + meta tail glued into one
# sentence -- find the first real sentence start inside the span, keep tail.
_SENT_START = re.compile("\\b(?:In|On|At|Once|One|There|When|As|After|Before|Long|Far|Deep|Now|Soon|It|He|She|They|The|A|An|But|And|Outside|Inside|Beyond|Within|High|Down|Up)\\s+[a-z][a-z'\\u2019-]*(?:\\s+[a-z])")
_PREV_BAD = re.compile('(?:the|a|an|of|and|or|to|in|on|with|by|for|from|its|his|her|their|our|your)\\s+$', re.I)

def _salvage(t, start, end):
    for mm in _SENT_START.finditer(t, start, end):
        if _PREV_BAD.search(t[:mm.start()]):
            continue
        return mm.start()
    return None

SALVAGE, LEAD_CUTS = [], []      # audit ledgers (fix5_dry.py reported both)

def cut_meta_leak(t):
    global echo_cuts            # module ledger (harness wrote mg.echo_cuts)
    n = 0
    while True:
        m = _PARA_HEAD.match(t)
        if not m:
            break
        para = m.group(1)
        sents = re.findall(r"[^.!?]*[.!?]", para)
        meta_ct = sum(1 for s in sents if _mhit(s))
        if not sents or meta_ct * 2 <= len(sents):
            break
        echo_cuts += sum(1 for s in sents if _ECHO_NP.match(s))
        t = t[m.end():]
        n += len(sents)
    for i in range(0, len(parts := re.split(r"(\n\s*\n)", t)), 2):
        sents = re.findall(r"[^.!?]*[.!?]", parts[i])
        meta_ct = sum(1 for s in sents if _mhit(s))
        if sents and meta_ct * 2 > len(sents):
            parts[i] = ""
            echo_cuts += sum(1 for s in sents if _ECHO_NP.match(s))
            n += len(sents)
            continue
        frags = re.findall(r"[^.!?]*[.!?]|[^.!?]+$", parts[i], re.S)
        keep, exc = [], 0
        for f in frags:
            mf = _mhit(f, frag=True)
            if mf is None:
                keep.append(f)
            elif _ABBR_DOT.search(mf.group(0)):
                e = _excise_lead(f)
                if e:
                    keep.append(e); exc += 1
        if exc or len(keep) != len(frags):
            parts[i] = "".join(keep)
            n += len(frags) - len(keep) + exc
    t = "".join(parts)
    while True:
        m = _mhit(t)
        if m is None:
            m = _LEAD_LEAK.match(t)
            if m is None:
                break
            LEAD_CUTS.append((m.group(0).strip()[:70], t[:90]))
            t = t[m.end():]
            n += 1
            continue
        hit = m.group(0)
        if _ECHO_NP.match(t):
            echo_cuts += 1
            em = _ECHO_NP.match(t)
            cut = _salvage(t, em.end(), m.end())
            if cut is not None:
                SALVAGE.append((t[:90], t[cut:cut + 60]))
                t = t[cut:]
                n += 1
                continue
        if _ABBR_DOT.search(hit):
            t = _excise_lead(t) or ""
        else:
            t = t[m.end():]
        n += 1
    return t, n
_TEMPLATE_LBL = re.compile(       # outline/char-sheet labels anywhere in prose
    r"\b(?:Setting|Characters?|Date|Genre|Plot Summary|Plot|Themes?|Objective|Synopsis)"
    r"\s*:\s*\S")                 # \b: 'Update:' must not match as 'date:'
_TEMPLATE_HEAD = re.compile(      # survived stripping = heading at text start;
    r"^\s*(?i:(?:Chapter|Part|Scene))\s+(?:\d+|(?i:one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve))\s*(?::|[-\u2013\u2014]|[A-Z][a-z])"
    + '|^\\s*\\*{0,2}(?i:(?:Chapter|Part|Scene|Episode|Act))\\s*:\\s*\\S')  # title-ish
                                  # continuation required so 'Chapter 1 covered
                                  # the war.' (legit opener) passes, 'Chapter 1:
                                  # The Whispering Woods' / 'Chapter 1 The…' fail
_OUTLINE = re.compile(            # Roman-numeral section heads; >=2 = outline
    r"(?:^|\s)(?:VI{0,3}|IV|V|I{1,3})\.\s")
_OUTLINE_INTRO = re.compile(      # single-head explicit intro sheet
    r"^\s*(?:I{1,3}|IV|VI{0,3})\.\s*(?:Introduction|INTRODUCTION|Background|Overview)")
_SHEET = re.compile(              # lettered instruction bullets: 'A. Set up the
    r"(?:^|\s)[A-E]\.\s+(?:Set up|Give an?|Describe|Explain|Introduce|Create a|"
    r"Write a|Establish)")        # story… B. Give an overview…' prompt sheets
_SHEET2 = re.compile(             # char-sheet heads: 'A. **Title:**  B. **Plot:**'
    r"A\.\s*\*{0,2}(?:Title|Plot|Character|Setting)", re.I)
_SHEET_BUL = re.compile(r"(?:^|\n)\s*[A-Z]\.\s")   # ≥2 lettered bullets = spec sheet
_NUM_BUL = re.compile(r"(?:^|\n)\s*\d{1,2}\.\s")  # ≥2 numbered bullets = format spec

_DEFINE = re.compile(  # 10-07 census: essay openers defining story as a form
    r"\b(?:is|should be) a unique type of\b|\b(?:is|should be) a narrative in which\b"
    r"|\b(?:is|should be) a type of (?:storytelling|story|fiction|writing)\b", re.I)

_SPEAKER = re.compile('(?m)^\\*{0,2}[A-Z][A-Za-z\' ]{0,25}:\\*{0,2}\\s*(?:\\(|\\"|$)', 8)   # 10-03 screenplay cue line

def gates_all(t, raw=None):
    """(hard_gates, soft_review_flags) — rows can fail several at once. Hard =
    reject; soft = ship but flag for the human/agent review pass. `raw` = the
    newline-preserved source: callers normalize whitespace before gating, which
    makes line-anchored patterns (^/\n bullets) blind past position 0 — the
    bullet counters use `raw` (falls back to t) to see real line structure."""
    n_tok = len(cp.toks(t))
    why, soft = [], []
    if not 40 <= n_tok <= MG_LEN_MAX: why.append("len")   # 10-02: widened for box llama.cpp build (EOSes later, 160-240 band); content gates unchanged
    elif n_tok < 55 or n_tok > MG_LEN_MAX - 15: soft.append("len-edge")
    if cp.has_loop(t): why.append("loop")
    words = re.findall(r"[a-z0-9']+", t.lower())   # normalized: punctuation drift + fused-jam tokens ("manyIn") must not hide repeats
    if len(words) >= 8:                            # loop-ish: gate needs 8-gram; 6-gram density is the soft signal
        g6 = [" ".join(words[i:i+6]) for i in range(len(words) - 5)]
        if len(set(g6)) / len(g6) < 0.85: soft.append("loop-ish")
        g8c = {}                                   # echo: opening restart / duplicated passage
        for g in [" ".join(words[i:i+8]) for i in range(len(words) - 7)]:
            g8c[g] = g8c.get(g, 0) + 1
        if max(g8c.values()) >= 2: soft.append("echo")
    if cp.FORMAT_LEAK.search(t): why.append("format")
    if _META_OPEN.match(t):                        # _echo_fp: a narratival
        ms = _META_SENT.match(t) or _META_OPEN.match(t)   #  'A tale of woe
        if not _echo_fp(ms.group(0)): why.append("meta")   #   swept…' saves
    if _DEFINE.search(t): why.append("define")     #   opener never fires it (10-07)
    if _TEMPLATE_LBL.search(t) or _TEMPLATE_HEAD.match(t): why.append("template")
    if (len(_OUTLINE.findall(t)) >= 2 or _OUTLINE_INTRO.match(t) or _SHEET.search(t)
            or _SHEET2.search(t)):
        why.append("outline")
    src = raw if raw is not None else t
    if (len(_SHEET_BUL.findall(src)) >= 2 or len(_NUM_BUL.findall(src)) >= 2) \
            and "outline" not in why:
        why.append("outline")
    # (digits soft flag REMOVED 10-02 — user: years/ages in stories are fine; was
    # review-only anyway, 851 rows flagged, 0 hard fails)
    ok, s = _eng(t)
    if not ok: why.append("english")
    elif s: soft.append(s)
    if _LEAD_LEAK.match(t) and "meta" not in why:   # 10-03 lead lane: AI/
        why.append("meta")                           #   refusal/instr speech
    frac = _coh_frac(t)
    if frac < 0.10: why.append("incoherent")         # 10-03 loosened (>=1 shared word)
    elif frac < 0.15: soft.append("coherence-low")   # bottom band of loosened metric
    if len(_SPEAKER.findall(src)) >= 2:              # 10-03 screenplay (user:
        why.append("screenplay")                     #   >=2 cue lines in VERBATIM raw)
    return why, soft

def _gates(t):
    w = gates_all(t)
    return w[0] if w else None

def free():
    if os.path.exists(FK_OUT):
        print("exists:", FK_OUT, f"({sum(1 for _ in open(FK_OUT))})"); return
    names = load_names()
    rng = random.Random(5); rng.shuffle(names)
    names = names[:1500]                       # ~22 min at ~1 gen/s
    print(f"taxonomy names: {len(names)} sampled")
    n = kept = 0
    rej = {}
    with open(FK_OUT, "w") as f:
        for i, nm in enumerate(names):
            prompt = f"Write a {nm} story.\n\n"
            t = sl._gen(sl.URL_G, prompt, seed=31000 + i, mx=240)
            t = " ".join(t.split())
            if len(cp.toks(t)) < 40:           # one retry on short (old-gen trick)
                t = " ".join(sl._gen(sl.URL_G, prompt, seed=31000 + i + 555555, mx=240).split())
            n += 1
            why = _gates(t)
            if why: rej[why] = rej.get(why, 0) + 1; continue
            kept += 1
            f.write(json.dumps({"k": sl.h(t), "kind": "bland_free",
                                "genre": nm, "text": t}) + "\n")
            if (i + 1) % 100 == 0:
                f.flush(); print(f"  {i+1}/{len(names)} kept={kept} rej={rej}", flush=True)
    print(f"bland_free kept {kept}/{n} -> {FK_OUT}")

def peek():
    rows = [json.loads(l) for l in open(FK_OUT)]
    rng = random.Random(11); rng.shuffle(rows)
    for r in rows[:5]:
        print(f"--- {r['k']}\n{r['text']}\n")

# ---------- rewrite twins: matched eval instrument ----------
PROMPT = ("Rewrite the paragraph below. Replace every specific detail (names, places, "
          "numbers, unusual objects) with vague general wording. Keep the same number "
          "of sentences, the same grammar, and the same flow. Output only the rewritten "
          "paragraph.\n\nParagraph: {t}\n\nRewritten paragraph:")

def pair_src():
    rng = random.Random(7)                     # same replay, deeper slice
    fk, fl = [], []
    for f in (D + "source/genre500_uscrub_train.jsonl", D + "source/genre500_uscrub_val.jsonl"):
        for line in open(f): fk.append(cp.first_words(json.loads(line)["text"], cp.NWIN))
    for line in open(D + "train_3turn.jsonl"):
        fl.append(cp.first_words("\n\n".join(json.loads(line)["pieces"]), cp.NWIN))
    rng.shuffle(fk); rng.shuffle(fl)
    return fl[800:950]                         # disjoint from build [0:300] + extras [300:800]

def ok_rewrite(src, t):
    if not t or t == src: return False
    if re.search(r"^#|\*\*|^Paragraph|^Rewritten", t, re.M): return False
    if "______" in t or re.search(r"\d", t): return False
    r = len(cp.toks(t)) / max(1, len(cp.toks(src)))
    if not 0.6 <= r <= 1.5: return False
    ns, nt = len(cp.split_re.split(src)), len(cp.split_re.split(t))
    if abs(ns - nt) > 1: return False
    a = set(w.lower() for w in cp.toks(src)); b = set(w.lower() for w in cp.toks(t))
    ov = len(a & b) / max(1, len(a | b))
    return 0.15 <= ov <= 0.9                   # related rewrite, not a copy

def pairs():
    if os.path.exists(PKP):
        print("exists:", PKP, f"({sum(1 for _ in open(PKP))})"); return
    srcs = pair_src()
    kept = 0
    with open(PKP, "w") as f:
        for i, t in enumerate(srcs):
            got = None
            for j in range(3):
                r = sl._gen(sl.URL_G, PROMPT.format(t=t), seed=5000 + 7 * i + j, mx=400)
                if ok_rewrite(t, r): got = r; break
            if got:
                kept += 1
                f.write(json.dumps({"kf": sl.h(t), "kg": sl.h(got),
                                    "fresh": t, "generic": got}) + "\n")
            if (i + 1) % 50 == 0:
                f.flush(); print(f"  {i+1}/{len(srcs)} kept={kept}", flush=True)
    print(f"pairs kept {kept}/{len(srcs)} -> {PKP}")

# ---------- teacher-first test (judge :8936, run after scale_labels) ----------
def report():
    have = {json.loads(l)["k"]: json.loads(l)["s"] for l in open(sl.CK)}
    src_of = {}
    for l in open(sl.CK):
        r = json.loads(l); src_of[r["k"]] = r["src"]
    fresh = [have[k] for k, v in src_of.items() if v == "fresh_laguna_x"]
    # free class: judge + AUC vs fresh
    if os.path.exists(FK_OUT):
        free_rows = [json.loads(l) for l in open(FK_OUT)]
        need = [(r["k"], r["text"]) for r in free_rows if r["k"] not in have]
        print(f"free: {len(free_rows)} rows, to-judge={len(need)}")
        with open(sl.CK, "a") as f:
            for n, (k, t) in enumerate(need):
                s = sl._judge_para(t)
                if s is not None:
                    have[k] = s
                    f.write(json.dumps({"k": k, "s": round(s, 4), "src": "bland_free"}) + "\n")
                if (n + 1) % 100 == 0: f.flush(); print(f"  {n+1}/{len(need)}", flush=True)
        fs = [have[r["k"]] for r in free_rows if r["k"] in have]
        if fresh and fs:
            wins = sum(1 for x in fs for y in fresh if x > y) + 0.5 * sum(1 for x in fs for y in fresh if x == y)
            auc = wins / (len(fs) * len(fresh))
            print(f"\nFREE TEACHER TEST  n_free={len(fs)} n_fresh={len(fresh)}  "
                  f"AUC(bland_free vs fresh_laguna_x)={auc:.3f}  "
                  f"median_free={sorted(fs)[len(fs)//2]:.3f} median_fresh={sorted(fresh)[len(fresh)//2]:.3f}")
            print("GO" if auc > 0.85 else "WEAK: judge barely separates fluent-bland from fresh")
    # pairs: matched ranking
    if os.path.exists(PKP):
        pairs_ = [json.loads(l) for l in open(PKP)]
        need = []
        for p in pairs_:
            if p["kf"] not in have: need.append((p["kf"], p["fresh"]))
            if p["kg"] not in have: need.append((p["kg"], p["generic"]))
        print(f"pairs: {len(pairs_)}, to-judge={len(need)}")
        with open(sl.CK, "a") as f:
            for n, (k, t) in enumerate(need):
                s = sl._judge_para(t)
                if s is not None:
                    have[k] = s
                    f.write(json.dumps({"k": k, "s": round(s, 4), "src": "gen_pair"}) + "\n")
                if (n + 1) % 100 == 0: f.flush(); print(f"  {n+1}/{len(need)}", flush=True)
        m, win = [], 0
        for p in pairs_:
            sf, sg = have.get(p["kf"]), have.get(p["kg"])
            if sf is None or sg is None: continue
            d = sg - sf; m.append(d); win += d > 0
        if m:
            m.sort()
            print(f"\nPAIRS TEACHER TEST n={len(m)} ranked-correct={win/len(m):.1%} "
                  f"median-margin={m[len(m)//2]:+.3f}")
            print("GO" if win / len(m) > 0.85 else
                  "STOP: teacher ~blind on matched pairs -- rethink before distilling")

if __name__ == "__main__":
    {"free": free, "peek": peek, "pairs": pairs, "report": report}.get(
        sys.argv[1] if len(sys.argv) > 1 else "free", free)()
