#!/usr/bin/env python3
"""Post-restrip audit over the 10 final shards. CONFIG: SHARDS=10, BAND=40..MG_LEN_MAX(700)."""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as bd  # producer's own compiled regexes — never substrings

D = "blandv2_pull/sharded10"
FILES = [f"{D}/bland_prose_stopped.shard{i}of10" for i in range(10)]
BAND = (40, int(os.environ.get("MG_LEN_MAX", "700")))

n = 0
verdicts = collections.Counter()
stop_c = collections.Counter()
ends_bad = raw_legacy = band_bad = residue = residue_failed = 0
prose_style_mismatch = empty_surface = 0
first_words = collections.Counter()
kcounts = collections.Counter()
for path in FILES:
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            verdicts[r.get("verdict")] += 1
            stop_c[r.get("stop")] += 1
            if not r.get("ends"):
                ends_bad += 1
            if "raw" in r:
                raw_legacy += 1
            p = r.get("prose")
            surface = p if p is not None else r.get("text")
            # producer invariant: prose stored IFF verdict==style (strip was
            # non-identity); pass rows train on text directly.
            if (p is not None) != (r.get("verdict") == "style"):
                prose_style_mismatch += 1
            if not surface or not surface.strip():
                empty_surface += 1
                continue
            if bd._HEAD_LINE.search(surface):
                # only CONSUMED surfaces matter: failed rows are excluded
                # downstream (verdict filter); their verbatim text keeps
                # whatever heading made them fail — by design.
                if r.get("verdict") == "failed":
                    residue_failed += 1
                else:
                    residue += 1
            nt = r.get("n_tok", -1)
            if not (BAND[0] <= nt <= BAND[1]):
                band_bad += 1
            kcounts[r.get("k", "")] += 1
            w = surface.split(" ", 1)[0]
            first_words[w] += 1

kd = sum(c - 1 for c in kcounts.values() if c > 1)
usable = verdicts["pass"] + verdicts["style"]
ship = usable - sum(v for s, v in (("length", stop_c["length"]),) ) - ends_bad
print(f"rows           = {n} (expect 197006)")
print(f"verdicts       = {dict(verdicts)}")
print(f"stop           = {dict(stop_c)}")
print(f"ends False     = {ends_bad}")
print(f"prose<->style mismatches = {prose_style_mismatch} (prose stored IFF verdict==style)")
print(f"empty surface  = {empty_surface}")
print(f"legacy raw key = {raw_legacy}")
print(f"n_tok band violations = {band_bad} (band {BAND})")
print(f"heading residue on CONSUMED surfaces = {residue} (failed-row residue = {residue_failed}, excluded anyway)")
print(f"k-dupes        = {kd}")
print(f"SFT-usable (pass+style) = {usable}; after stop/ends filter ~ {usable - stop_c['length']}")
top = first_words.most_common(5)
print(f"top first words = {top}")
assert n == 197006, "row count mismatch"
assert prose_style_mismatch == 0 and raw_legacy == 0 and residue == 0 and kd == 0
assert empty_surface == 0
print("AUDIT PASS")
