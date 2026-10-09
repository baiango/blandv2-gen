#!/usr/bin/env python3
"""row_order.py — canonical key order for corpus rows (HF dataset viewer readability).

Canonical order (optional keys skipped when absent; unknown keys appended at
end in original order so NOTHING is ever dropped):

    k, genre, parent, prompt, text, prose, n_tok, seed, stop, ends, verdict,
    draws, att, review, gates, world, energy, tags, confidence,
    gate_decision, gate_score, gate_fails, gate_defaulted

Prose sits right after text (verbatim vs stripped side by side); driver meta
grouped; labels last. Import reorder_row() from any writer that emits rows.
"""
import hashlib
import json

CANON = ["k", "genre", "parent", "prompt", "text", "prose", "n_tok", "seed",
         "stop", "ends", "verdict", "draws", "att", "review", "gates",
         "world", "energy", "tags", "confidence",
         "gate_decision", "gate_score", "gate_fails", "gate_defaulted"]


def reorder_row(r):
    """Return dict with keys in canonical order; unknown extras kept at end."""
    out = {}
    for key in CANON:
        if key in r:
            out[key] = r[key]
    for key in r:
        if key not in out:
            out[key] = r[key]
    return out


def canon_md5(r):
    """Content fingerprint independent of key order (for before/after verify)."""
    return hashlib.md5(
        json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
