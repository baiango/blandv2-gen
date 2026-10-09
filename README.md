# blandv2-gen

Producer for the **blandv2** corpus — 197,016 bland-prose SFT rows (196,997
usable after the standard filter) drawn from llama.cpp `llama-server`
`/v1/completions`, style-stripped, gate-labeled, shipped as JSONL. This repo
commits the full producer: drivers, gates, strip machinery, taxonomy leaves,
dead-tree patch driver, the genre-labeling chain (`labeling/`), and the audit.
**The corpus data itself is not
committed.**

## Model pin

`Qwen3-0.6B-Base.Q5_K_S` (mradermacher GGUF). `box_setup_v2.sh` downloads and
sha256-verifies it, then starts `llama-server`. All samplers are pinned in the
request body — chain `["top_k","top_p","min_p","temperature"]`, top_k 40,
top_p 0.95, temperature 1.0, min_p 0.1, no repetition/ngram penalties — so
server launch flags are defaults only.

## How a row ships

`leaves.jsonl` (197,006 leaf lines — one `n` subgenre title each, 119,534
unique `p` parent genres; **line order = idx = shard map**) → per leaf, draws
at seed `911000000 + idx*10 + att` (att 0..59 retry budget, one draw per seed)
→ strip + glue (`build_dataset._style_strip_n`) → mechanical gates → verdict.
Emitted verdicts: `pass` | `style` | `failed` (gate names in the row's `gates`
field) | `dupe` | `exhausted` — in the shipped run only `pass`/`style`/`failed`
occur (k-dupes skip silently, the patch refilled every exhausted leaf). Row
schema: see `build_dataset_v2.attempt` (row keys) and `main` (adds
`parent`/`draws`) — code is truth. Every discarded draw (gate-fail or dupe) is
appended to `<OUT>.rejects.jsonl` with `att` + gates — nothing is dropped
in-memory.

## Quickstart

```bash
# 1. server — downloads + sha-verifies the model, starts llama-server (CUDA box)
bash box_setup_v2.sh

# 2. generate this worker's shard (multi-host/worker launch: box_launch_v2.sh)
BLAND_URL=http://127.0.0.1:8080 BLAND_OUT=out.jsonl BLAND_SHARD=0/4 \
MG_LEN_MAX=700 python3 build_dataset_v2.py

# 3. patch dead trees (leaves that exhausted their budget) — seeds CONTINUE
#    the stream at att>=60, k-deduped against the existing shards
PATCH_MAX=140 python3 patch_dead_trees.py

# 4. post-process + audit — restrip.py / audit_final_shards.py (see docstrings)
```

## Environment

| var | meaning |
|---|---|
| `BLAND_URL` / `BLAND_URLS` | llama-server endpoint(s); comma list = multi-server fan-out |
| `BLAND_OUT` | output `.jsonl`; discards also append to `<BLAND_OUT>.rejects.jsonl` |
| `BLAND_SHARD` | `i/n` — this worker's leaf range |
| `BLAND_WORKERS` | draw threads |
| `BLAND_ATTEMPTS` | per-leaf retry budget (shipped run: 60) |
| `BLAND_CAP` | max server tokens per draw (default 800) |
| `BLAND_LEAVES` | leaves path (default: `leaves.jsonl` beside the code) |
| `MG_LEN_MAX` | target prose length cap (shipped run: 700) |
| `PATCH_MAX` / `PATCH_WORKERS` | patch driver: extra attempts per dead leaf / threads |
| `TAXO_PARQUET` | only needed to re-derive `leaves.jsonl` from the taxonomy parquet |

## Files

| file | role |
|---|---|
| `build_dataset_v2.py` | v2 driver: draw → strip → gates → emit |
| `build_dataset.py` | v1 strip/glue machinery + gate regexes (imported verbatim) |
| `patch_dead_trees.py` | refill leaves that exhausted; seed stream continues at att≥60 |
| `restrip.py` | idempotent re-strip/re-gate from verbatim text |
| `audit_final_shards.py` | final audit: residue, k-dupes, prose⟺style, band |
| `make_generic.py` / `ceiling_probe.py` / `dataset_battery.py` / `scale_labels.py` | gate + probe deps |
| `leaves.jsonl` | the 197,006-leaf taxonomy (119,534 unique parent genres; committing it pins idx/shard/seed mapping) |
| `box_setup_v2.sh` / `box_launch_v2.sh` | server setup + the 4-box launch used for the shipped run |
| `labeling/` | post-build chain: genre labels (WORLD+ENERGY+TAGS, 3-vote), retok `n_tok`, canonical row order, cfgate gate-key merge |

## Shipped-run stats

4× RTX 3080Ti, ~45.6 h, **$23.49 actual** (vast.ai `show invoices-v1 -c`,
per-instance: 54420385 $13.62 + 54522591 $4.12 + 54522870 $3.83 + 54522460
$1.92) ≈ $0.035 per 1M generated tokens: 197,006 rows from 671M generated
tokens (14% ship yield, 476 kept tokens/row flat). Dead-tree patch: 10/10
refilled on a Mac at $0 in 233 s (att 60–69) → `patch_dead10.jsonl`; merged
shards = 197,016 rows. Final audit: consumed-surface residue 0, k-dupes 0,
prose⟺style exact; 196,997 usable (95,769 `pass` + 101,228 `style`); the 19
`failed` rows (14 loop, 4 incoherent, 1 loop+format) stay in the shards and
are excluded by the standard filter.

## Shipped artifact

The labeled corpus ships as the private HF dataset
[`baiango/bland-stories-labeled`](https://huggingface.co/datasets/baiango/bland-stories-labeled)
(cc0-1.0, single commit). Its card documents the flat schema — `prose` only on
`style` rows, verbatim `text` everywhere — and the label fields
(`world`/`energy`/`tags`/`confidence`) appended by `labeling/`.
