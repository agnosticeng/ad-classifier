# Benchmarks

Wall-clock performance of the classifier, with emphasis on **CPU** (the
fallback when no Metal GPU is available). Times are end-to-end wall clock and
include process start, model load, tokenization, and all forward passes.

## Environment

| Item | Value |
|---|---|
| Machine | Apple M5 Max, 18 cores, 128 GB unified memory |
| OS | macOS 27.0 (build 26A428) |
| Rust | rustc 1.97.1 (release build) |
| Engine | `lisa-engine` from [agnosticeng/lisa](https://github.com/agnosticeng/lisa) |
| Checkpoint | `convaiinnovations/laya` (English root, 804 MB safetensors) |
| Pipeline | wide `choice` (stage 1) → top-N `choice` (stage 2; N = 8 content, 3 ads), `head_max_len=512`, `max_len=8192` |

Model weights are mmapped, so load is negligible; the numbers are dominated by
the encoder forward passes. The CPU path runs on all cores (user CPU time is
~12-15× wall time), the Metal path is single-threaded host code around the GPU.

## Single question (`lisa decide`)

One typed `choice` question, one record, by number of options. This isolates the
per-question forward cost.

| options | device | wall | user CPU | CPU-time / wall |
|---:|---|---:|---:|---:|
| 2 | **cpu** | **1.07s** | 12.38s | 1214% |
| 8 | **cpu** | **1.28s** | 15.92s | 1287% |
| 37 | **cpu** | **2.51s** | 36.51s | 1483% |
| 2 | metal | 0.21s | 0.13s | 87% |
| 8 | metal | 0.21s | 0.13s | 89% |
| 37 | metal | 0.22s | 0.14s | 89% |

CPU time grows with the option count (more marker tokens); Metal is essentially
flat at ~0.2s because the GPU absorbs the extra width.

## Classifier end-to-end

Two questions per record (one wide, one shortlist). The classifier prints a
per-question and per-record timing line, so every run reports its own breakdown.
The shortlist size is per taxonomy: content uses 8, ads use 3.

### Per-question and per-record (CPU)

| set | question | options | CPU / question | CPU / record |
|---|---|---:|---:|---:|
| content | q1 wide | 37 | 3.16-3.22s | 4.33-4.44s |
| content | q2 shortlist | 8 | 1.14-1.25s | |
| ads | q1 wide | 46 | 4.35-4.51s | 5.11-5.39s |
| ads | q2 shortlist | 3 | 0.76-0.89s | |

### Per-question (Metal)

| set | question | options | Metal / question | Metal / record |
|---|---|---:|---:|---:|
| content | q1 wide | 37 | 0.12-0.20s | 0.19-0.28s |
| content | q2 shortlist | 8 | 0.075-0.077s | |
| ads | q1 wide | 46 | 0.15-0.22s | 0.22-0.30s |
| ads | q2 shortlist | 3 | 0.065-0.073s | |

### End-to-end totals

| set | device | all records | per record | labels |
|---|---|---:|---:|---|
| content (5) | **cpu** | **21.90s** | **4.38s** | 5/5 |
| content (5) | metal | 1.05s | 0.21s | 5/5 |
| ads (5) | **cpu** | **26.25s** | **5.25s** | 5/5 |
| ads (5) | metal | 1.17s | 0.23s | 5/5 |

Model load is ~0.03s (mmapped weights), so essentially all time is the two
forward passes. The wide question costs more with more options (46 > 37); the
shortlist question is ~0.8-1.2s on CPU for 3-8 options, i.e. there is a large
fixed per-question cost (state encoding + head) plus a per-option cost.

### Single record (content, article-1)

| device | stages | wall |
|---|---|---:|
| **cpu** | 2 | **4.32s** |
| **cpu** | 1 | **3.23s** |
| metal | 2 | 0.28s |

## Summary

| metric | CPU | Metal |
|---|---:|---:|
| one question, 37 options | 2.51s | 0.22s |
| classifier, content (5) | 21.9s | 1.05s |
| classifier, ads (5) | 26.3s | 1.17s |
| per content record | 4.38s | 0.21s |
| per ad record | 5.25s | 0.23s |

- **CPU is ~5-20× slower** than Metal, widening with option count.
- **Accuracy is identical** on both devices (same labels; the CPU forward is a
  portable f32 host path).
- Model load is not a factor (mmapped); cost is the forward passes.
- Earlier, a stage 1 that asked **one binary question per candidate** (38
  questions/record) took **251s** on CPU for the same articles. Replacing it
  with a single wide `choice` + shortlist (2 questions/record) cut that to
  **22s** — the change that makes CPU usable.

## Reproduce

```bash
# end-to-end classifier (prints its own timing line)
python scripts/classify.py --taxonomy content-taxonomy-3.1 --all --device cpu
python scripts/classify.py --taxonomy ad-product-taxonomy-2.0 --all --device cpu
python scripts/classify.py --taxonomy content-taxonomy-3.1 --id article-1 --device cpu

# one question, CPU vs Metal (q.json = a {"state":…,"questions":…} payload)
lisa decide --model convaiinnovations/laya --device cpu --input q.json
lisa decide --model convaiinnovations/laya --input q.json
```

The Rust classifier prints a per-question and per-record breakdown to stderr:

```
[model] loaded in 0.030s
  [article-1] q1(wide, 37 opts) 3.180s | q2(short, 8 opts) 1.136s
[article-1] total 4.316s
...
[5 record(s)] classify 21.902s
```

and `scripts/classify.py` adds the wall line, e.g.
`[classify] 23.08s wall over 5 record(s)`.
