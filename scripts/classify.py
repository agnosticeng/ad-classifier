#!/usr/bin/env python3
"""Thin wrapper that runs the Rust classifier (`src/main.rs`).

The two-stage strategy lives in Rust (crate `adtech-classifier`); this script
only maps the slug-based CLI and taxonomy paths onto that binary, so there is a
single implementation to maintain. It uses a prebuilt binary when present and
falls back to `cargo run`, which builds it on first use.

Usage:
    python scripts/classify.py --taxonomy content-taxonomy-3.1 --all
    python scripts/classify.py --taxonomy content-taxonomy-3.1 --id article-3
    python scripts/classify.py --taxonomy ad-product-taxonomy-2.0 --state "Luxury SUVs"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# slug -> taxonomy path, inputs path, default shortlist, extra Rust flags.
# shortlist: content needs 8 (article-3 only resolves with a wide shortlist);
# ad products do better with 3 (a wider shortlist lets a distractor win ad-4).
TAXONOMIES = {
    "content-taxonomy-3.1": {
        "taxonomy": "data/taxonomy/content-taxonomy-3.1.json",
        "inputs": "data/inputs/articles.json",
        "shortlist": 8,
        "extra": [],
    },
    "ad-product-taxonomy-2.0": {
        "taxonomy": "data/taxonomy/ad-product-taxonomy-2.0.json",
        "inputs": "data/inputs/ad-products.json",
        "shortlist": 3,
        "extra": ["--state-field", "product", "--subject", "ad creative"],
    },
}


def classifier_command(explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    for rel in ("target/release/classify", "target/debug/classify"):
        candidate = ROOT / rel
        if candidate.exists():
            return [str(candidate)]
    return ["cargo", "run", "--quiet", "--manifest-path", str(ROOT / "Cargo.toml"), "--"]


def record_count(args, spec: dict) -> int:
    if args.state or args.id:
        return 1
    try:
        data = json.loads((ROOT / spec["inputs"]).read_text(encoding="utf-8"))
        return len(data["inputs"])
    except (OSError, KeyError, json.JSONDecodeError):
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taxonomy", choices=sorted(TAXONOMIES), default="content-taxonomy-3.1")
    parser.add_argument("--id", help="classify a single input record")
    parser.add_argument("--all", action="store_true", help="classify every input record")
    parser.add_argument("--state", help="classify an ad-hoc text instead of a file record")
    parser.add_argument("--model", default="convaiinnovations/laya", help="Laya checkpoint (repo id or dir)")
    parser.add_argument("--shortlist", type=int, default=None, help="stage-2 option count (default: per taxonomy)")
    parser.add_argument("--head-max-len", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--depth", type=int, default=1, help="1 = top level, 0 = to leaves")
    parser.add_argument("--read", choices=("auto", "text", "abstract"), default="auto", help="which record field to feed as state")
    parser.add_argument("--stages", type=int, choices=(1, 2), default=2, help="1 = wide choice only, 2 = wide + shortlist refine")
    parser.add_argument("--temperature-by-options", action="append", default=[], metavar="TYPE:SIZE=TEMP")
    parser.add_argument("--device", help="metal | cpu")
    parser.add_argument("--json", action="store_true", help="print full JSON results")
    parser.add_argument("--classifier-bin", help="path to the Rust classify binary (default: build/target)")
    args = parser.parse_args()

    spec = TAXONOMIES[args.taxonomy]
    shortlist = args.shortlist if args.shortlist is not None else spec["shortlist"]
    cmd = classifier_command(args.classifier_bin) + [
        "--taxonomy", spec["taxonomy"],
        "--inputs", spec["inputs"],
        "--model", args.model,
        "--shortlist", str(shortlist),
        "--head-max-len", str(args.head_max_len),
        "--max-len", str(args.max_len),
        "--depth", str(args.depth),
        "--read", args.read,
        "--stages", str(args.stages),
        *spec["extra"],
    ]
    if args.all:
        cmd.append("--all")
    if args.id:
        cmd += ["--id", args.id]
    if args.state:
        cmd += ["--state", args.state]
    if args.device:
        cmd += ["--device", args.device]
    if args.json:
        cmd.append("--json")
    for item in args.temperature_by_options:
        cmd += ["--temperature-by-options", item]

    records = record_count(args, spec)
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    per = elapsed / records if records else elapsed
    print(
        f"[classify] {elapsed:.2f}s wall over {records} record(s) "
        f"({per:.2f}s/record incl. model load): {' '.join(cmd)}",
        file=sys.stderr,
    )
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
