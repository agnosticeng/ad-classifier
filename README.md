# The case for Laya in AdTech classification tasks

A Rust classifier that assigns [IAB taxonomy](https://iabtechlab.com/standards/)
categories to ad creatives and web articles with
[Laya](https://huggingface.co/convaiinnovations/laya). It depends on
`lisa-engine` from [agnosticeng/lisa](https://github.com/agnosticeng/lisa) and
runs the model **in-process** — no external binary, no Python.

## Contents

```
Cargo.toml                        the classifier crate (lisa-engine git dep)
src/main.rs                       the classifier
scripts/classify.py               thin wrapper (slug CLI over the same binary)
BENCHMARK.md                      CPU and Metal timings
data/
  taxonomy/
    ad-product-taxonomy-2.0.json  IAB Ad Product Taxonomy 2.0 as a tree
    content-taxonomy-3.1.json     IAB Content Taxonomy 3.1 as a tree
  inputs/
    ad-products.json              5 ad-product sample inputs from the case
    articles.json                 5 article sample inputs from the case
```

## Build

```bash
cargo build --release
```

Requires an Apple-Silicon Mac with Metal (the `--device cpu` fallback works
without a GPU), and Rust 1.97+.

## Usage

```bash
# content articles (default taxonomy; uses each article's English `abstract`)
cargo run -- --all
cargo run -- --id article-2
cargo run -- --state "A new electric SUV review"

# ad products
cargo run -- --all \
  --taxonomy data/taxonomy/ad-product-taxonomy-2.0.json \
  --inputs data/inputs/ad-products.json \
  --state-field product --subject "ad creative"
```

Output (one line per record; timings go to stderr):

```
[model] loaded in 0.041s
  [article-2] q1(wide, 37 opts) 0.119s | q2(short, 8 opts) 0.077s
[article-2] total 0.196s
article-2: Sports (0.90)
```

| option | default | meaning |
|---|---|---|
| `--taxonomy` | `data/taxonomy/content-taxonomy-3.1.json` | taxonomy JSON (nested `tree`) |
| `--inputs` | `data/inputs/articles.json` | records: `{id, text, abstract?}` |
| `--model` | `convaiinnovations/laya` | Laya checkpoint (repo id or dir) |
| `--device` | `metal` | `metal` or `cpu` |
| `--shortlist` | per taxonomy (8 content, 3 ads) | stage-2 option count |
| `--stages` | `2` | `1` = wide choice only, `2` = wide + shortlist refine |
| `--head-max-len` | `512` | option-prompt token budget |
| `--max-len` | `8192` | total token budget |
| `--depth` | `1` | `1` = top level only, `0` = descend to leaves |
| `--read` | `auto` | state source: `auto` (abstract then text), `text`, or `abstract` |
| `--state-field` / `--subject` | `article` / `web article` | state key and question wording |
| `--temperature-by-options` | — | `TYPE:SIZE=TEMP`, e.g. `choice:11+=0.8` |
| `--id` / `--all` / `--state` | — | select records, or pass ad-hoc text |
| `--json` | — | full JSON results (includes per-record `timings`) |

`scripts/classify.py` is a thin wrapper with a slug CLI and per-taxonomy
defaults; it calls the same binary (`--classifier-bin` to override, otherwise
`target/release`, `target/debug`, then `cargo run`):

```bash
python scripts/classify.py --taxonomy content-taxonomy-3.1 --all
python scripts/classify.py --taxonomy ad-product-taxonomy-2.0 --state "Luxury SUVs"
```

## Strategy

A single `choice` over a 34-46-option taxonomy is above Laya's ~20-option sweet
spot. The classifier refines it in two stages, **one forward pass each**:

1. **Wide stage** — one `choice` over every candidate at the level; keep the
   top-N (`--shortlist`) by probability.
2. **Refine stage** — one small `choice` over that shortlist.

Repeat per level to build a path (`--depth 0`). Both stages are single batched
calls, so this is **2 questions per level** and runs in ~0.2s/record on Metal.

## Data

Each taxonomy is a nested tree: `tree` is the list of root nodes and every node
embeds its `children` (leaves have no `children` key). `id` is the unique key
used in the Laya `choice` criteria.

```json
{
  "id": "ad-product-taxonomy-2.0",
  "name": "IAB Ad Product Taxonomy",
  "version": "2.0",
  "node_count": 583,
  "root_count": 46,
  "tree": [
    { "id": "1002", "name": "Alcohol",
      "children": [ { "id": "1003", "name": "Bars" }, { "id": "1004", "name": "Beer" } ] }
  ]
}
```

| Taxonomy | Version | Nodes | Roots | Tiers | Source |
|---|---|---|---|---|---|
| Ad Product | 2.0 | 583 | 46 | 3 | [IAB](https://iabtechlab.com/standards/ad-product-taxonomy/) |
| Content | 3.1 | 704 | 37 | 4 | [IAB](https://iabtechlab.com/standards/content-taxonomy/) |

Inputs: `data/inputs/articles.json` / `data/inputs/ad-products.json`, each
`{id, text, abstract?, url?, language?}`. Articles are classified from their
English `abstract` (a clean summary beats raw article text); ad creatives from
`text`.

## Results

On the bundled samples, two stages (`--stages 2`):

| article | label | conf |
|---|---|---|
| [Rogue OpenAI agent 'infiltrated' Australian government website in world first](https://www.bbc.co.uk/news/articles/c6vgy0333dppo) | Technology & Computing | 0.80 |
| [Fury v Joshua announced for 11 December in Cardiff](https://www.bbc.co.uk/sport/boxing/articles/c89n188d53no) | Sports | 0.90 |
| [Indian billionaire's payments firm plots biggest London flotation in years](https://www.theguardian.com/business/2026/sep/23/indian-billionaire-payments-firm-london-flotation-airtel-money) | Business and Finance | 0.94 |
| [Hurricane Polo, rapidly intensifying from strong El Niño patterns, rages off coast of Mexico](https://www.theguardian.com/world/2026/sep/22/hurricane-polo-mexico) | Disasters | 0.82 |
| [« L'idée d'un livre écrit par l'IA me dégoûte »… L'affaire Thélyson Orélien vue depuis les librairies](https://www.20minutes.fr/arts-stars/culture/rentree_litteraire/4248372-20260924-idee-livre-ecrit-ia-degoute-affaire-thelyson-orelien-vue-depuis-librairies) | Books and Literature | 0.97 |

| set | labels |
|---|---|
| content (5 articles) | **5/5** |
| ad products (5 creatives) | **5/5** |

For comparison, stage 1 alone already scores 5/5 on content (the wide choice is
decisive on these articles), but only 3/5 on ad products — the refine stage is
what fixes ad-1 and ad-3. The shortlist sizes are tuned per taxonomy on these
small samples — content needs 8, ads need 3 — so treat them as starting points,
not constants.

## Notes and limits

Pulled from Laya's own model card, relevant here:

- **High-cardinality choices degrade past ~20 options.** This is why the
  two-stage shortlist exists; raise `--head-max-len` for wide stages if needed.
- **Non-English state hurts the English checkpoint** (the ad-4 sample is French;
  it only classifies once translated to English). Use English state, or
  `--model convaiinnovations/laya-multilingual` for non-Latin scripts.
- **Confidence ships over-confident.** Refit a temperature per question type on
  your own data before trusting probabilities; `--temperature-by-options`
  calibrates but cannot change the chosen label.
- **Long inputs** need a larger `--max-len` (up to 8192 on the multilingual
  encoder).

## Performance

See [BENCHMARK.md](BENCHMARK.md). Summary: ~0.2s/record on Metal, ~4-5s/record
on CPU (2 questions/record, model load is negligible).

## References

- https://iabtechlab.com/standards/ad-product-taxonomy/
- https://github.com/InteractiveAdvertisingBureau/Taxonomies/blob/main/Ad%20Product%20Taxonomies/Ad%20Product%20Taxonomy%202.0.tsv
- https://iabtechlab.com/standards/content-taxonomy/
- https://github.com/InteractiveAdvertisingBureau/Taxonomies/blob/develop/Content%20Taxonomies/Content%20Taxonomy%203.1.tsv
- https://huggingface.co/convaiinnovations/laya
