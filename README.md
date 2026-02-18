# MTD-KG: Ontology-Constrained Knowledge Graph Construction for Mass Transport Deposits

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Ontology-Constrained RAG Extraction with Lightweight LLMs for Domain Knowledge Graph Construction: Application to Mass-Transport Deposits**
>
> Feryal Talbi, John Armitage, Alain Rabaute, Jean Charlety, Jean-Noel Vittaut, Antoine Bouziat, Sylvie Leroy — Sorbonne University / IFPEN / LIP6

---

## Overview

This repository contains the complete pipeline for constructing a geological knowledge graph (KG) from scientific literature on **Mass Transport Deposits (MTDs)** — submarine landslides that reshape continental margins.

The pipeline extracts structured **subject–relation–object triples** from a corpus of 41 peer-reviewed papers using:
- **Retrieval-Augmented Generation (RAG)** with BM25 indexing
- **Lightweight local LLMs** (Qwen 2.5, 1.5B–7B parameters)
- **Ontology-constrained extraction** guided by the Le Bouteiller (2019) geological schema
- **GraphJudge-inspired verification** to reduce hallucinations
- **SciBERT embedding-based canonicalization** for entity merging

The pipeline was developed iteratively across **4 runs**, each introducing architectural improvements. All runs are documented with full metrics and reproducible scripts.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PDF Corpus (41 papers)                │
└──────────────────────────┬──────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  00_prepare  │  Text chunking + BM25 indexing
                    │    _index    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ 02_extract  │  RAG + LLM triple extraction
                    │  _triples   │  (multi-strategy, few-shot)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │02b_verify   │  CoT verification against source
                    │  _triples   │  (STRONG/WEAK/NOT_SUPPORTED)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ 03_validate │  Type constraints + lexicon filter
                    │  _and_clean │  + verification-aware filtering
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ 03b_canon   │  SciBERT entity clustering
                    │  icalize    │  + deduplication
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ 04_evaluate │  Metrics, RDF export, LaTeX tables
                    └─────────────┘
```

---

## Run Comparison

The pipeline evolved across 4 runs, each addressing specific weaknesses:

| Metric | Run 1 | Run 2 | Run 3 | Run 4 |
|--------|------:|------:|------:|------:|
| **Raw triples** | 629 | 613 | 142 | 242 |
| **Clean triples** | 161 | 275 | 117 | 137 |
| **Unique nodes** | 189 | 377 | 90 | 88 |
| **relatedTo %** | 28.6% | 54.5% | 0.0% | 0.0% |
| **hasDescriptor %** | 1.2% | 1.1% | 62.4% | 64.2% |
| **LB2019 recall** | ~30% | ~25% | 62% | 65.4% |
| **Descriptor coverage** | 4/13 | 4/13 | 11/13 | 13/13 |
| **Ontology conformance** | — | — | — | 100% |
| **Relation balance** | — | — | — | 0.58 |
| **Hallucination rate** | — | — | — | 0.78* |

\* *Run 4 measures hallucination but does not yet filter NOT_SUPPORTED triples. Run 5 (in progress) applies verification-aware filtering to bring this below 0.15.*

### Run Specifications

| Feature | Run 1 | Run 2 | Run 3 | Run 4 |
|---------|-------|-------|-------|-------|
| **Corpus** | w/ Le Bouteiller | w/o Le Bouteiller | w/o Le Bouteiller | w/o Le Bouteiller |
| **LLM** | Qwen 1.5B | Qwen 1.5B | Qwen 1.5B | Qwen 2.5 7B |
| **Extraction strategy** | 1 (SVO) | 1 (SVO) | 3 (Causal+Desc+Context) | 3 + gap-filling |
| **Few-shot examples** | No | No | Yes | Yes |
| **Retrieval** | BM25 | BM25 | BM25 | BM25 + SciBERT rerank |
| **Verification** | No | No | No | GraphJudge-style CoT |
| **Validation** | None | Partial | Strict | SHACL type constraints |
| **Canonicalization** | No | No | No | SciBERT clustering |
| **relatedTo handling** | Kept | Kept | Reclassified/removed | Schema-constrained |
| **Provenance tracking** | No | No | Yes | Yes + chunk text |
| **RDF export** | No | No | No | Yes (Turtle) |

---

## Quick Start

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Step 0: Prepare index (run once)
python -u src/00_prepare_index.py \
    --chunks-dir output/step1

# Step 1: Extract triples
python -u src/02_rag_extract_triples_v4.py \
    --index-dir output/step1 \
    --output output/step4/raw_triples_v4.jsonl

# Step 2: Verify triples (requires GPU or patience on CPU)
python -u src/02b_verify_triples_v5.py \
    --input output/step4/raw_triples_v4.jsonl \
    --output output/step4/verified_triples_v5.jsonl \
    --backend hf --model Qwen/Qwen2.5-7B-Instruct

# Step 3: Validate, clean, canonicalize
python -u src/03_validate_and_clean_v5.py \
    --input output/step4/verified_triples_v5.jsonl \
    --outdir output/step4 \
    --verif-policy normal

# Step 4: Evaluate
python -u src/04_visualize_and_compare_v4.py \
    --run4 output/step4/canonical_triples_v5.jsonl \
    --outdir output/eval_v5 --rdf-run run5
```

### SLURM (GPU cluster)

```bash
sbatch scripts/run_kg_v5.sh
```

---

## Repository Structure

```
mtd-kg-pipeline/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
│
├── src/                               # Pipeline source code
│   ├── 00_prepare_index.py            # Text chunking + BM25 indexing
│   ├── 02_rag_extract_triples_v4.py   # RAG extraction (v4: multi-strategy)
│   ├── 02b_verify_triples_v5.py       # GraphJudge-style verification
│   ├── 03_validate_and_clean_v5.py    # Validation + verification filtering
│   ├── 04_visualize_and_compare_v4.py # Evaluation metrics + RDF export
│   ├── audit_verification.py          # Manual audit helper
│   └── rag/                           # RAG utilities
│       ├── __init__.py
│       ├── llm_hf.py                  # HuggingFace LLM backend
│       ├── llm_ollama.py              # Ollama LLM backend
│       ├── chunking.py                # Text chunking
│       ├── extract.py                 # Triple extraction logic
│       └── postprocess.py             # Normalization + filtering
│
├── configs/                           # Configuration files
│   ├── schema_step1.json              # Ontology schema (5 node types, 14 relations)
│   ├── lexicon.json                   # Geological lexicon (Le Bouteiller 2019)
│   └── lb_reference_edges.json        # 26 reference edges for recall
│
├── results/                           # Run results (metrics + outputs)
│   ├── run1/                          # spaCy SVO, with Le Bouteiller corpus
│   │   └── RUN_CARD.md
│   ├── run2/                          # spaCy SVO, without Le Bouteiller
│   │   └── RUN_CARD.md
│   ├── run3/                          # RAG + Qwen 1.5B, 3 strategies
│   │   └── RUN_CARD.md
│   └── run4/                          # RAG + verification + SHACL
│       └── RUN_CARD.md
│
├── scripts/                           # SLURM and utility scripts
│   ├── run_kg_v5.sh                   # Full pipeline SLURM job
│   └── run_policy_sweep.sh            # Compare verification policies
│
└── docs/                              # Documentation
    ├── PIPELINE_EVOLUTION.md          # Detailed run-by-run changelog
    └── METRICS.md                     # Metric definitions
```

---

## Metrics Reference

| Metric | Definition | Target |
|--------|-----------|--------|
| **LB Recall** | Fraction of 26 Le Bouteiller (2019) reference edges recovered | > 65% |
| **Descriptor Coverage** | Fraction of 13 canonical seismic descriptors found | 13/13 |
| **Ontology Conformance** | % of triples satisfying SHACL-style type constraints | > 95% |
| **Hallucination Rate** | % of triples marked NOT_SUPPORTED by verification | < 15% |
| **Relation Balance** | Normalized Shannon entropy of relation distribution | > 0.5 |
| **relatedTo %** | % of triples using the generic catch-all relation | 0% |

---

## Citation

```bibtex

```

---

## License


The geological corpus used in this work is derived from published scientific literature and is not redistributed. Users must obtain their own copies of the source papers.

## Acknowledgments
This project is co-funded by the European Union’s Horizon Europe research and innovation programme Cofund SOUND.AI under the Marie Sklodowska-Curie Grant Agreement No 101081674.

