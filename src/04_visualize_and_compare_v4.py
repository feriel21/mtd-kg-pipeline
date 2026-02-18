#!/usr/bin/env python3
"""
04_visualize_and_compare_v4.py — Formal KG evaluation + interoperability exports.

Backward-compatible name (called by run_pipeline_v4.sh).

Outputs (journal-ready):
1) Full metrics JSON for each run (Rank-A suite)
2) RDF/Turtle export for selected run
3) LaTeX-ready comparison table Run1–Run4

FIX (important):
- Hallucination rate now uses _verdict if present (SUPPORTED/NOT_SUPPORTED/UNCERTAIN)
- Hallucination is computed on DECIDED cases only:
    NOT_SUPPORTED / (SUPPORTED + NOT_SUPPORTED)
- If only _verified exists (bool), it falls back to old behavior.
- Prints verification breakdown: supported / not_supported / uncertain / decided.
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter
import numpy as np

# Optional but recommended
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ─────────────────────────────────────────────────────────────────────────────
# LB2019 reference edges
# ─────────────────────────────────────────────────────────────────────────────

LB_REFERENCE_EDGES = [
    ("flow behavior", "affects", "internal facies"),
    ("sedimentation rate", "controls", "thickness"),
    ("topographic confinement", "controls", "morphology"),
    ("material heterogeneity", "affects", "flow behavior"),
    ("fluid overpressure", "triggers", "slope failure"),
    ("seismic loading", "triggers", "slope failure"),
    ("burial compaction", "controls", "thickness"),
    ("erosion", "affects", "basal surface"),
    ("remobilization", "affects", "transparent facies"),
    ("frontal compression", "affects", "upper surface"),
    ("retrogressive failure", "causes", "headscarp"),
    ("mass transport deposit", "hasDescriptor", "chaotic"),
    ("mass transport deposit", "hasDescriptor", "transparent"),
    ("mass transport deposit", "hasDescriptor", "blocky"),
    ("mass transport deposit", "hasDescriptor", "massive"),
    ("mass transport deposit", "hasDescriptor", "hummocky"),
    ("turbidite", "hasDescriptor", "layered"),
    ("turbidite", "hasDescriptor", "continuous"),
    ("mass transport deposit", "occursIn", "continental slope"),
    ("mass transport deposit", "occursIn", "continental margin"),
    ("debris flow", "occursIn", "slope"),
    ("turbidite", "occursIn", "submarine fan"),
    ("headscarp", "partOf", "mass transport deposit"),
    ("toe", "partOf", "mass transport deposit"),
    ("basal surface", "partOf", "mass transport deposit"),
    ("upper surface", "partOf", "mass transport deposit"),
]


def load_triples_jsonl(path: str):
    triples = []
    if not path or not os.path.exists(path):
        return triples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                triples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return triples


def compute_lb_recall(triples):
    found = []
    missing = []
    for s_ref, r_ref, t_ref in LB_REFERENCE_EDGES:
        matched = False
        for t in triples:
            s = (t.get("source_norm") or "").lower()
            r = (t.get("relation_norm") or "").lower()
            tg = (t.get("target_norm") or "").lower()

            s_ref_l = s_ref.lower()
            r_ref_l = r_ref.lower()
            t_ref_l = t_ref.lower()

            # require relation match to avoid false positives
            if r != r_ref_l:
                continue

            # exact
            if s == s_ref_l and tg == t_ref_l:
                matched = True
                break

            # fuzzy contains
            if (s_ref_l in s or s in s_ref_l) and (t_ref_l in tg or tg in t_ref_l):
                matched = True
                break

        if matched:
            found.append((s_ref, r_ref, t_ref))
        else:
            missing.append((s_ref, r_ref, t_ref))
    recall = len(found) / len(LB_REFERENCE_EDGES) if LB_REFERENCE_EDGES else 0.0
    return recall, found, missing


def _compute_verification_metrics(triples):
    """
    Returns dict:
      {
        verified_count,
        supported_count,
        not_supported_count,
        uncertain_count,
        decided_count,
        hallucination_rate (decided-only) or None
      }

    Priority:
      - Use _verdict if present (SUPPORTED/NOT_SUPPORTED/UNCERTAIN)
      - Else fall back to _verified boolean (True/False)
    """
    verified = [t for t in triples if ("_verdict" in t) or ("_verified" in t)]
    if not verified:
        return {
            "verified_count": 0,
            "supported_count": 0,
            "not_supported_count": 0,
            "uncertain_count": 0,
            "decided_count": 0,
            "hallucination_rate": None,
        }

    supported = 0
    not_supported = 0
    uncertain = 0

    for t in verified:
        v = t.get("_verdict", None)
        if v is None:
            # backward compatibility
            if bool(t.get("_verified")):
                supported += 1
            else:
                not_supported += 1
            continue

        v = str(v).upper().strip()
        if v == "SUPPORTED":
            supported += 1
        elif v == "NOT_SUPPORTED":
            not_supported += 1
        else:
            uncertain += 1

    decided = supported + not_supported
    if decided > 0:
        halluc = not_supported / decided
    else:
        halluc = None

    return {
        "verified_count": len(verified),
        "supported_count": supported,
        "not_supported_count": not_supported,
        "uncertain_count": uncertain,
        "decided_count": decided,
        "hallucination_rate": halluc,
    }


def compute_all_metrics(triples):
    """Compute the full Rank-A evaluation suite."""
    metrics = {}

    # 1) Volume
    metrics["total_triples"] = len(triples)
    nodes = set((t.get("source_norm") or "") for t in triples) | \
            set((t.get("target_norm") or "") for t in triples)
    nodes.discard("")
    metrics["unique_nodes"] = len(nodes)

    # 2) LB2019 Recall
    recall, found, missing = compute_lb_recall(triples)
    metrics["lb_recall"] = float(recall)
    metrics["lb_found"] = len(found)
    metrics["lb_missing"] = len(missing)

    # 3) Ontology Conformance Rate
    passes = sum(1 for t in triples if t.get("_constraint_ok", True))
    metrics["ontology_conformance"] = passes / max(1, len(triples))

    # 4) Verification + Hallucination Rate (FIXED)
    vstats = _compute_verification_metrics(triples)
    metrics["verified_count"] = vstats["verified_count"]
    metrics["verified_supported"] = vstats["supported_count"]
    metrics["verified_not_supported"] = vstats["not_supported_count"]
    metrics["verified_uncertain"] = vstats["uncertain_count"]
    metrics["verified_decided"] = vstats["decided_count"]
    metrics["hallucination_rate"] = vstats["hallucination_rate"]

    # 5) Relation Balance Entropy + relation distribution
    rel_dist = Counter((t.get("relation_norm") or "unknown") for t in triples)
    metrics["relation_distribution"] = dict(rel_dist)

    counts = np.array(list(rel_dist.values()), dtype=float) if rel_dist \
        else np.array([1.0], dtype=float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
    metrics["relation_entropy"] = float(entropy)
    metrics["relation_balance"] = float(entropy / max_entropy) \
        if max_entropy > 0 else 0.0

    total = max(1, len(triples))
    metrics["relatedTo_pct"] = 100.0 * rel_dist.get("relatedTo", 0) / total
    metrics["hasDescriptor_pct"] = 100.0 * rel_dist.get("hasDescriptor", 0) / total

    # 6) Multi-Paper Support
    support_dist = Counter(int(t.get("_support_count", 1) or 1) for t in triples)
    metrics["support_distribution"] = dict(support_dist)
    metrics["multi_paper_triples"] = int(
        sum(c for n, c in support_dist.items() if n >= 2)
    )

    # 7) Graph Topology
    if HAS_NX:
        G = nx.DiGraph()
        for t in triples:
            s = t.get("source_norm") or ""
            tg = t.get("target_norm") or ""
            r = t.get("relation_norm") or ""
            if not s or not tg or s == tg:
                continue
            G.add_edge(s, tg, relation=r)
        metrics["graph_nodes"] = int(G.number_of_nodes())
        metrics["graph_edges"] = int(G.number_of_edges())
        metrics["graph_density"] = float(nx.density(G)) \
            if G.number_of_nodes() > 1 else 0.0
        degs = [d for _, d in G.degree()] if G.number_of_nodes() else [0]
        metrics["avg_degree"] = float(np.mean(degs))
        metrics["connected_components"] = int(
            nx.number_weakly_connected_components(G)
        ) if G.number_of_nodes() else 0
    else:
        for k in ["graph_nodes", "graph_edges", "graph_density",
                  "avg_degree", "connected_components"]:
            metrics[k] = None

    # 8) Descriptor Coverage (LB2019 13 descriptors)
    LB_DESC = {
        "chaotic", "transparent", "blocky", "massive", "hummocky",
        "discontinuous", "high-amplitude", "low-amplitude",
        "undeformed", "layered", "stratified", "continuous", "parallel"
    }
    found_desc = set()
    for t in triples:
        for term in [(t.get("source_norm") or ""), (t.get("target_norm") or "")]:
            term_l = term.lower()
            for d in LB_DESC:
                if d in term_l:
                    found_desc.add(d)
    metrics["descriptor_coverage"] = len(found_desc) / len(LB_DESC)
    metrics["descriptors_found"] = sorted(found_desc)
    metrics["descriptors_missing"] = sorted(LB_DESC - found_desc)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# RDF/Turtle export
# ─────────────────────────────────────────────────────────────────────────────

def export_rdf_turtle(triples, output_path):
    PREFIX = """@prefix geo: <http://example.org/geo-kg/> .
@prefix lb: <http://example.org/lb2019/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix prov: <http://www.w3.org/ns/prov#> .

"""

    def uri(text):
        return (text or "").strip().replace(" ", "_").replace("-", "_").replace("/", "_")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(PREFIX)
        for t in triples:
            s_norm = t.get("source_norm") or ""
            p_norm = t.get("relation_norm") or ""
            o_norm = t.get("target_norm") or ""
            if not s_norm or not p_norm or not o_norm:
                continue

            s = f"geo:{uri(s_norm)}"
            p = f"lb:{uri(p_norm)}"
            o = f"geo:{uri(o_norm)}"
            f.write(f"{s} {p} {o} .\n")

            st = t.get("source_type_norm")
            tt = t.get("target_type_norm")
            if st:
                f.write(f"{s} a lb:{uri(st)} .\n")
            if tt:
                f.write(f"{o} a lb:{uri(tt)} .\n")

            prov_data = t.get("_provenance", {})
            if prov_data and isinstance(prov_data, dict):
                q = prov_data.get("query")
                if q:
                    qid = uri(f"query_{abs(hash(q)) % 10**10}")
                    f.write(f"geo:{qid} a prov:Entity .\n")
                    escaped_q = q.replace('"', '\\"')
                    f.write(f'geo:{qid} rdfs:label "{escaped_q}" .\n')
                    f.write(f"{s} prov:wasDerivedFrom geo:{qid} .\n")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX comparison table
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, kind="float"):
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if isinstance(v, (float, np.floating)):
        if kind == "pct":
            return f"{v:.1f}"
        if kind == "ratio":
            return f"{v:.2f}"
        return f"{v:.3f}"
    return str(v)


def generate_latex_table(metrics_by_run, output_path):
    header = r"""
\begin{table}[htb]
\centering
\caption{KG extraction quality across pipeline iterations.}
\label{tab:kg-comparison}
\begin{tabular}{lrrrr}
\toprule
Metric & Run 1 & Run 2 & Run 3 & Run 4 \\
\midrule
"""
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""

    rows = [
        ("Clean triples", ("total_triples", "int")),
        ("Unique nodes", ("unique_nodes", "int")),
        ("relatedTo (\\%)", ("relatedTo_pct", "pct")),
        ("hasDescriptor (\\%)", ("hasDescriptor_pct", "pct")),
        ("LB2019 recall", ("lb_recall", "ratio")),
        ("Desc. coverage (13)", ("descriptor_coverage", "ratio")),
        ("Ontology conformance", ("ontology_conformance", "ratio")),
        ("Hallucination rate", ("hallucination_rate", "ratio")),
        ("Relation balance", ("relation_balance", "ratio")),
        ("Graph density", ("graph_density", "ratio")),
    ]

    def get(run, key):
        return metrics_by_run.get(run, {}).get(key, None)

    lines = [header.strip("\n")]
    for label, (key, kind) in rows:
        vals = []
        for run in ["run1", "run2", "run3", "run4"]:
            v = get(run, key)
            if kind == "int":
                vals.append(_fmt(v))
            elif kind == "pct":
                vals.append(_fmt(v, "pct"))
            else:
                vals.append(_fmt(v, "ratio"))
        line = f"{label} & {' & '.join(vals)} \\\\"
        lines.append(line)
    lines.append(footer.strip("\n"))

    table = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(table)
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Formal KG evaluation + exports (v4)")
    p.add_argument("--run1", default="", help="Path to cleaned_triples_run1.jsonl (optional)")
    p.add_argument("--run2", default="", help="Path to cleaned_triples_run2.jsonl (optional)")
    p.add_argument("--run3", default="", help="Path to cleaned_triples_v3.jsonl (optional)")
    p.add_argument("--run4", default=os.environ.get("KG_RUN4", ""), help="Path to cleaned_triples_v4.jsonl")
    p.add_argument("--outdir", default=os.environ.get("KG_EVAL_DIR", "output/eval_v4"), help="Output directory")
    p.add_argument("--rdf-run", default="run4", choices=["run1", "run2", "run3", "run4"], help="Which run to export as RDF/Turtle")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = {"run1": args.run1, "run2": args.run2, "run3": args.run3, "run4": args.run4}

    metrics_by_run = {}
    triples_by_run = {}

    for run_name, path in runs.items():
        if not path:
            continue
        triples = load_triples_jsonl(path)
        triples_by_run[run_name] = triples
        metrics = compute_all_metrics(triples)
        metrics["input_path"] = path
        metrics_by_run[run_name] = metrics

        mpath = outdir / f"{run_name}_metrics.json"
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"  {run_name}: {len(triples)} triples -> {mpath}")

    if not metrics_by_run:
        print("  No runs provided/found. Provide at least --run4 path.")
        return

    # RDF/Turtle export
    rdf_run = args.rdf_run
    if rdf_run in triples_by_run and triples_by_run[rdf_run]:
        turtle_path = outdir / f"{rdf_run}.ttl"
        export_rdf_turtle(triples_by_run[rdf_run], str(turtle_path))
        print(f"  RDF/Turtle exported: {turtle_path}")
    else:
        print(f"  RDF export skipped (no triples for {rdf_run}).")

    # LaTeX table
    latex_path = outdir / "kg_comparison_table.tex"
    _ = generate_latex_table(metrics_by_run, latex_path)
    print(f"  LaTeX table: {latex_path}")

    # Quick console summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY (v4)")
    print("=" * 70)
    for run_name in ["run1", "run2", "run3", "run4"]:
        if run_name not in metrics_by_run:
            continue
        m = metrics_by_run[run_name]

        halluc = "—"
        if m.get("hallucination_rate") is not None:
            halluc = f"{m['hallucination_rate']:.2f}"

        print(
            f"  {run_name.upper():4s} | "
            f"triples={m['total_triples']:4d} "
            f"nodes={m['unique_nodes']:4d} "
            f"LB_recall={m['lb_recall']:.2f} "
            f"Desc_cov={m['descriptor_coverage']:.2f} "
            f"Onto={m['ontology_conformance']:.2f} "
            f"RelBal={m['relation_balance']:.2f} "
            f"relatedTo%={m['relatedTo_pct']:.1f} "
            f"hasDesc%={m['hasDescriptor_pct']:.1f} "
            f"halluc={halluc} "
            f"(verif S/NS/U/dec={m.get('verified_supported',0)}/{m.get('verified_not_supported',0)}/{m.get('verified_uncertain',0)}/{m.get('verified_decided',0)})"
        )


if __name__ == "__main__":
    main()
