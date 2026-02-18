#!/usr/bin/env python3
"""
03_validate_and_clean_v5.py — Validation, Cleaning & Canonicalization
======================================================================

Upgrades from v4:
  1. VERIFICATION-AWARE FILTERING: Rejects NOT_SUPPORTED triples from the
     verification step. This is the single biggest change for reducing
     hallucination rate.
  2. Configurable verification policy via --verif-policy:
       strict  — keep only STRONG_SUPPORT
       normal  — keep STRONG_SUPPORT + WEAK_SUPPORT (default)
       relaxed — keep everything except NOT_SUPPORTED
       off     — ignore verification (v4 behaviour)
  3. NO_CHUNK triples are treated as UNCERTAIN and handled by policy.
  4. Better reporting: separate rejection reasons for verification vs
     validation so you can diagnose whether the problem is extraction
     quality or verification accuracy.
  5. All v4 features preserved: type constraints, descriptor partial
     matching, LB recall, canonicalization, embedding dedup.

Inputs:
  - verified_triples_v5.jsonl  (from 02b_verify_triples_v5)

Outputs:
  - canonical_triples_v5.jsonl
  - canonical_map_v5.json
  - rejected_triples_v5.jsonl  (NEW: all rejected triples with reasons)
  - cleaning_stats_v5.json

Usage:
  # Default (normal policy — keeps STRONG + WEAK):
  python -u 03_validate_and_clean_v5.py \\
      --input  output/step4/verified_triples_v5.jsonl \\
      --outdir output/step4

  # Strict (only STRONG_SUPPORT survives):
  python -u 03_validate_and_clean_v5.py \\
      --input  output/step4/verified_triples_v5.jsonl \\
      --outdir output/step4 \\
      --verif-policy strict

  # Compare policies quickly:
  for p in strict normal relaxed off; do
      python -u 03_validate_and_clean_v5.py \\
          --input output/step4/verified_triples_v5.jsonl \\
          --outdir output/step4_$p --verif-policy $p 2>&1 | tail -20
  done
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Le Bouteiller (2019) reference descriptors
LB_DESCRIPTORS = {
    "blocky", "chaotic", "continuous", "discontinuous",
    "high-amplitude", "hummocky", "layered", "low-amplitude",
    "massive", "parallel", "stratified", "transparent", "undeformed",
}

# Le Bouteiller (2019) reference edges for recall computation
LB_REFERENCE_EDGES = [
    ("mass transport deposit", "hasDescriptor", "chaotic"),
    ("mass transport deposit", "hasDescriptor", "transparent"),
    ("mass transport deposit", "hasDescriptor", "hummocky"),
    ("mass transport deposit", "hasDescriptor", "blocky"),
    ("mass transport deposit", "hasDescriptor", "discontinuous"),
    ("mass transport deposit", "hasDescriptor", "massive"),
    ("turbidite", "hasDescriptor", "parallel"),
    ("turbidite", "hasDescriptor", "continuous"),
    ("turbidite", "hasDescriptor", "layered"),
    ("turbidite", "hasDescriptor", "high-amplitude"),
    ("debris flow", "hasDescriptor", "chaotic"),
    ("debris flow", "hasDescriptor", "hummocky"),
    ("slide", "hasDescriptor", "blocky"),
    ("slide", "hasDescriptor", "undeformed"),
    ("hemipelagite", "hasDescriptor", "parallel"),
    ("hemipelagite", "hasDescriptor", "continuous"),
    ("hemipelagite", "hasDescriptor", "low-amplitude"),
    ("slope failure", "causes", "mass transport deposit"),
    ("earthquake", "triggers", "slope failure"),
    ("pore pressure", "controls", "slope failure"),
    ("turbidity current", "formedBy", "debris flow"),
    ("mass transport deposit", "occursIn", "continental slope"),
    ("mass transport deposit", "occursIn", "abyssal plain"),
    ("debris flow", "occursIn", "continental slope"),
    ("turbidite", "occursIn", "basin floor"),
    ("slide", "overlies", "hemipelagite"),
]

# Valid relations (ontology)
VALID_RELATIONS = {
    "hasDescriptor", "occursIn", "formedBy", "partOf",
    "triggers", "causes", "controls", "affects",
    "overlies", "underlies", "associatedWith",
    "contains", "transports", "erodes", "deposits",
}

# Type constraints: which entity types can appear as subject/object
# for each relation
TYPE_CONSTRAINTS = {
    "hasDescriptor": {"object_type": "Descriptor"},
    "occursIn":      {"object_type": "Setting"},
    "overlies":      {"subject_type": "Geological_Object", "object_type": "Geological_Object"},
    "underlies":     {"subject_type": "Geological_Object", "object_type": "Geological_Object"},
}

# Known descriptor terms (from LB2019 + extensions)
KNOWN_DESCRIPTORS = LB_DESCRIPTORS | {
    "mounded", "divergent", "convergent", "wavy", "contorted",
    "folded", "faulted", "deformed", "disrupted", "draping",
    "onlapping", "erosional", "aggradational", "progradational",
    "retrogradational", "tabular", "lenticular", "wedge-shaped",
    "sheet-like", "channelised", "irregular", "smooth", "rough",
    "thick", "thin", "variable-amplitude", "moderate-amplitude",
}

# Known settings/environments
KNOWN_SETTINGS = {
    "continental slope", "continental shelf", "continental margin",
    "abyssal plain", "basin floor", "submarine canyon", "channel",
    "deep-water environment", "deep-water environments", "deep water",
    "passive margin", "active margin", "accretionary prism",
    "trench", "mid-ocean ridge", "seamount", "delta", "fan",
    "submarine fan", "levee", "overbank",
}

# Vague / too-generic terms to reject
VAGUE_TERMS = {
    "it", "they", "this", "that", "these", "those",
    "something", "thing", "stuff", "area", "region",
    "feature", "process", "event", "result", "effect",
    "study", "analysis", "data", "figure", "table",
    "example", "case", "type", "kind", "form",
}

# Blacklist patterns (case-study names, geographic locations, etc.)
BLACKLIST_PATTERNS = [
    r"^fig(ure)?\.?\s*\d",
    r"^\d+(\.\d+)?$",
    r"^table\s+\d",
    r"^section\s+\d",
    r"^et\s+al",
    r"^\w{1,2}$",           # single/double char
]
BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS]


# ═══════════════════════════════════════════════════════════════════════
#  VERIFICATION FILTER
# ═══════════════════════════════════════════════════════════════════════

def check_verification(triple: dict, policy: str) -> tuple[bool, str]:
    """
    Check whether a triple passes the verification policy.

    Policies:
      strict  — only STRONG_SUPPORT passes
      normal  — STRONG_SUPPORT and WEAK_SUPPORT pass
      relaxed — everything except NOT_SUPPORTED passes
      off     — everything passes (v4 behaviour)

    Returns (passes: bool, reason: str)
    """
    if policy == "off":
        return True, "ok"

    verif = triple.get("_verification", {})
    verdict = verif.get("verdict", "MISSING")

    if verdict == "MISSING":
        # No verification data — treat as unverified
        if policy == "strict":
            return False, "verif_missing"
        else:
            return True, "ok_unverified"

    if policy == "strict":
        if verdict == "STRONG_SUPPORT":
            return True, "ok"
        else:
            return False, f"verif_rejected_{verdict.lower()}"

    elif policy == "normal":
        if verdict in ("STRONG_SUPPORT", "WEAK_SUPPORT"):
            return True, "ok"
        elif verdict == "NO_CHUNK":
            # No chunk = can't verify — keep but flag
            return True, "ok_no_chunk"
        elif verdict == "UNPARSEABLE":
            # LLM failed to parse — keep but flag
            return True, "ok_unparseable"
        else:
            return False, f"verif_not_supported"

    elif policy == "relaxed":
        if verdict == "NOT_SUPPORTED":
            return False, "verif_not_supported"
        else:
            return True, "ok"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════
#  VALIDATION CHECKS (same as v4 + improvements)
# ═══════════════════════════════════════════════════════════════════════

def normalize_entity(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Remove trailing punctuation
    text = text.rstrip(".,;:")
    return text


def check_basic(triple: dict) -> tuple[bool, str]:
    """Basic sanity checks."""
    s = triple.get("source_norm", triple.get("source", "")).strip()
    t = triple.get("target_norm", triple.get("target", "")).strip()
    r = triple.get("relation_norm", triple.get("relation", "")).strip()

    if not s or not t or not r:
        return False, "empty_field"
    if len(s) < 3 or len(t) < 3:
        return False, "too_short"
    if s.lower() == t.lower():
        return False, "self_loop"

    # Blacklist
    for pattern in BLACKLIST_RE:
        if pattern.match(s):
            return False, f"blacklisted_source"
        if pattern.match(t):
            return False, f"blacklisted_target"

    # Vague terms
    s_vague = normalize_entity(s) in VAGUE_TERMS
    t_vague = normalize_entity(t) in VAGUE_TERMS
    if s_vague and t_vague:
        return False, "both_vague"
    if s_vague:
        return False, "vague_source"
    if t_vague:
        return False, "vague_target"

    return True, "ok"


def check_relation(triple: dict) -> tuple[bool, str]:
    """Check that the relation is in the ontology."""
    r = triple.get("relation_norm", triple.get("relation", ""))
    if r not in VALID_RELATIONS:
        return False, "invalid_relation"
    return True, "ok"


def check_type_constraint(triple: dict) -> tuple[bool, str]:
    """
    Check SHACL-style type constraints for specific relations.
    E.g., hasDescriptor requires object to be a known descriptor.
    """
    r = triple.get("relation_norm", triple.get("relation", ""))
    s = normalize_entity(triple.get("source_norm", triple.get("source", "")))
    t = normalize_entity(triple.get("target_norm", triple.get("target", "")))

    constraints = TYPE_CONSTRAINTS.get(r)
    if not constraints:
        return True, "ok"

    if r == "hasDescriptor":
        # Object must look like a descriptor
        if t not in KNOWN_DESCRIPTORS:
            # Partial match: check if any known descriptor is a substring
            partial = any(d in t or t in d for d in KNOWN_DESCRIPTORS)
            if partial:
                return False, "descriptor_partial"
            return False, "type_constraint"

    if r == "occursIn":
        # Object should be a setting/environment
        if t not in KNOWN_SETTINGS:
            # Allow if it contains setting-like keywords
            setting_kw = {"slope", "basin", "shelf", "margin", "canyon",
                          "fan", "plain", "deep", "environment", "channel",
                          "trench", "delta", "levee", "ridge"}
            if not any(kw in t for kw in setting_kw):
                return False, "type_constraint"

    return True, "ok"


def check_lexicon_coverage(triple: dict, lexicon: set) -> tuple[bool, str]:
    """
    Check if at least one entity is in the geological lexicon.
    Both being outside the lexicon suggests noise.
    """
    s = normalize_entity(triple.get("source_norm", triple.get("source", "")))
    t = normalize_entity(triple.get("target_norm", triple.get("target", "")))

    s_in = s in lexicon or any(term in s for term in lexicon if len(term) > 4)
    t_in = t in lexicon or any(term in t for term in lexicon if len(term) > 4)

    if not s_in and not t_in:
        return False, "both_not_in_lexicon"
    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════
#  DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════

def triple_key(triple: dict) -> str:
    """Canonical key for deduplication."""
    s = normalize_entity(triple.get("source_norm", triple.get("source", "")))
    r = triple.get("relation_norm", triple.get("relation", ""))
    t = normalize_entity(triple.get("target_norm", triple.get("target", "")))
    return f"{s}||{r}||{t}"


# ═══════════════════════════════════════════════════════════════════════
#  CANONICALIZATION (embedding-based entity merging)
# ═══════════════════════════════════════════════════════════════════════

def build_canonical_map(
    entities: list[str],
    distance_threshold: float = 0.06,
    lb_descriptors: set = LB_DESCRIPTORS,
) -> dict[str, str]:
    """
    Cluster entities by SciBERT cosine distance and build a merge map.
    Blocks merges between LB descriptor pairs (antonyms, etc.).
    """
    if len(entities) < 2:
        return {}

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        print("  WARNING: sentence-transformers or sklearn not available, "
              "skipping canonicalization.")
        return {}

    print(f"  Computing SciBERT embeddings for {len(entities)} entities...")
    print(f"  Loading SciBERT for entity embeddings...")
    model = SentenceTransformer("allenai/scibert_scivocab_uncased")
    embeddings = model.encode(entities, show_progress_bar=False)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Cosine distance clustering
    print(f"  Clustering (distance_threshold={distance_threshold})...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    # Build merge map: within each cluster, map all to the shortest name
    clusters = defaultdict(list)
    for ent, label in zip(entities, labels):
        clusters[label].append(ent)

    canonical_map = {}
    blocked = []

    for label, members in clusters.items():
        if len(members) < 2:
            continue

        # Check for blocked merges (both are LB descriptors)
        lb_members = [m for m in members if m in lb_descriptors]
        if len(lb_members) >= 2:
            # Block: don't merge LB descriptors with each other
            for m in members:
                blocked.append(
                    f"BLOCKED: '{m}' (cluster with {members}, "
                    f"both LB descriptors)"
                )
            continue

        # Pick canonical: prefer the longest, most specific form
        # (e.g., "deep-water environments" over "deep-water environment")
        canonical = max(members, key=lambda x: (len(x.split()), len(x)))

        for m in members:
            if m != canonical:
                canonical_map[m] = canonical

    if blocked:
        print(f"  Blocked {len(blocked)} bad merges:")
        for b in blocked[:10]:
            print(f"    {b}")

    if canonical_map:
        print(f"  Applied merges:")
        for old, new in sorted(canonical_map.items()):
            print(f"    '{old}' -> '{new}'")

    return canonical_map


def apply_canonical_map(triples: list[dict], canonical_map: dict) -> int:
    """Apply canonical map to all triples. Returns count of entities merged."""
    merged = 0
    for t in triples:
        for field in ["source_norm", "target_norm", "source", "target"]:
            val = t.get(field, "")
            norm = normalize_entity(val)
            if norm in canonical_map:
                t[field] = canonical_map[norm]
                merged += 1
    return merged


# ═══════════════════════════════════════════════════════════════════════
#  LB RECALL & DESCRIPTOR COVERAGE
# ═══════════════════════════════════════════════════════════════════════

def compute_lb_recall(triples: list[dict]) -> tuple[int, int, list]:
    """Compute recall against LB2019 reference edges."""
    found_keys = set()
    for t in triples:
        s = normalize_entity(t.get("source_norm", t.get("source", "")))
        r = t.get("relation_norm", t.get("relation", ""))
        o = normalize_entity(t.get("target_norm", t.get("target", "")))
        found_keys.add((s, r, o))

    hits = 0
    missing = []
    for s, r, o in LB_REFERENCE_EDGES:
        if (s, r, o) in found_keys:
            hits += 1
        else:
            missing.append((s, r, o))

    return hits, len(LB_REFERENCE_EDGES), missing


def compute_descriptor_coverage(triples: list[dict]) -> tuple[set, set]:
    """Check which LB descriptors appear as objects in hasDescriptor triples."""
    found = set()
    for t in triples:
        r = t.get("relation_norm", t.get("relation", ""))
        if r == "hasDescriptor":
            o = normalize_entity(t.get("target_norm", t.get("target", "")))
            if o in LB_DESCRIPTORS:
                found.add(o)
    missing = LB_DESCRIPTORS - found
    return found, missing


# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate, clean & canonicalize KG triples (v5)"
    )
    parser.add_argument("--input", default=None,
                        help="Input JSONL (default: $KG_INPUT)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: $KG_OUTPUT_DIR)")
    parser.add_argument("--verif-policy", default="normal",
                        choices=["strict", "normal", "relaxed", "off"],
                        help="Verification filtering policy (default: normal)")
    parser.add_argument("--lexicon", default=None,
                        help="Path to lexicon.json (optional, for coverage check)")
    parser.add_argument("--cluster-threshold", type=float, default=0.06,
                        help="Cosine distance threshold for entity clustering")
    args = parser.parse_args()

    # Resolve paths
    input_path = args.input or os.environ.get("KG_INPUT", "")
    outdir = args.outdir or os.environ.get("KG_OUTPUT_DIR", "output/step4")

    if not input_path:
        print("ERROR: Provide --input or set $KG_INPUT")
        sys.exit(1)

    input_path = Path(input_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load triples ──
    print(f"Loading triples from {input_path}...")
    triples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))
    print(f"  Loaded {len(triples)} triples")

    # ── Load lexicon (optional) ──
    lexicon = set()
    if args.lexicon and Path(args.lexicon).exists():
        with open(args.lexicon) as f:
            lex_data = json.load(f)
        if isinstance(lex_data, list):
            for entry in lex_data:
                lexicon.add(normalize_entity(entry.get("term", "")))
                for alias in entry.get("aliases", []):
                    lexicon.add(normalize_entity(alias))
        elif isinstance(lex_data, dict):
            for term in lex_data:
                lexicon.add(normalize_entity(term))
        print(f"  Loaded lexicon: {len(lexicon)} terms")
    else:
        # Build minimal lexicon from known terms
        lexicon = (
            {normalize_entity(d) for d in KNOWN_DESCRIPTORS} |
            {normalize_entity(s) for s in KNOWN_SETTINGS} |
            {
                "mass transport deposit", "mtd", "turbidite", "debris flow",
                "slide", "slump", "hemipelagite", "pelagite",
                "turbidity current", "slope failure", "submarine landslide",
                "earthquake", "pore pressure", "sedimentation",
                "erosion", "deposition", "seafloor", "sediment",
                "continental slope", "continental shelf",
            }
        )
        print(f"  Using built-in lexicon: {len(lexicon)} terms")

    # ═══════════════════════════════════════════════════════════════════
    #  STEP 1: VERIFICATION FILTER
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n  Verification policy: {args.verif_policy}")

    passed_verif = []
    rejected = []
    verif_reasons = Counter()

    for t in triples:
        passes, reason = check_verification(t, args.verif_policy)
        if passes:
            passed_verif.append(t)
        else:
            t["_reject_reason"] = reason
            rejected.append(t)
            verif_reasons[reason] += 1

    print(f"  After verification filter: {len(passed_verif)} kept, "
          f"{len(rejected)} rejected")
    if verif_reasons:
        for reason, count in verif_reasons.most_common():
            print(f"    {reason:35s}: {count}")

    # ═══════════════════════════════════════════════════════════════════
    #  STEP 2: VALIDATION CHECKS
    # ═══════════════════════════════════════════════════════════════════

    cleaned = []
    validation_reasons = Counter()
    n_dupes = 0
    seen_keys = set()

    for t in passed_verif:
        # Basic checks
        ok, reason = check_basic(t)
        if not ok:
            t["_reject_reason"] = reason
            rejected.append(t)
            validation_reasons[reason] += 1
            continue

        # Relation check
        ok, reason = check_relation(t)
        if not ok:
            t["_reject_reason"] = reason
            rejected.append(t)
            validation_reasons[reason] += 1
            continue

        # Type constraint check
        ok, reason = check_type_constraint(t)
        if not ok:
            t["_reject_reason"] = reason
            rejected.append(t)
            validation_reasons[reason] += 1
            continue

        # Lexicon coverage check
        ok, reason = check_lexicon_coverage(t, lexicon)
        if not ok:
            t["_reject_reason"] = reason
            rejected.append(t)
            validation_reasons[reason] += 1
            continue

        # Deduplication
        key = triple_key(t)
        if key in seen_keys:
            n_dupes += 1
            continue
        seen_keys.add(key)

        cleaned.append(t)

    print(f"\n{'='*60}")
    print(f"  VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"  After verif:  {len(passed_verif)} | "
          f"Cleaned: {len(cleaned)} | "
          f"Rejected: {len([r for r in rejected if '_reject_reason' in r and not r['_reject_reason'].startswith('verif')])} | "
          f"Dupes: {n_dupes}")

    if validation_reasons:
        for reason, count in validation_reasons.most_common():
            print(f"    {reason:35s}: {count}")

    # Relation distribution
    rel_counts = Counter()
    for t in cleaned:
        rel_counts[t.get("relation_norm", t.get("relation", ""))] += 1
    print(f"  Relations:")
    for rel, count in rel_counts.most_common():
        print(f"    {rel:20s}: {count}")

    # LB Recall
    hits, total, missing = compute_lb_recall(cleaned)
    print(f"  LB Recall: {hits}/{total} = {hits/total:.1%}" if total > 0 else "")

    # Descriptor coverage
    found_desc, missing_desc = compute_descriptor_coverage(cleaned)
    print(f"  Desc Coverage: {len(found_desc)}/{len(LB_DESCRIPTORS)} "
          f"found={sorted(found_desc)}")
    print(f"  Missing: {sorted(missing_desc)}")

    # ═══════════════════════════════════════════════════════════════════
    #  STEP 3: CANONICALIZATION
    # ═══════════════════════════════════════════════════════════════════

    # Collect unique entities
    entities_before = set()
    for t in cleaned:
        s = normalize_entity(t.get("source_norm", t.get("source", "")))
        o = normalize_entity(t.get("target_norm", t.get("target", "")))
        entities_before.add(s)
        entities_before.add(o)

    entity_list = sorted(entities_before)
    print(f"\n  Unique entities: {len(entity_list)}")

    # Build canonical map via embedding clustering
    canonical_map = build_canonical_map(
        entity_list,
        distance_threshold=args.cluster_threshold,
        lb_descriptors=LB_DESCRIPTORS,
    )

    print(f"  Canonical map: {len(canonical_map)} merge rules")

    # Apply canonical map
    print(f"  Applying canonical map to {len(cleaned)} triples...")
    n_merged = apply_canonical_map(cleaned, canonical_map)
    print(f"  Entity occurrences merged: {n_merged}")

    # Re-deduplicate after canonicalization
    print(f"  Re-deduplicating after canonicalization...")
    final = []
    seen_keys2 = set()
    n_dupes2 = 0
    n_self_loops = 0

    for t in cleaned:
        s = normalize_entity(t.get("source_norm", t.get("source", "")))
        o = normalize_entity(t.get("target_norm", t.get("target", "")))

        # Check for self-loops created by canonicalization
        if s == o:
            n_self_loops += 1
            continue

        key = triple_key(t)
        if key in seen_keys2:
            n_dupes2 += 1
            continue
        seen_keys2.add(key)
        final.append(t)

    # Count entities after
    entities_after = set()
    for t in final:
        s = normalize_entity(t.get("source_norm", t.get("source", "")))
        o = normalize_entity(t.get("target_norm", t.get("target", "")))
        entities_after.add(s)
        entities_after.add(o)

    print(f"  Before: {len(cleaned)}")
    print(f"  After:  {len(final)}")
    print(f"  Duplicates removed: {n_dupes2}")

    # ═══════════════════════════════════════════════════════════════════
    #  WRITE OUTPUTS
    # ═══════════════════════════════════════════════════════════════════

    # Canonical triples
    out_triples = outdir / "canonical_triples_v5.jsonl"
    with open(out_triples, "w") as f:
        for t in final:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(final)} canonical triples to {out_triples}")

    # Canonical map
    out_map = outdir / "canonical_map_v5.json"
    with open(out_map, "w") as f:
        json.dump(canonical_map, f, indent=2, ensure_ascii=False)
    print(f"  Wrote canonical map to {out_map}")

    # Rejected triples (for debugging)
    out_rejected = outdir / "rejected_triples_v5.jsonl"
    with open(out_rejected, "w") as f:
        for t in rejected:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rejected)} rejected triples to {out_rejected}")

    # Stats
    # Recompute verification-aware hallucination rate
    verif_decided = 0
    verif_supported = 0
    verif_strong = 0
    for t in final:
        v = t.get("_verification", {}).get("verdict", "")
        if v in ("STRONG_SUPPORT", "WEAK_SUPPORT", "NOT_SUPPORTED"):
            verif_decided += 1
            if v in ("STRONG_SUPPORT", "WEAK_SUPPORT"):
                verif_supported += 1
            if v == "STRONG_SUPPORT":
                verif_strong += 1

    final_halluc_rate = (
        1 - (verif_supported / verif_decided) if verif_decided > 0 else 0.0
    )

    stats = {
        "input_triples":         len(triples),
        "after_verif_filter":    len(passed_verif),
        "verif_policy":          args.verif_policy,
        "verif_rejected":        dict(verif_reasons),
        "after_validation":      len(cleaned),
        "validation_rejected":   dict(validation_reasons),
        "duplicates_pass1":      n_dupes,
        "duplicates_pass2":      n_dupes2,
        "self_loops_removed":    n_self_loops,
        "entities_before":       len(entities_before),
        "entities_after":        len(entities_after),
        "merge_rules":           len(canonical_map),
        "output_triples":        len(final),
        "lb_recall":             f"{hits}/{total}",
        "lb_recall_pct":         round(hits / total, 4) if total > 0 else 0,
        "desc_coverage":         f"{len(found_desc)}/{len(LB_DESCRIPTORS)}",
        "desc_found":            sorted(found_desc),
        "desc_missing":          sorted(missing_desc),
        "final_halluc_rate":     round(final_halluc_rate, 4),
        "final_strong_support":  verif_strong,
        "final_supported":       verif_supported,
        "final_decided":         verif_decided,
    }

    out_stats = outdir / "cleaning_stats_v5.json"
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ═══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  CANONICALIZATION SUMMARY (v5)")
    print(f"{'='*60}")
    print(f"  Input triples:       {len(triples)}")
    print(f"  Verif policy:        {args.verif_policy}")
    print(f"  Verif rejected:      {sum(verif_reasons.values())}")
    print(f"  Validation rejected: {sum(validation_reasons.values())}")
    print(f"  Output triples:      {len(final)}")
    print(f"  Entities before:     {len(entities_before)}")
    print(f"  Entities after:      {len(entities_after)}")
    print(f"  Merge rules:         {len(canonical_map)}")
    print(f"  Duplicates removed:  {n_dupes + n_dupes2}")
    print(f"  Self-loops removed:  {n_self_loops}")
    print(f"  ---")
    print(f"  LB Recall:           {hits}/{total} = {hits/total:.1%}" if total > 0 else "")
    print(f"  Desc Coverage:       {len(found_desc)}/{len(LB_DESCRIPTORS)}")
    print(f"  Final halluc rate:   {final_halluc_rate:.1%}  "
          f"(S={verif_strong} W={verif_supported - verif_strong} "
          f"decided={verif_decided})")
    print(f"{'='*60}")
    print(f"  Triples:   {out_triples}")
    print(f"  Rejected:  {out_rejected}")
    print(f"  Stats:     {out_stats}")


if __name__ == "__main__":
    main()