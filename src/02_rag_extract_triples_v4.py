#!/usr/bin/env python3
"""
Run 3 — RAG-based triple extraction with ALL improvements.

v4 Changes:
A) Add SciBERT reranking after BM25 (semantic rerank)
B) Add gap-filling queries (second pass) for missing LB edges (optional)
C) Expand query volume strategically
D) Dual temperature extraction (0.1 + 0.3)
E) Save chunk text in provenance (best chunk id/text + combined score + temperature)
F) Log computational cost (wall clock + token in/out)

FIXES applied:
- dtype instead of torch_dtype (transformers >= 4.37)
- Shard-aware default output filename
"""

import json
import argparse
import os
import re
import sys
import time
from pathlib import Path

# NEW (A) — SciBERT reranker
from sentence_transformers import SentenceTransformer

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            _reranker = SentenceTransformer("allenai/scibert_scivocab_uncased")
        except Exception as e:
            raise RuntimeError(
                "Failed to load SciBERT reranker via SentenceTransformer. "
                f"Original error: {e}"
            )
    return _reranker


# ── Configuration ─────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("KG_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TOP_K = 5
MAX_TOKENS = 1024

# ── Le Bouteiller (2019) Reference Data ───────────────────────────────────
LB_DESCRIPTORS = [
    "chaotic", "transparent", "blocky", "massive", "hummocky",
    "discontinuous", "high-amplitude", "low-amplitude", "undeformed",
    "layered", "stratified", "continuous", "parallel"
]

LB_PROPERTIES = [
    "morphology", "position", "basal surface", "upper surface",
    "internal facies", "headscarp", "global environment"
]

LB_OBJECTS = [
    "mass transport deposit", "mass transport complex", "debris flow",
    "turbidite", "submarine landslide", "slide", "slump",
    "debris flow deposit", "turbidity current", "channel levee",
    "levee", "canyon", "submarine fan", "slope failure"
]

LB_PROCESSES = [
    "debris flow", "turbidity current", "slope failure", "retrogressive failure",
    "erosion", "remobilization", "deposition", "sedimentation",
    "compaction", "fluid migration", "fluid overpressure",
    "seismic loading", "gas hydrate dissociation", "wave loading",
    "rapid sedimentation", "tectonic activity"
]

# ── Prompt Templates ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a geological knowledge extraction system specialized in 
mass-transport deposits (MTDs) and submarine sedimentary processes. You extract 
structured knowledge from scientific text as JSON triples.

CRITICAL RULES:
- NEVER extract author names, figure/table references, or citation markers
- NEVER extract case-study specific labels (e.g., "MTD A", "Zone B", "Unit 3")
- NEVER use single abbreviations alone (e.g., "BS", "US") — expand them
- NEVER create self-loops (source == target)
- Use SPECIFIC geological terms, not generic words like "material" or "data"
- Every triple must represent a GENERAL geological fact, not a case-specific observation
"""

PROMPT_CAUSAL = """Extract CAUSAL geological relationships from this text.
Focus on: what triggers, causes, controls, or affects what?

ALLOWED RELATIONS (pick the MOST SPECIFIC one):
- triggers: X initiates/triggers/activates Y (sudden onset)
- causes: X directly produces/results in Y  
- controls: X governs/regulates/determines Y (ongoing influence)
- affects: X influences/modifies/impacts Y (general effect)

NEVER use "relatedTo" — always choose one of the four above.

FEW-SHOT EXAMPLES:
Text: "Pore-fluid overpressure triggers slope failure on continental margins"
→ {{"source": "fluid overpressure", "relation": "triggers", "target": "slope failure",
    "source_type": "Process", "target_type": "Process"}}

Text: "Burial compaction controls the thickness of mass transport deposits"
→ {{"source": "burial compaction", "relation": "controls", "target": "thickness",
    "source_type": "Process", "target_type": "Descriptor"}}

Text: "Retrogressive failure produces hummocky upper surfaces"
→ {{"source": "retrogressive failure", "relation": "causes", "target": "hummocky surface",
    "source_type": "Process", "target_type": "Descriptor"}}

Text: "Flow behaviour impacts internal facies distribution within MTDs"
→ {{"source": "flow behavior", "relation": "affects", "target": "internal facies",
    "source_type": "Process", "target_type": "Descriptor"}}

Now extract from this text (return ONLY JSON array, no other text):
---
{context}
---
Query focus: {query}

Return a JSON array of objects with keys: source, relation, target, source_type, target_type.
Valid types: SeismicObject, Process, Descriptor, Setting, Evidence.
"""

PROMPT_DESCRIPTOR = """Extract DESCRIPTOR relationships from this text.
Focus on: what seismic/geological properties characterize each object?

RELATION TO USE: hasDescriptor
This means: Object X is characterized by / shows / exhibits property Y

SEISMIC DESCRIPTORS TO LOOK FOR:
chaotic, transparent, blocky, massive, hummocky, discontinuous,
high-amplitude, low-amplitude, undeformed, layered, stratified,
continuous, parallel, erosional, irregular, lobate, elongated,
tabular, ponded, arcuate, stepped, deformed, disrupted

FEW-SHOT EXAMPLES:
Text: "MTDs typically show chaotic to transparent seismic facies"
→ [{{"source": "mass transport deposit", "relation": "hasDescriptor", "target": "chaotic", 
     "source_type": "SeismicObject", "target_type": "Descriptor"}},
    {{"source": "mass transport deposit", "relation": "hasDescriptor", "target": "transparent",
     "source_type": "SeismicObject", "target_type": "Descriptor"}}]

Text: "The basal surface is characterized by high-amplitude, irregular reflectors"
→ [{{"source": "basal surface", "relation": "hasDescriptor", "target": "high-amplitude",
     "source_type": "SeismicObject", "target_type": "Descriptor"}},
    {{"source": "basal surface", "relation": "hasDescriptor", "target": "irregular",
     "source_type": "SeismicObject", "target_type": "Descriptor"}}]

Text: "Turbidites display layered, parallel, continuous reflections"
→ [{{"source": "turbidite", "relation": "hasDescriptor", "target": "layered",
     "source_type": "SeismicObject", "target_type": "Descriptor"}},
    {{"source": "turbidite", "relation": "hasDescriptor", "target": "parallel",
     "source_type": "SeismicObject", "target_type": "Descriptor"}},
    {{"source": "turbidite", "relation": "hasDescriptor", "target": "continuous",
     "source_type": "SeismicObject", "target_type": "Descriptor"}}]

Now extract from this text (return ONLY JSON array, no other text):
---
{context}
---
Query focus: {query}

Return a JSON array. ONLY use relation "hasDescriptor". 
Valid source_type: SeismicObject. Valid target_type: Descriptor.
"""

PROMPT_CONTEXT = """Extract CONTEXTUAL geological relationships from this text.
Focus on: where things occur, what they are part of, and what forms them.

ALLOWED RELATIONS:
- occursIn: X is found in / located in / occurs in environment Y
- partOf: X is a component/part/zone of Y
- formedBy: X is formed by / produced by / deposited by process Y
- overlies: X is stratigraphically above Y
- underlies: X is stratigraphically below Y

NEVER use "relatedTo" — always choose one of the above.

FEW-SHOT EXAMPLES:
Text: "Debris flows are common on continental slopes and margins"
→ {{"source": "debris flow", "relation": "occursIn", "target": "continental slope",
    "source_type": "Process", "target_type": "Setting"}}

Text: "The headscarp is the upslope part of the mass transport deposit"
→ {{"source": "headscarp", "relation": "partOf", "target": "mass transport deposit",
    "source_type": "SeismicObject", "target_type": "SeismicObject"}}

Text: "Turbidites are deposited by turbidity currents"
→ {{"source": "turbidite", "relation": "formedBy", "target": "turbidity current",
    "source_type": "SeismicObject", "target_type": "Process"}}

Now extract from this text (return ONLY JSON array, no other text):
---
{context}
---
Query focus: {query}

Return a JSON array of objects with keys: source, relation, target, source_type, target_type.
"""


# ── Query Generation ──────────────────────────────────────────────────────

def generate_queries():
    queries = []

    # TYPE A: Causal queries
    causal_pairs = [
        ("fluid overpressure", "slope failure"),
        ("seismic loading", "slope failure"),
        ("gas hydrate dissociation", "slope instability"),
        ("rapid sedimentation", "fluid overpressure"),
        ("wave loading", "sediment remobilization"),
        ("tectonic activity", "slope instability"),
        ("debris flow", "erosion"),
        ("turbidity current", "deposition"),
        ("retrogressive failure", "headscarp"),
        ("flow behavior", "internal facies"),
        ("sedimentation rate", "thickness"),
        ("topographic confinement", "morphology"),
        ("material heterogeneity", "flow behavior"),
        ("burial compaction", "thickness"),
        ("frontal compression", "upper surface"),
        ("basal erosion", "basal surface"),
        ("remobilization", "transparent facies"),
        ("grain heterogeneity", "flow behavior"),
        ("slope gradient", "flow velocity"),
        ("fluid migration", "acoustic property"),
    ]
    for src, tgt in causal_pairs:
        queries.append({
            "query": f"How does {src} affect or cause {tgt} in mass transport deposits?",
            "strategy": "causal",
            "focus": f"{src} → {tgt}"
        })

    lb_causal_edges = [
        ("flow behaviour", "internal facies distribution"),
        ("pre-existing geomorphologic pathways", "flow direction"),
        ("sedimentation rate", "slope instability"),
        ("topographic confinement", "deposit morphology"),
        ("material heterogeneity", "flow behavior"),
        ("fluid overpressure", "slope failure"),
        ("seismic shaking", "slope failure"),
        ("rapid burial", "overpressure"),
        ("erosion of substrate", "basal surface character"),
        ("post-deposition compaction", "thickness reduction"),
    ]
    for src, tgt in lb_causal_edges:
        queries.append({
            "query": f"What is the relationship between {src} and {tgt} in MTDs?",
            "strategy": "causal",
            "focus": f"LB2019: {src} → {tgt}"
        })

    # TYPE B: Descriptor queries
    for obj in ["mass transport deposit", "turbidite", "debris flow deposit",
                "submarine landslide", "slide", "slump", "channel levee"]:
        queries.append({
            "query": f"What are the seismic characteristics of {obj}? "
                     f"Consider: chaotic, transparent, blocky, massive, hummocky, "
                     f"discontinuous, high-amplitude, low-amplitude, layered, parallel",
            "strategy": "descriptor",
            "focus": f"descriptors of {obj}"
        })

    for prop in LB_PROPERTIES:
        queries.append({
            "query": f"Describe the seismic character of the {prop} of mass transport deposits. "
                     f"What reflector patterns, amplitudes, and continuity are observed?",
            "strategy": "descriptor",
            "focus": f"descriptors of MTD {prop}"
        })

    descriptor_combos = [
        ("chaotic", "transparent", "MTD internal facies"),
        ("blocky", "preserved blocks", "MTD intact rafted blocks"),
        ("hummocky", "irregular", "MTD upper surface"),
        ("erosional", "high-amplitude", "MTD basal surface"),
        ("layered", "continuous", "hemipelagic background"),
        ("discontinuous", "disrupted", "deformed strata"),
        ("massive", "transparent", "homogeneous flow deposit"),
    ]
    for d1, d2, context in descriptor_combos:
        queries.append({
            "query": f"In the context of {context}, what seismic facies show "
                     f"{d1} and {d2} characteristics?",
            "strategy": "descriptor",
            "focus": f"descriptors: {d1} + {d2}"
        })

    # Reverse descriptor queries
    for desc in ["chaotic", "transparent", "blocky", "massive", "hummocky",
                 "discontinuous", "layered", "parallel", "continuous",
                 "high-amplitude", "low-amplitude", "undeformed", "stratified"]:
        queries.append({
            "query": f"What geological objects or features are described as "
                     f"{desc} in seismic data? Which deposits show {desc} character?",
            "strategy": "descriptor",
            "focus": f"reverse descriptor: {desc}"
        })

    # Process chain queries
    process_chains = [
        "What happens after slope failure? What deposits form?",
        "How do debris flows transition into turbidity currents?",
        "What is the sequence from trigger to deposit in mass wasting?",
        "How does retrogressive failure propagate upslope?",
        "What processes create the basal surface of MTDs?",
        "What processes shape the upper surface of MTDs?",
        "How does flow transformation affect deposit character?",
    ]
    for q in process_chains:
        queries.append({"query": q, "strategy": "causal", "focus": "process chain"})

    # Explicit missing descriptor queries
    queries.append({
        "query": "What seismic features appear undeformed or undisturbed "
                 "adjacent to mass transport deposits?",
        "strategy": "descriptor",
        "focus": "GAP: undeformed descriptor"
    })
    queries.append({
        "query": "Where are stratified or well-stratified reflections "
                 "observed in relation to MTDs and turbidites?",
        "strategy": "descriptor",
        "focus": "GAP: stratified descriptor"
    })

    # TYPE C: Context queries
    for obj in LB_OBJECTS[:8]:
        queries.append({
            "query": f"Where does {obj} occur? In what geological setting or environment?",
            "strategy": "context",
            "focus": f"setting of {obj}"
        })

    component_queries = [
        "What are the main structural parts of a mass transport deposit? headscarp, body, toe",
        "What is the stratigraphic position of MTDs relative to other units?",
        "What overlies and underlies mass transport deposits?",
        "What are the components of a mass transport complex?",
        "Where do debris flows typically occur in a basin?",
        "What depositional environments are associated with turbidites?",
    ]
    for q in component_queries:
        queries.append({"query": q, "strategy": "context", "focus": "structural/stratigraphic context"})

    print(f"  Generated {len(queries)} queries:")
    print(f"    Causal:     {sum(1 for q in queries if q['strategy']=='causal')}")
    print(f"    Descriptor: {sum(1 for q in queries if q['strategy']=='descriptor')}")
    print(f"    Context:    {sum(1 for q in queries if q['strategy']=='context')}")

    return queries


# ── BM25 Retrieval ────────────────────────────────────────────────────────

def load_bm25_index(index_dir):
    import pickle
    with open(os.path.join(index_dir, "bm25_index.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    with open(os.path.join(index_dir, "chunks.jsonl"), "r") as f:
        chunks = [json.loads(line) for line in f]
    return bm25, chunks


def retrieve_chunks(bm25, chunks, query_text, top_k=TOP_K):
    tokenized_query = query_text.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "text": chunks[idx]["text"],
                "source_file": chunks[idx].get("source_file", "unknown"),
                "chunk_id": chunks[idx].get("chunk_id", idx),
                "score": float(scores[idx])
            })
    return results


def rerank_chunks(query_text, bm25_results, top_k=5):
    if not bm25_results:
        return bm25_results
    reranker = get_reranker()
    q_emb = reranker.encode(query_text)
    texts = [r["text"] for r in bm25_results]
    t_embs = reranker.encode(texts)
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity([q_emb], t_embs)[0]
    bm25_scores = [r["score"] for r in bm25_results]
    bm25_max = max(bm25_scores) if bm25_scores else 1.0
    bm25_norm = [s / bm25_max for s in bm25_scores]
    for i, r in enumerate(bm25_results):
        r["semantic_score"] = float(scores[i])
        r["combined_score"] = float(0.4 * bm25_norm[i] + 0.6 * scores[i])
    return sorted(bm25_results, key=lambda x: -x["combined_score"])[:top_k]


# ── LLM Extraction ────────────────────────────────────────────────────────

_model = None
_tokenizer = None
TOTAL_TOKENS_IN = 0
TOTAL_TOKENS_OUT = 0

def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"🔄 Loading model {MODEL_NAME}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if device == "cuda" else torch.float32

    # FIX: use dtype= instead of torch_dtype= (transformers >= 4.37)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dt,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        _model = _model.to(device)

    print(f"✅ Model loaded on {device}")
    return _model, _tokenizer


def extract_triples_llm(context, query, strategy, temperature=0.1):
    global TOTAL_TOKENS_IN, TOTAL_TOKENS_OUT
    model, tokenizer = load_model()

    if strategy == "causal":
        user_prompt = PROMPT_CAUSAL.format(context=context, query=query)
    elif strategy == "descriptor":
        user_prompt = PROMPT_DESCRIPTOR.format(context=context, query=query)
    elif strategy == "context":
        user_prompt = PROMPT_CONTEXT.format(context=context, query=query)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
    )

    in_tok = int(inputs.input_ids.shape[1])
    out_tok = int(outputs.shape[1] - inputs.input_ids.shape[1])
    TOTAL_TOKENS_IN += in_tok
    TOTAL_TOKENS_OUT += max(0, out_tok)

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def parse_llm_response(response_text):
    text = response_text.strip()
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        triples = json.loads(match.group())
        if isinstance(triples, list):
            return triples
    except json.JSONDecodeError:
        pass
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group())
            return [obj] if isinstance(obj, dict) else []
    except json.JSONDecodeError:
        pass
    return []


# ── Pre-validation ─────────────────────────────────────────────────────────

BLACKLIST_PATTERNS = [
    r'^[A-Z][a-z]+ et al\.?$',
    r'^[A-Z]\. [A-Z][a-z]+$',
    r'^(MTD|MTC|Unit)\s*[A-Z0-9]+$',
    r'^(Zone|Area|Region)\s*[A-Z0-9]+$',
    r'^[A-Z]{1,4}$',
    r'^(Fig|Figure|Table|Tab)\b',
    r'^\d+(\.\d+)?$',
    r'^(text|image|caption|paragraph)$',
]
BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS]

VAGUE_TERMS = {
    "material", "data", "model", "result", "study", "analysis",
    "method", "approach", "technique", "system", "type", "form",
    "case", "example", "observation", "information", "parameter",
    "value", "feature", "element", "component", "structure",
    "event", "condition", "factor", "effect", "property",
    "image", "figure", "table", "text", "section", "paper",
    "author", "reference", "citation", "abstract"
}


def pre_validate_triple(triple):
    s = triple.get("source", "").strip()
    t = triple.get("target", "").strip()
    r = triple.get("relation", "").strip()
    if not s or not t or not r:
        return False, "empty_field"
    if len(s) < 3 or len(t) < 3:
        return False, "too_short"
    if s.lower() == t.lower():
        return False, "self_loop"
    for pattern in BLACKLIST_RE:
        if pattern.match(s):
            return False, f"blacklisted_source: {s}"
        if pattern.match(t):
            return False, f"blacklisted_target: {t}"
    s_vague = s.lower() in VAGUE_TERMS
    t_vague = t.lower() in VAGUE_TERMS
    if s_vague and t_vague:
        return False, f"both_vague: {s} / {t}"
    return True, "ok"


# ── Gap filling ───────────────────────────────────────────────────────────

def load_lb_reference_edges(path):
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    edges = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                edges.append((str(item[0]), str(item[1]), str(item[2])))
            elif isinstance(item, dict):
                edges.append((str(item.get("source", "")), str(item.get("relation", "")), str(item.get("target", ""))))
    return [e for e in edges if e[0] and e[1] and e[2]]


def generate_gap_queries(cleaned_triples, lb_reference_edges):
    found_pairs = set()
    for t in cleaned_triples:
        s = t.get("source_norm", t.get("source", "")).lower()
        tgt = t.get("target_norm", t.get("target", "")).lower()
        found_pairs.add((s, tgt))
    gap_queries = []
    for s_ref, r_ref, t_ref in lb_reference_edges:
        s_ref_l = s_ref.lower()
        t_ref_l = t_ref.lower()
        matched = False
        for fs, ft in found_pairs:
            if (s_ref_l in fs or fs in s_ref_l) and (t_ref_l in ft or ft in t_ref_l):
                matched = True
                break
        if not matched:
            if r_ref in ("triggers", "causes", "controls", "affects"):
                strategy = "causal"
            elif r_ref == "hasDescriptor":
                strategy = "descriptor"
            else:
                strategy = "context"
            gap_queries.append({
                "query": f"Describe the relationship between {s_ref} and {t_ref} "
                         f"in mass transport deposits and turbidites.",
                "strategy": strategy,
                "focus": f"GAP-FILL: {s_ref} --[{r_ref}]--> {t_ref}"
            })
    return gap_queries


# ── Main Pipeline ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run 3 KG extraction (v4)")
    parser.add_argument("--index-dir", default="output/step1",
                        help="Directory with BM25 index and chunks")
    # FIX: shard-aware default output
    default_shard = os.environ.get('SHARD_ID', '0')
    default_nshards = os.environ.get('N_SHARDS', '1')
    if default_nshards == '1':
        default_output = "output/step4/raw_triples_v4.jsonl"
    else:
        default_output = f"output/step4/raw_triples_v4_shard{default_shard}.jsonl"

    parser.add_argument("--output", default=default_output,
                        help="Output JSONL file")
    parser.add_argument("--shard-id", type=int,
                        default=int(os.environ.get("SHARD_ID", "0")))
    parser.add_argument("--n-shards", type=int,
                        default=int(os.environ.get("N_SHARDS", "1")))
    parser.add_argument("--lb-edges", default=os.environ.get("KG_LB_EDGES", ""),
                        help="Optional JSON file with LB reference edges to gap-fill")
    args = parser.parse_args()

    start_time = time.time()

    print("Loading BM25 index...")
    bm25, chunks = load_bm25_index(args.index_dir)
    print(f"  {len(chunks)} chunks loaded")

    print("Generating queries...")
    all_queries = generate_queries()

    shard_queries = [q for i, q in enumerate(all_queries) if i % args.n_shards == args.shard_id]
    print(f"\n{'='*60}")
    print(f"  Run 4 KG Extraction — shard {args.shard_id}/{args.n_shards}")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Output: {args.output}")
    print(f"  Total queries: {len(all_queries)}")
    print(f"  This shard:    {len(shard_queries)} queries")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    stats = {
        "total_queries": 0,
        "total_raw_triples": 0,
        "total_accepted": 0,
        "total_rejected": 0,
        "reject_reasons": {},
        "by_strategy": {"causal": 0, "descriptor": 0, "context": 0},
        "by_relation": {},
        "relatedTo_count": 0,
        "gap_fill": {
            "enabled": False, "lb_edges_loaded": 0,
            "gap_queries_generated": 0, "gap_queries_processed": 0,
            "gap_new_triples_added": 0
        }
    }

    accepted_triples = []
    seen = set()

    def add_triple(triple_obj):
        s = (triple_obj.get("source", "") or "").strip().lower()
        r = (triple_obj.get("relation", "") or "").strip().lower()
        t = (triple_obj.get("target", "") or "").strip().lower()
        key = (s, r, t)
        if not s or not r or not t:
            return False
        if key in seen:
            return False
        seen.add(key)
        accepted_triples.append(triple_obj)
        return True

    def process_queries(query_list, is_gap_pass=False):
        nonlocal stats
        for qi, qinfo in enumerate(query_list):
            query_text = qinfo["query"]
            strategy = qinfo["strategy"]
            focus = qinfo.get("focus", "")
            stats["total_queries"] += 1
            if is_gap_pass:
                stats["gap_fill"]["gap_queries_processed"] += 1
            if (qi + 1) % 5 == 0 or qi == 0:
                tag = "GAP" if is_gap_pass else "RUN"
                print(f"  [{tag}] [{qi+1}/{len(query_list)}] [{strategy:10s}] {focus[:60]}...")

            retrieved_bm25 = retrieve_chunks(bm25, chunks, query_text, top_k=TOP_K * 2)
            if not retrieved_bm25:
                continue
            try:
                retrieved = rerank_chunks(query_text, retrieved_bm25, top_k=TOP_K)
            except Exception as e:
                print(f"    ⚠️ Reranker error (fallback to BM25): {e}")
                retrieved = retrieved_bm25[:TOP_K]
            if not retrieved:
                continue

            context = "\n---\n".join([r["text"] for r in retrieved])
            source_files = list(set(r["source_file"] for r in retrieved))

            for temp in [0.1, 0.3]:
                try:
                    response = extract_triples_llm(context, query_text, strategy, temperature=temp)
                    raw_triples = parse_llm_response(response)
                except Exception as e:
                    print(f"    ⚠️ LLM error: {e}")
                    continue

                stats["total_raw_triples"] += len(raw_triples)

                for triple in raw_triples:
                    is_valid, reason = pre_validate_triple(triple)
                    if not is_valid:
                        stats["total_rejected"] += 1
                        stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1
                        continue

                    rel = triple.get("relation", "")
                    if rel.lower() in ("relatedto", "related_to", "related to"):
                        stats["relatedTo_count"] += 1

                    triple["_provenance"] = {
                        "query": query_text,
                        "strategy": strategy,
                        "focus": focus,
                        "source_files": source_files,
                        "chunk_scores": [r.get("combined_score", r.get("score", 0.0)) for r in retrieved],
                        "best_chunk_id": retrieved[0].get("chunk_id"),
                        "best_chunk_text": (retrieved[0].get("text", "")[:500] if retrieved else ""),
                        "temperature": temp,
                    }

                    if add_triple(triple):
                        stats["total_accepted"] += 1
                        stats["by_strategy"][strategy] = stats["by_strategy"].get(strategy, 0) + 1
                        stats["by_relation"][rel] = stats["by_relation"].get(rel, 0) + 1
                        if is_gap_pass:
                            stats["gap_fill"]["gap_new_triples_added"] += 1

    process_queries(shard_queries, is_gap_pass=False)

    lb_edges = load_lb_reference_edges(args.lb_edges) if args.lb_edges else []
    if lb_edges:
        stats["gap_fill"]["enabled"] = True
        stats["gap_fill"]["lb_edges_loaded"] = len(lb_edges)
        gap_queries = generate_gap_queries(accepted_triples, lb_edges)
        stats["gap_fill"]["gap_queries_generated"] = len(gap_queries)
        gap_shard = [q for i, q in enumerate(gap_queries) if i % args.n_shards == args.shard_id]
        process_queries(gap_shard, is_gap_pass=True)

    with open(args.output, "w", encoding="utf-8") as fout:
        for t in accepted_triples:
            fout.write(json.dumps(t, ensure_ascii=False) + "\n")

    wall_clock = time.time() - start_time
    stats["wall_clock_seconds"] = wall_clock
    stats["total_tokens_input"] = TOTAL_TOKENS_IN
    stats["total_tokens_output"] = TOTAL_TOKENS_OUT

    print(f"\n{'='*60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Queries processed:  {stats['total_queries']}")
    print(f"  Raw triples:        {stats['total_raw_triples']}")
    print(f"  Pre-validated:      {stats['total_accepted']} accepted (deduped)")
    print(f"  Pre-rejected:       {stats['total_rejected']} rejected")
    print(f"  relatedTo count:    {stats['relatedTo_count']} "
          f"({100*stats['relatedTo_count']/max(1,stats['total_accepted']):.0f}%)")
    print(f"  Wall clock (s):     {stats['wall_clock_seconds']:.1f}")

    print(f"\n  By strategy:")
    for strat, count in stats["by_strategy"].items():
        print(f"    {strat:12s}: {count}")
    print(f"\n  By relation:")
    for rel, count in sorted(stats["by_relation"].items(), key=lambda x: -x[1]):
        print(f"    {rel:20s}: {count}")

    stats_path = args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to {stats_path}")


if __name__ == "__main__":
    main()