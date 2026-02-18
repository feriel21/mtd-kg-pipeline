#!/usr/bin/env python3
"""
02b_verify_triples_v5.py — Triple Verification (GraphJudge-inspired, v5) — FIXED
=========================================================================

Fixes integrated:
1) Ollama connection auto-fallback to HuggingFace (optional) + clean error handling.
2) Backend errors are counted as BACKEND_ERROR (not UNPARSEABLE).
3) Multi-chunk verification supports _provenance.top_chunks (top-2) from extractor v4 fixed.
4) Relation normalization + filter: unknown relations -> SKIPPED_RELATION (no LLM call).
5) Correct stats: hallucination rate on DECIDED only (S/W/NS).

Outputs:
- verified_triples_v5.jsonl (triples with _verification)
- verification_audit_v5.jsonl (raw responses for manual review)
- verification_stats_v5.json (summary)

Usage:
  python -u 02b_verify_triples_v5.py \
    --input  output/step4/raw_triples_v4.jsonl \
    --output output/step4/verified_triples_v5.jsonl \
    --backend ollama \
    --model qwen2.5:7b-instruct \
    --ollama-url http://localhost:11434 \
    --hf-fallback-model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict


# ── Allowed relations + normalization ──────────────────────────────────

ALLOWED_RELATIONS = {
    "hasDescriptor", "occursIn", "formedBy", "partOf",
    "triggers", "causes", "controls", "affects",
    "overlies", "underlies", "indicates", "evidences", "relatedTo",
}

RELATION_MAP = {
    "hasdescriptor": "hasDescriptor",
    "occursin": "occursIn",
    "formedby": "formedBy",
    "partof": "partOf",
    "triggeredby": "triggers",
    "triggered_by": "triggers",
    "overlays": "overlies",
    "overlies": "overlies",
    "underlies": "underlies",
    "relatedto": "relatedTo",
    "related_to": "relatedTo",
    "related to": "relatedTo",
}

def normalize_relation(rel: str) -> str:
    if not rel:
        return ""
    raw = rel.strip()
    key = raw.lower().replace(" ", "").replace("_", "").replace("-", "")
    for k, v in RELATION_MAP.items():
        kk = k.lower().replace(" ", "").replace("_", "").replace("-", "")
        if key == kk:
            return v
    return raw


# ── Relation glosses ───────────────────────────────────────────────────

RELATION_GLOSSES = {
    "hasDescriptor":  "Subject is characterised by / exhibits Object",
    "occursIn":       "Subject is found in / located in Object",
    "formedBy":       "Subject is formed by / produced by Object",
    "partOf":         "Subject is a part / component of Object",
    "triggers":       "Subject initiates / triggers Object",
    "causes":         "Subject directly produces Object",
    "controls":       "Subject governs / regulates Object",
    "affects":        "Subject influences / modifies Object",
    "overlies":       "Subject is stratigraphically above Object",
    "underlies":      "Subject is stratigraphically below Object",
    "indicates":      "Subject indicates Object",
    "evidences":      "Subject provides evidence for Object",
    "relatedTo":      "Subject is related to Object",
}


# ── Prompts ────────────────────────────────────────────────────────────

COT_VERIFY_PROMPT = """\
You are a geological fact-checker. Verify whether a claimed knowledge-graph triple
is supported by the provided source text passage.

=== SOURCE TEXT ===
{chunk_text}
=== END SOURCE TEXT ===

=== CLAIMED TRIPLE ===
  Subject:  {source}
  Relation: {relation} (meaning: {relation_gloss})
  Object:   {target}
=== END TRIPLE ===

Follow steps:

STEP 1 — EVIDENCE: Copy the most relevant sentence(s) from the source text
that relate to this triple. If none, write "NO EVIDENCE FOUND".

STEP 2 — REASONING: In 1-2 sentences, explain whether the evidence supports
the triple. Use ONLY the source text (no external knowledge).

STEP 3 — VERDICT: Choose exactly one:
  STRONG_SUPPORT   — text explicitly states it
  WEAK_SUPPORT     — text implies it / direct inference
  NOT_SUPPORTED    — not stated or not implied

Format EXACTLY:
EVIDENCE: <...>
REASONING: <...>
VERDICT: <STRONG_SUPPORT or WEAK_SUPPORT or NOT_SUPPORTED>
"""

SIMPLE_VERIFY_PROMPT = """\
Text: {chunk_text}

Does this text state or clearly imply that "{source}" {relation_gloss} "{target}"?

Answer with ONLY one of:
STRONG_SUPPORT / WEAK_SUPPORT / NOT_SUPPORTED
"""


# ── LLM backends ───────────────────────────────────────────────────────

def _make_ollama_fn(model: str, base_url: str) -> Callable[[str, str], str]:
    import requests

    def _call(system: str, user: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 500,
            },
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    return _call


def _make_hf_fn(model: str) -> Callable[[str, str], str]:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"  [HF] Loading model {model} ...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("  [HF] Model loaded.")

    def _call(system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(hf_model.device)
        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    return _call


def make_llm_fn(
    backend: str,
    model: str,
    base_url: str,
    hf_fallback_model: str = "",
    allow_fallback: bool = True,
) -> Tuple[Callable[[str, str], str], str]:
    """
    Returns (llm_fn, backend_used).
    If backend=ollama and it fails to connect and allow_fallback=True -> fallback to HF.
    """
    if backend == "hf":
        return _make_hf_fn(model), "hf"

    if backend != "ollama":
        raise ValueError("backend must be 'ollama' or 'hf'")

    # Try quick connection check by calling /api/tags (fast)
    try:
        import requests
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        r.raise_for_status()
    except Exception as e:
        if allow_fallback and hf_fallback_model:
            print(f"  [WARN] Ollama not reachable at {base_url}: {e}")
            print(f"  [WARN] Falling back to HF model: {hf_fallback_model}")
            return _make_hf_fn(hf_fallback_model), "hf"
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}. "
            "Run: ollama serve  (or use --backend hf)."
        )

    return _make_ollama_fn(model, base_url), "ollama"


# ── Response parsing ─────────────────────────────────────────────────

def parse_cot_response(response: str) -> dict:
    result = {
        "evidence": "",
        "reasoning": "",
        "verdict": "UNPARSEABLE",
        "raw_response": response,
    }

    ev = re.search(r"EVIDENCE:\s*(.+?)(?=\nREASONING:|\nVERDICT:|\Z)", response, re.DOTALL | re.IGNORECASE)
    if ev:
        result["evidence"] = ev.group(1).strip()

    rs = re.search(r"REASONING:\s*(.+?)(?=\nVERDICT:|\Z)", response, re.DOTALL | re.IGNORECASE)
    if rs:
        result["reasoning"] = rs.group(1).strip()

    vd = re.search(r"VERDICT:\s*(STRONG_SUPPORT|WEAK_SUPPORT|NOT_SUPPORTED)", response, re.IGNORECASE)
    if vd:
        result["verdict"] = vd.group(1).upper()
        return result

    up = response.upper()
    if "STRONG_SUPPORT" in up:
        result["verdict"] = "STRONG_SUPPORT"
    elif "WEAK_SUPPORT" in up:
        result["verdict"] = "WEAK_SUPPORT"
    elif "NOT_SUPPORTED" in up or "NOT SUPPORTED" in up:
        result["verdict"] = "NOT_SUPPORTED"

    return result


# ── Chunk selection (v4 extractor compatible) ─────────────────────────

def get_chunk_texts(triple: dict, max_chunks: int = 2) -> List[str]:
    prov = triple.get("_provenance", {}) or {}
    texts: List[str] = []

    # Preferred: new extractor saves top_chunks[]
    top_chunks = prov.get("top_chunks", None)
    if isinstance(top_chunks, list) and top_chunks:
        for ch in top_chunks[:max_chunks]:
            t = (ch or {}).get("text", "") or ""
            if t and t not in texts:
                texts.append(t)
        if texts:
            return texts

    # Fallback: older fields
    best = prov.get("best_chunk_text", "") or ""
    if best:
        texts.append(best)

    second = prov.get("second_chunk_text", "") or ""
    if second and second != best:
        texts.append(second)

    return texts[:max_chunks]


# ── Verification core ────────────────────────────────────────────────

def verify_single_triple(
    triple: dict,
    llm_fn: Callable[[str, str], str],
    max_retries: int = 1,
    chunk_chars: int = 1500,
) -> dict:

    # Relation normalize + filter
    rel_raw = triple.get("relation_norm", triple.get("relation", ""))
    rel = normalize_relation(str(rel_raw))
    if rel not in ALLOWED_RELATIONS:
        return {
            "verdict": "SKIPPED_RELATION",
            "evidence": "",
            "reasoning": f"Relation '{rel}' is not in ALLOWED_RELATIONS.",
            "raw_response": "",
            "chunks_checked": 0,
        }

    chunk_texts = get_chunk_texts(triple, max_chunks=2)
    if not chunk_texts:
        return {
            "verdict": "NO_CHUNK",
            "evidence": "",
            "reasoning": "No source chunk text found in provenance.",
            "raw_response": "",
            "chunks_checked": 0,
        }

    source = triple.get("source_norm", triple.get("source", ""))
    target = triple.get("target_norm", triple.get("target", ""))
    gloss = RELATION_GLOSSES.get(rel, f"Subject {rel} Object")

    verdict_rank = {"STRONG_SUPPORT": 3, "WEAK_SUPPORT": 2, "NOT_SUPPORTED": 1, "UNPARSEABLE": 0}
    best = None
    best_score = -1

    system_prompt = (
        "You are a precise geological fact-checker. "
        "Use ONLY the provided source text. Follow formatting exactly."
    )

    for chunk_text in chunk_texts:
        user_prompt = COT_VERIFY_PROMPT.format(
            chunk_text=(chunk_text or "")[:chunk_chars],
            source=source,
            relation=rel,
            relation_gloss=gloss,
            target=target,
        )

        try:
            resp = llm_fn(system_prompt, user_prompt)
        except Exception as e:
            return {
                "verdict": "BACKEND_ERROR",
                "evidence": "",
                "reasoning": f"Backend error: {e}",
                "raw_response": "",
                "chunks_checked": 0,
            }

        parsed = parse_cot_response(resp)

        if parsed["verdict"] == "UNPARSEABLE" and max_retries > 0:
            simple = SIMPLE_VERIFY_PROMPT.format(
                chunk_text=(chunk_text or "")[:1200],
                source=source,
                relation_gloss=gloss.lower(),
                target=target,
            )
            try:
                resp2 = llm_fn(system_prompt, simple)
                parsed2 = parse_cot_response(resp2)
                if parsed2["verdict"] != "UNPARSEABLE":
                    parsed = parsed2
                    parsed["raw_response"] = f"[RETRY]\n{resp2}"
            except Exception as e:
                return {
                    "verdict": "BACKEND_ERROR",
                    "evidence": "",
                    "reasoning": f"Backend error on retry: {e}",
                    "raw_response": "",
                    "chunks_checked": 0,
                }

        score = verdict_rank.get(parsed["verdict"], 0)
        if score > best_score:
            best_score = score
            best = parsed

    if best is None:
        return {
            "verdict": "UNPARSEABLE",
            "evidence": "",
            "reasoning": "No parseable response.",
            "raw_response": "",
            "chunks_checked": len(chunk_texts),
        }

    best["chunks_checked"] = len(chunk_texts)
    return best


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Triple Verification v5 — FIXED")
    parser.add_argument("--input", required=True, help="Input JSONL (raw triples)")
    parser.add_argument("--output", required=True, help="Output JSONL (with _verification)")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "hf"])
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--hf-fallback-model", default="", help="HF model to fallback to if Ollama not reachable")
    parser.add_argument("--no-fallback", action="store_true", help="Disable ollama->hf fallback")
    parser.add_argument("--skip-verified", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-chars", type=int, default=1500)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    triples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))

    print(f"  Loaded {len(triples)} triples from {input_path}")
    if args.limit > 0:
        triples = triples[:args.limit]
        print(f"  Limited to first {args.limit} triples")

    # Init LLM (with optional fallback)
    print(f"  Requested backend: {args.backend}  model: {args.model}")
    llm_fn, backend_used = make_llm_fn(
        backend=args.backend,
        model=args.model,
        base_url=args.ollama_url,
        hf_fallback_model=args.hf_fallback_model,
        allow_fallback=(not args.no_fallback),
    )
    print(f"  Using backend: {backend_used}")

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_path = output_dir / "verification_audit_v5.jsonl"
    stats_path = output_dir / "verification_stats_v5.json"

    counts = {
        "STRONG_SUPPORT": 0,
        "WEAK_SUPPORT": 0,
        "NOT_SUPPORTED": 0,
        "NO_CHUNK": 0,
        "UNPARSEABLE": 0,
        "BACKEND_ERROR": 0,
        "SKIPPED_RELATION": 0,
        "skipped": 0,
    }

    start = time.time()

    with open(args.output, "w", encoding="utf-8") as f_out, open(audit_path, "w", encoding="utf-8") as f_audit:
        for i, triple in enumerate(triples):
            if args.skip_verified and "_verification" in triple:
                counts["skipped"] += 1
                f_out.write(json.dumps(triple, ensure_ascii=False) + "\n")
                continue

            verification = verify_single_triple(
                triple=triple,
                llm_fn=llm_fn,
                max_retries=1,
                chunk_chars=args.chunk_chars,
            )

            verdict = verification.get("verdict", "UNPARSEABLE")
            counts[verdict] = counts.get(verdict, 0) + 1

            rel_raw = triple.get("relation_norm", triple.get("relation", ""))
            rel_norm = normalize_relation(str(rel_raw))

            triple["_verification"] = {
                "verdict": verdict,
                "evidence": verification.get("evidence", ""),
                "reasoning": verification.get("reasoning", ""),
                "chunks_checked": verification.get("chunks_checked", 0),
                "backend": backend_used,
                "model": (args.hf_fallback_model if (backend_used == "hf" and args.backend == "ollama") else args.model),
                "version": "v5_cot_fixed",
                "relation_norm": rel_norm,
            }

            f_out.write(json.dumps(triple, ensure_ascii=False) + "\n")

            audit_entry = {
                "index": i,
                "source": triple.get("source_norm", triple.get("source", "")),
                "relation": rel_norm,
                "target": triple.get("target_norm", triple.get("target", "")),
                "verdict": verdict,
                "evidence": verification.get("evidence", ""),
                "reasoning": verification.get("reasoning", ""),
                "raw_response": verification.get("raw_response", ""),
            }
            f_audit.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")

            if (i + 1) % 25 == 0 or (i + 1) == len(triples):
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"  [{i+1}/{len(triples)}] "
                    f"S={counts['STRONG_SUPPORT']} "
                    f"W={counts['WEAK_SUPPORT']} "
                    f"NS={counts['NOT_SUPPORTED']} "
                    f"NC={counts['NO_CHUNK']} "
                    f"U={counts['UNPARSEABLE']} "
                    f"BE={counts['BACKEND_ERROR']} "
                    f"SK={counts['SKIPPED_RELATION']} "
                    f"({rate:.1f} triples/s)"
                )

    elapsed = time.time() - start

    decided = counts["STRONG_SUPPORT"] + counts["WEAK_SUPPORT"] + counts["NOT_SUPPORTED"]
    supported = counts["STRONG_SUPPORT"] + counts["WEAK_SUPPORT"]

    halluc = (counts["NOT_SUPPORTED"] / decided) if decided > 0 else None
    strong_rate = (counts["STRONG_SUPPORT"] / decided) if decided > 0 else None

    stats = {
        "input_file": str(input_path),
        "output_file": str(args.output),
        "backend_used": backend_used,
        "requested_backend": args.backend,
        "model_requested": args.model,
        "model_used": (args.hf_fallback_model if (backend_used == "hf" and args.backend == "ollama") else args.model),
        "total_triples": len(triples),
        "counts": counts,
        "decided_total": decided,
        "supported_total": supported,
        "hallucination_rate_decided": halluc,
        "strong_support_rate_decided": strong_rate,
        "elapsed_seconds": round(elapsed, 1),
        "triples_per_second": round(len(triples) / elapsed, 2) if elapsed > 0 else 0,
        "chunk_chars": args.chunk_chars,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("  VERIFICATION SUMMARY (v5 FIXED)")
    print(f"{'='*60}")
    print(f"  Total triples:      {len(triples)}")
    print(f"  STRONG_SUPPORT:     {counts['STRONG_SUPPORT']}")
    print(f"  WEAK_SUPPORT:       {counts['WEAK_SUPPORT']}")
    print(f"  NOT_SUPPORTED:      {counts['NOT_SUPPORTED']}")
    print(f"  NO_CHUNK:           {counts['NO_CHUNK']}")
    print(f"  UNPARSEABLE:        {counts['UNPARSEABLE']}")
    print(f"  BACKEND_ERROR:      {counts['BACKEND_ERROR']}")
    print(f"  SKIPPED_RELATION:   {counts['SKIPPED_RELATION']}")
    print(f"  Skipped existing:   {counts['skipped']}")
    print("  ---")
    if halluc is None:
        print("  Hallucination rate: — (no decided triples)")
        print("  Strong support:     —")
    else:
        print(f"  Hallucination rate: {halluc:.1%} (on decided)")
        print(f"  Strong support:     {strong_rate:.1%} (on decided)")
    print(f"  Elapsed:            {elapsed:.0f}s  ({len(triples)/elapsed:.1f} triples/s)" if elapsed > 0 else "")
    print(f"{'='*60}")
    print(f"  Output:  {args.output}")
    print(f"  Audit:   {audit_path}")
    print(f"  Stats:   {stats_path}")


if __name__ == "__main__":
    main()
