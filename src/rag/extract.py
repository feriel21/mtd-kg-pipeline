"""
rag/extract.py — REWRITE for small LLMs (1.5B–7B)
══════════════════════════════════════════════════

Changes:
  1. Simpler system prompt (examples > lists)
  2. Adaptive context length (1000 for 1.5B, 2000 for 7B+)
  3. 4-level fallback JSON parsing
  4. Logs parse failures with raw output for debugging
"""

import json
import re
from typing import List, Dict, Any, Optional
from rag.llm_hf import hf_chat

# ── System prompt: 2 examples teach the format better than listing all types ──
SYSTEM = """You extract geological knowledge as JSON triples from scientific text.

Example:
{"triples":[
{"source":"debris flow","source_type":"Process","relation":"CAUSES","target":"slope failure","target_type":"Process"},
{"source":"MTD","source_type":"Geological_Object","relation":"HAS_DESCRIPTOR","target":"chaotic reflections","target_type":"Descriptor"},
{"source":"turbidite","source_type":"Geological_Object","relation":"OCCURS_IN","target":"channel system","target_type":"Environmental_Control"}
]}

Node types: Geological_Object, Descriptor, Process, Environmental_Control, Evidence
Relations: CAUSES, AFFECTS, TRIGGERS, CONTROLS, CONDITIONS, INDICATES, SUGGESTS, EVIDENCES, PART_OF, HAS_DESCRIPTOR, HAS_PROPERTY, HAS_PHASE, LOCATED_IN, RELATED_TO

Rules:
- Extract ONLY facts explicitly stated in the text below.
- Return ONLY valid JSON. No explanation, no markdown.
- source and target must be short entity names (not sentences).
- If nothing to extract: {"triples":[]}
"""


def _extract_json_robust(raw: str) -> Optional[dict]:
    """
    4-level fallback to extract JSON from messy LLM output.
    
    Level 1: Direct json.loads on cleaned text
    Level 2: Brace-matching to find {"triples": ...}
    Level 3: Fix common JSON errors (trailing commas, unclosed brackets)
    Level 4: Regex extraction of individual triple objects
    """
    raw = raw.strip()
    # Remove markdown code fences
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)

    # Level 1: direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "triples" in obj:
            return obj
        if isinstance(obj, list):
            return {"triples": obj}
    except json.JSONDecodeError:
        pass

    # Level 2: brace matching starting from {"triples"
    for start_pattern in ['{"triples"', '"triples"']:
        start = raw.find(start_pattern)
        if start != -1:
            # Find the opening brace
            if raw[start] != '{':
                left = raw.rfind("{", 0, start)
                if left != -1:
                    start = left
                else:
                    continue

            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(raw)):
                ch = raw[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(raw[start:i+1])
                                if isinstance(obj, dict):
                                    return obj
                            except json.JSONDecodeError:
                                break

    # Level 3: fix common JSON errors
    cleaned = raw
    # Strip any leading non-JSON text
    first_brace = cleaned.find("{")
    if first_brace > 0:
        cleaned = cleaned[first_brace:]
    # Fix trailing commas
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    # Close unclosed brackets
    open_b = cleaned.count("{") - cleaned.count("}")
    open_sq = cleaned.count("[") - cleaned.count("]")
    cleaned = cleaned + "]" * max(0, open_sq) + "}" * max(0, open_b)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Level 4: regex extraction of individual triple-like objects
    pattern = r'\{\s*"source"\s*:\s*"[^"]+"\s*,\s*"source_type"\s*:\s*"[^"]*"\s*,\s*"relation"\s*:\s*"[^"]+"\s*,\s*"target"\s*:\s*"[^"]+"\s*,\s*"target_type"\s*:\s*"[^"]*"[^}]*\}'
    matches = re.findall(pattern, raw)
    if matches:
        triples = []
        for m in matches:
            try:
                t = json.loads(m)
                if t.get("source") and t.get("target"):
                    triples.append(t)
            except json.JSONDecodeError:
                continue
        if triples:
            return {"triples": triples}

    # Also try simpler pattern (just source + target, any order of fields)
    pattern2 = r'\{[^{}]*"source"\s*:\s*"[^"]*"[^{}]*"target"\s*:\s*"[^"]*"[^{}]*\}'
    matches2 = re.findall(pattern2, raw)
    if matches2:
        triples = []
        for m in matches2:
            try:
                t = json.loads(m)
                if t.get("source") and t.get("target"):
                    triples.append(t)
            except json.JSONDecodeError:
                continue
        if triples:
            return {"triples": triples}

    return None


def extract_triples_with_context(
    model: str,
    context_text: str,
    allowed_node_types: List[str],
    allowed_relations: List[str],
    lexicon_preview: List[str],
) -> Dict[str, Any]:
    """
    Extract geological triples from context using LLM.
    """
    # Adaptive context length based on model size
    if any(small in model for small in ["0.5B", "1.5B", "1B", "3B"]):
        max_ctx = 1000
    else:
        max_ctx = 2000

    ctx_truncated = context_text[:max_ctx]

    user = f"""TEXT:
{ctx_truncated}

Extract geological relations from this text as JSON triples.
Return ONLY the JSON object with "triples" key."""

    raw = hf_chat(
        model_name=model,
        system=SYSTEM,
        user=user,
        max_new_tokens=450,
        temperature=0.1,  # more deterministic for extraction
        top_p=0.85,
    )

    # Debug logging
    from pathlib import Path
    Path("output/step2").mkdir(parents=True, exist_ok=True)
    with open("output/step2/llm_raw_debug.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"QUERY CONTEXT ({len(ctx_truncated)} chars):\n")
        f.write(ctx_truncated[:500] + "\n...\n")
        f.write(f"RAW LLM OUTPUT ({len(raw)} chars):\n")
        f.write(raw[:2000] + "\n")

    # Parse with robust extraction
    result = _extract_json_robust(raw)

    if result and isinstance(result.get("triples"), list):
        n = len(result["triples"])
        if n > 0:
            with open("output/step2/llm_raw_debug.txt", "a") as f:
                f.write(f"→ PARSED {n} triples\n")
        return result

    # Log failure
    with open("output/step2/llm_raw_debug.txt", "a") as f:
        f.write(f"→ PARSE FAILED — raw output was:\n{raw[:1000]}\n")

    return {"triples": []}