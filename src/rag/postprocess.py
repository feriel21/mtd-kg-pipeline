"""
rag/postprocess.py — REWRITE
═════════════════════════════

Changes:
  1. REMOVED _guess_source_from_context (invents false knowledge)
  2. Added _is_garbage_entity filter (rejects sentences-as-entities)
  3. Better relation mapping (case-insensitive, more coverage)
  4. Unknown relations → RELATED_TO fallback (instead of dropping)
  5. Empty source/target → REJECT (not guess)
"""

import re
from typing import Dict, Any, List

ALLOWED_NODE_TYPES = {
    "Geological_Object", "Descriptor", "Process",
    "Environmental_Control", "Evidence"
}

TYPE_MAP = {
    # processes
    "event": "Process",
    "flow": "Process",
    "mechanism": "Process",
    "trigger": "Process",
    # evidence
    "fact": "Evidence",
    "reference": "Evidence",
    "author": "Evidence",
    "person": "Evidence",
    "study": "Evidence",
    # descriptors
    "topic": "Descriptor",
    "property": "Descriptor",
    "feature": "Descriptor",
    "state": "Descriptor",
    "distance": "Descriptor",
    "node": "Descriptor",
    "attribute": "Descriptor",
    "characteristic": "Descriptor",
    # geological objects
    "rock": "Geological_Object",
    "material": "Geological_Object",
    "deposit": "Geological_Object",
    "structure": "Geological_Object",
    "system": "Geological_Object",
    "body": "Geological_Object",
    "unit": "Geological_Object",
    # environmental controls
    "environment": "Environmental_Control",
    "location": "Environmental_Control",
    "setting": "Environmental_Control",
    "region": "Environmental_Control",
    "area": "Environmental_Control",
}

REL_MAP = {
    # → CAUSES
    "results in": "CAUSES",
    "results_in": "CAUSES",
    "leads to": "CAUSES",
    "leads_to": "CAUSES",
    "produces": "CAUSES",
    "generates": "CAUSES",
    "triggers": "TRIGGERS",
    # → AFFECTS
    "impacts": "AFFECTS",
    "impacted_by": "AFFECTS",
    "affected_by": "AFFECTS",
    "influences": "AFFECTS",
    "modifies": "AFFECTS",
    # → HAS_DESCRIPTOR
    "characterized_by": "HAS_DESCRIPTOR",
    "characterized by": "HAS_DESCRIPTOR",
    "described_by": "HAS_DESCRIPTOR",
    "shows": "HAS_DESCRIPTOR",
    "has_property": "HAS_DESCRIPTOR",
    "has property": "HAS_DESCRIPTOR",
    "has_thickness": "HAS_DESCRIPTOR",
    "has_volume": "HAS_DESCRIPTOR",
    # → EVIDENCES / INDICATES
    "identified_by": "EVIDENCES",
    "supported_by": "EVIDENCES",
    "indicates": "INDICATES",
    "suggests": "SUGGESTS",
    # → PART_OF
    "is_part_of": "PART_OF",
    "part_of": "PART_OF",
    "belongs_to": "PART_OF",
    "is a": "PART_OF",
    "is_a": "PART_OF",
    # → LOCATED_IN
    "located_in": "LOCATED_IN",
    "located in": "LOCATED_IN",
    "occurs_in": "LOCATED_IN",
    "occurs in": "LOCATED_IN",
    "found_in": "LOCATED_IN",
    # → CONTROLS
    "controls": "CONTROLS",
    "regulates": "CONTROLS",
    # → RELATED_TO
    "related_to": "RELATED_TO",
    "associated_with": "RELATED_TO",
    "connected_to": "RELATED_TO",
}

MAX_ENTITY_LENGTH = 80


def _map_type(t: str) -> str:
    """Map raw type string to allowed node type."""
    if not t:
        return "Process"  # safe default
    if t in ALLOWED_NODE_TYPES:
        return t
    key = t.strip().lower().replace("_", " ").replace("-", " ")
    for k, v in TYPE_MAP.items():
        if k in key:
            return v
    return "Process"


def _map_relation(r: str, allowed_relations: List[str]) -> str:
    """Map raw relation to allowed relation. Returns RELATED_TO as fallback."""
    if not r:
        return "RELATED_TO" if "RELATED_TO" in allowed_relations else ""

    # Direct match
    if r in allowed_relations:
        return r

    # Uppercase match
    up = r.strip().upper().replace(" ", "_")
    if up in allowed_relations:
        return up

    # Lowercase lookup in map
    key = r.strip().lower().replace("_", " ")
    if key in REL_MAP and REL_MAP[key] in allowed_relations:
        return REL_MAP[key]

    # Try with underscores preserved
    key2 = r.strip().lower()
    if key2 in REL_MAP and REL_MAP[key2] in allowed_relations:
        return REL_MAP[key2]

    # Fallback
    return "RELATED_TO" if "RELATED_TO" in allowed_relations else ""


def _is_garbage_entity(text: str) -> bool:
    """
    Detect entities that are clearly not valid geological concepts.
    Returns True if the entity should be rejected.
    """
    if not text or not text.strip():
        return True
    text = text.strip()

    # Too long → probably a sentence fragment
    if len(text) > MAX_ENTITY_LENGTH:
        return True

    # Starts with section number (e.g., "4.2. Morphology Descriptors...")
    if re.match(r"^\d+[\.\)]\s*\d*", text):
        return True

    # Contains parenthetical references like "(i)", "(ii)", "(see ..."
    if re.search(r"\([ivx]+\)", text) or "(see " in text.lower():
        return True

    # Contains "et al." in a long string (it's a citation, not an entity)
    if "et al." in text and len(text) > 30:
        return True

    # Contains "e.g." or "i.e." (explanation text, not entity)
    if "e.g." in text.lower() or "i.e." in text.lower():
        return True

    # Mostly non-alphabetic characters
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    if alpha / max(1, len(text)) < 0.5:
        return True

    # Contains verb phrases (likely a sentence fragment)
    verb_patterns = [" is ", " are ", " was ", " were ", " may be ", " can be "]
    if any(vp in text.lower() for vp in verb_patterns) and len(text) > 40:
        return True

    return False


def repair_triples(
    llm_out: Dict[str, Any],
    context_text: str,
    allowed_relations: List[str]
) -> List[Dict[str, Any]]:
    """
    Clean, normalize, and filter LLM-extracted triples.
    
    Key principle: REJECT bad triples, don't INVENT missing fields.
    A KG with fewer but correct triples is better than one with
    many hallucinated triples.
    """
    triples = llm_out.get("triples", [])
    if not isinstance(triples, list):
        return []

    fixed = []
    for t in triples:
        if not isinstance(t, dict):
            continue

        src = (t.get("source") or "").strip()
        st = (t.get("source_type") or "").strip()
        rel = (t.get("relation") or "").strip()
        tgt = (t.get("target") or "").strip()
        tt = (t.get("target_type") or "").strip()

        # ── REJECT if source or target is empty ──
        # DO NOT GUESS — false triples are worse than missing triples
        if not src or not tgt:
            continue

        # ── REJECT garbage entities ──
        if _is_garbage_entity(src) or _is_garbage_entity(tgt):
            continue

        # ── REJECT if source == target (self-loop) ──
        if src.lower().strip() == tgt.lower().strip():
            continue

        # ── Map types and relations ──
        st = _map_type(st)
        tt = _map_type(tt)
        rel2 = _map_relation(rel, allowed_relations)

        if not rel2:
            continue

        # ── Build clean triple ──
        evidence = t.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {"quote": "", "confidence": 0.5}
        if "confidence" not in evidence:
            evidence["confidence"] = 0.5

        fixed.append({
            "source": src,
            "source_type": st,
            "relation": rel2,
            "target": tgt,
            "target_type": tt,
            "evidence": evidence,
        })

    return fixed