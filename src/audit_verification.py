#!/usr/bin/env python3
"""
audit_verification.py — Quick audit of verification results
=============================================================

Reads the verification audit log and prints a random sample of
NOT_SUPPORTED triples with their source chunks, so you can manually
check whether the verifier was correct.

Usage:
  python audit_verification.py \\
      --audit output/step4/verification_audit_v5.jsonl \\
      --sample 25 \\
      --verdict NOT_SUPPORTED

  # Or inspect WEAK_SUPPORT to calibrate thresholds:
  python audit_verification.py \\
      --audit output/step4/verification_audit_v5.jsonl \\
      --sample 15 \\
      --verdict WEAK_SUPPORT
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Audit verification results")
    parser.add_argument("--audit", required=True, help="Path to verification_audit_v5.jsonl")
    parser.add_argument("--sample", type=int, default=25, help="Number of triples to sample")
    parser.add_argument("--verdict", default="NOT_SUPPORTED",
                        help="Which verdict to sample (default: NOT_SUPPORTED)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load audit entries
    entries = []
    with open(args.audit) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Filter by verdict
    filtered = [e for e in entries if e.get("verdict", "") == args.verdict]
    print(f"Total entries: {len(entries)}")
    print(f"Entries with verdict={args.verdict}: {len(filtered)}")

    if not filtered:
        print("No entries found with that verdict.")
        return

    # Sample
    random.seed(args.seed)
    sample = random.sample(filtered, min(args.sample, len(filtered)))

    print(f"\n{'='*70}")
    print(f"  AUDIT SAMPLE: {len(sample)} x {args.verdict}")
    print(f"{'='*70}")

    for i, entry in enumerate(sample):
        print(f"\n--- [{i+1}/{len(sample)}] Triple #{entry.get('index', '?')} ---")
        print(f"  Subject:  {entry.get('source', '')}")
        print(f"  Relation: {entry.get('relation', '')}")
        print(f"  Object:   {entry.get('target', '')}")
        print(f"  Verdict:  {entry.get('verdict', '')}")

        evidence = entry.get("evidence", "")
        if evidence:
            print(f"  Evidence: {evidence[:200]}")

        reasoning = entry.get("reasoning", "")
        if reasoning:
            print(f"  Reasoning: {reasoning[:200]}")

        print(f"  ---")
        print(f"  YOUR JUDGMENT:  [ ] Agree (truly not supported)")
        print(f"                  [ ] Disagree (IS supported, verifier wrong)")
        print(f"                  [ ] Partial (relation exists but worded differently)")

    print(f"\n{'='*70}")
    print(f"Count your judgments:")
    print(f"  Agree    = ___  (verifier correct → extraction problem)")
    print(f"  Disagree = ___  (verifier wrong → verifier needs tuning)")
    print(f"  Partial  = ___  (prompt/gloss needs refinement)")
    print(f"")
    print(f"If Disagree > 40%: focus on improving verification prompt/model")
    print(f"If Agree > 60%:    focus on improving extraction prompts/retrieval")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()