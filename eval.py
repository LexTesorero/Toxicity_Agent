# eval.py
import csv
import os
from datetime import datetime
from agentai.agent import ToxicityAgent

def run_eval():
    agent   = ToxicityAgent()
    results = []
    case_num = 1

    print("\n" + "="*60)
    print("  QUALITY EVALUATION — Multi-Agent Content Moderation")
    print("  Type your input, press Enter twice to submit.")
    print("  Type 'done' on a blank line to finish and export.")        
    print("="*60)

    while True:
        print(f"\n  ── Test Case {case_num} ──")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == "done" and not lines:
                break
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)

        if lines and lines[0].strip().lower() == "done":
            break
        if not lines or all(l.strip() == "" for l in lines):
            continue

        content = "\n".join(lines).strip()

        # run pipeline
        print()
        result = agent.detect_and_respond(content)

        print("\n" + "-"*60)
        print(f"  INPUT:           {content[:120]}")
        if result["translated"]:
            print(f"  TRANSLATED:      {result['translated'][:120]}")
        print(f"  LANGUAGE:        {result['detected_language']}")
        print(f"  SARCASM:         {result['is_sarcasm']}")
        if result["is_sarcasm"] == "sarcastic":
            print(f"  TRUE MEANING:    {result['meaning'][:120]}")
        print(f"  CLASSIFICATION:  {result['classification']} — {result['sub_label']}")
        if result.get("classifier_note"):
            print(f"  CLASSIFIER NOTE: {result['classifier_note'][:200]}")
        print(f"  EXPLANATION:     {result['explanation'][:200]}")
        print("-"*60)

        results.append({
            "case":           case_num,
            "input":          content,
            "language":       result["detected_language"],
            "sarcasm":        result["is_sarcasm"],
            "classification": result["classification"],
            "sub_label":      result["sub_label"],
            "classifier_note":result.get("classifier_note", ""),
            "explanation":    result["explanation"],
        })

        case_num += 1


if __name__ == "__main__":
    run_eval()