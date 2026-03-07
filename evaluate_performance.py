"""
evaluate_performance.py
-----------------------
Runs a set of test inputs through the full ToxicityAgent pipeline
and records per-agent timing and token usage for Section 8.2.

Usage:
    python evaluate_performance.py

Output:
    - Per-run table printed to console
    - Averages and cost breakdown printed at the end
    - Results saved to performance_results.json
"""

import time
import json
from agentai.agent import ToxicityAgent

# ── Groq pricing (per 1M tokens) ─────────────────────────────────────────────
# Update these if Groq changes their rates: https://console.groq.com/docs/pricing
PRICE_PER_1M = {
    "gpt-oss-120b":           {"input": 3.00, "output": 9.00},
    "gpt-oss-safeguard-20b":  {"input": 1.50, "output": 4.50},
}

# Models assigned per agent (must match llm_registry.py)
AGENT_MODELS = {
    "translation":    "gpt-oss-120b",
    "sarcasm":        "gpt-oss-120b",
    "classification": "gpt-oss-safeguard-20b",
    "responder":      "gpt-oss-safeguard-20b",
}

# ── Test cases ────────────────────────────────────────────────────────────────
TEST_INPUTS = [
    "You are such an idiot, I hate you.",
    "The meeting is scheduled for 3 PM tomorrow.",
    "Oh wow, great job breaking everything again.",
    "Thank you so much for your help today, I really appreciate it!",
    "Bobo ka talaga, wala kang kwenta.",
    "Yeah sure, because people like you are SO helpful.",
    "Galing mo naman, palagi kang nakakatulong sa lahat.",
    "I disagree with your analysis; here is my counter-argument.",
    "Oh sure, I totally trust someone who can't even spell their own name.",
    "Di ko gets yung sinabi niya kanina ah.",
]

# ── Token extraction helper ───────────────────────────────────────────────────
def extract_tokens(response_obj) -> dict:
    """
    Tries multiple attribute paths that Groq/LangChain may use
    to attach token usage to a response object.
    Returns {"input": int, "output": int}
    """
    usage = (
        getattr(response_obj, "usage_metadata", None)
        or getattr(response_obj, "response_metadata", {}).get("token_usage", {})
        or {}
    )
    return {
        "input":  usage.get("input_tokens")  or usage.get("prompt_tokens", 0),
        "output": usage.get("output_tokens") or usage.get("completion_tokens", 0),
    }

# ── Patched agent methods ─────────────────────────────────────────────────────
def patched_translate(agent_self, content: str, collector: dict) -> dict:
    prompt = agent_self._build_prompt(content)
    t0 = time.perf_counter()
    raw_response = agent_self.registry.llm_translator.invoke(prompt)
    collector["timings"]["translation"] = round(time.perf_counter() - t0, 3)
    collector["tokens"]["translation"]  = extract_tokens(raw_response)

    raw = raw_response.content if hasattr(raw_response, "content") else raw_response
    raw = raw.strip()
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    result = {"detected_language": "unknown", "is_english": True, "translated": content}
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("DETECTED_LANGUAGE:"):
            result["detected_language"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("IS_ENGLISH:"):
            result["is_english"] = line.split(":", 1)[1].strip().upper() == "YES"
        elif line.upper().startswith("TRANSLATED:"):
            idx = raw.upper().find("TRANSLATED:")
            result["translated"] = raw[idx:].split(":", 1)[1].strip()
            break

    return result


def patched_detect(agent_self, orig: str, content: str, collector: dict) -> dict:
    import re
    prompt = agent_self._build_prompt(orig, content)
    t0 = time.perf_counter()
    raw_response = agent_self.registry.llm_sarcasm.invoke(prompt)
    collector["timings"]["sarcasm"] = round(time.perf_counter() - t0, 3)
    collector["tokens"]["sarcasm"]  = extract_tokens(raw_response)

    raw = raw_response.content if hasattr(raw_response, "content") else raw_response
    raw = raw.strip().upper()
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    is_sarcasm = "no"
    toxicity   = "NEUTRAL"
    meaning    = content

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("IS_SARCASTIC:"):
            val = line.replace("IS_SARCASTIC:", "").strip()
            if val == "YES":     is_sarcasm = "sarcastic"
            elif val == "UNKNOWN": is_sarcasm = "ambiguous"
        elif line.startswith("TOXICITY:"):
            toxicity = line.replace("TOXICITY:", "").strip()
        elif line.startswith("TRUE_MEANING:"):
            meaning = line.replace("TRUE_MEANING:", "").strip() or content

    return {"is_sarcasm": is_sarcasm, "toxicity": toxicity, "meaning": meaning}


def patched_classify(agent_self, orig: str, content: str, sarcasm_result: dict, collector: dict):
    import re
    prompt = agent_self._build_prompt(orig, content, sarcasm_result)
    t0 = time.perf_counter()
    raw_response = agent_self.registry.llm_classifier.invoke(prompt)
    collector["timings"]["classification"] = round(time.perf_counter() - t0, 3)
    collector["tokens"]["classification"]  = extract_tokens(raw_response)

    raw = raw_response.content if hasattr(raw_response, "content") else raw_response
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    TOXICITY  = None
    SUB_LABEL = None
    REASON    = "No reason provided."

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("LABEL:"):
            label_part = stripped[6:].strip().upper()
            match = re.match(r'^(TOXIC|NEUTRAL|GOOD)\s*-\s*([A-Z][A-Z\s]+?)$', label_part)
            if match:
                TOXICITY  = match.group(1).strip()
                SUB_LABEL = match.group(2).strip()
        elif stripped.upper().startswith("REASON:"):
            REASON = stripped[7:].strip()

    if not TOXICITY:
        fallback  = re.search(r'\b(TOXIC|NEUTRAL|GOOD)\b', raw.upper())
        TOXICITY  = fallback.group(1) if fallback else "NEUTRAL"
        SUB_LABEL = "UNKNOWN"

    return TOXICITY, SUB_LABEL, REASON


def patched_respond(agent_self, content: str, classification: str, sub_label: str, sarcasm_result: dict, classifier_note: str, collector: dict) -> str:
    import re
    prompt = agent_self._build_prompt(content, classification, sub_label, sarcasm_result, classifier_note)
    t0 = time.perf_counter()
    raw_response = agent_self.registry.llm_responder.invoke(prompt)
    collector["timings"]["responder"] = round(time.perf_counter() - t0, 3)
    collector["tokens"]["responder"]  = extract_tokens(raw_response)

    raw = raw_response.content if hasattr(raw_response, "content") else raw_response
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    match = re.search(r'Explanation:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else raw.strip()


# ── Cost calculation ──────────────────────────────────────────────────────────
def calc_cost(agent: str, tokens: dict) -> float:
    model  = AGENT_MODELS[agent]
    rates  = PRICE_PER_1M[model]
    return (tokens["input"] * rates["input"] + tokens["output"] * rates["output"]) / 1_000_000


# ── Main runner ───────────────────────────────────────────────────────────────
def run_evaluation():
    agent = ToxicityAgent()
    results = []

    print("\n" + "=" * 70)
    print("  PERFORMANCE EVALUATION")
    print("=" * 70)

    for i, content in enumerate(TEST_INPUTS, 1):
        print(f"\n  [TC-{i:02d}] {content[:60]}{'…' if len(content) > 60 else ''}")

        collector = {
            "timings": {},
            "tokens":  {},
        }

        # run pipeline with collectors
        translation      = patched_translate(agent.translator, content, collector)
        working_content  = translation["translated"]
        sarcasm_result   = patched_detect(agent.sarcasm, content, working_content, collector)
        toxicity, sub_label, classifier_note = patched_classify(
            agent.classifier, content, working_content, sarcasm_result, collector
        )
        explanation = patched_respond(
            agent.responder, working_content, toxicity, sub_label, sarcasm_result, classifier_note, collector
        )

        collector["timings"]["total"] = round(sum(collector["timings"].values()), 3)

        results.append({
            "id":              f"TC-{i:02d}",
            "input":           content,
            "classification":  toxicity,
            "sub_label":       sub_label,
            "is_sarcasm":      sarcasm_result["is_sarcasm"],
            "language":        translation["detected_language"],
            "timings":         collector["timings"],
            "tokens":          collector["tokens"],
        })

        t = collector["timings"]
        print(f"         Timing  → translate: {t.get('translation')}s | sarcasm: {t.get('sarcasm')}s | classify: {t.get('classification')}s | respond: {t.get('responder')}s | TOTAL: {t.get('total')}s")
        tok = collector["tokens"]
        for agent_name, tkn in tok.items():
            cost = calc_cost(agent_name, tkn)
            print(f"         Tokens  [{agent_name}] in: {tkn['input']} / out: {tkn['output']} — ${cost:.6f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    agent_names = ["translation", "sarcasm", "classification", "responder"]

    # Timing averages
    print("\n  RESPONSE TIME (seconds)")
    print(f"  {'Test':<8} {'Translate':>10} {'Sarcasm':>10} {'Classify':>10} {'Respond':>10} {'Total':>10}")
    print("  " + "-" * 58)
    for r in results:
        t = r["timings"]
        print(f"  {r['id']:<8} {t.get('translation', 0):>10.3f} {t.get('sarcasm', 0):>10.3f} {t.get('classification', 0):>10.3f} {t.get('responder', 0):>10.3f} {t.get('total', 0):>10.3f}")

    for label, key in [("Avg", None), ("Min", None), ("Max", None)]:
        vals = {a: [r["timings"].get(a, 0) for r in results] for a in agent_names + ["total"]}
        fn   = {"Avg": lambda x: sum(x)/len(x), "Min": min, "Max": max}[label]
        row  = {a: round(fn(vals[a]), 3) for a in agent_names + ["total"]}
        print(f"  {label:<8} {row['translation']:>10.3f} {row['sarcasm']:>10.3f} {row['classification']:>10.3f} {row['responder']:>10.3f} {row['total']:>10.3f}")

    # Token + cost breakdown
    print("\n  TOKEN USAGE & COST PER AGENT (averages across all test cases)")
    print(f"  {'Agent':<16} {'Model':<26} {'Avg In':>8} {'Avg Out':>9} {'Cost/Query':>12}")
    print("  " + "-" * 74)

    total_cost = 0.0
    for a in agent_names:
        model      = AGENT_MODELS[a]
        avg_in     = round(sum(r["tokens"].get(a, {}).get("input", 0)  for r in results) / len(results))
        avg_out    = round(sum(r["tokens"].get(a, {}).get("output", 0) for r in results) / len(results))
        cost       = calc_cost(a, {"input": avg_in, "output": avg_out})
        total_cost += cost
        print(f"  {a:<16} {model:<26} {avg_in:>8} {avg_out:>9} ${cost:>11.6f}")

    print(f"  {'TOTAL':<16} {'—':<26} {'':>8} {'':>9} ${total_cost:>11.6f}")

    # Cost at scale
    print("\n  COST PROJECTION AT SCALE")
    print(f"  {'Volume':<20} {'Total Tokens':>14} {'Estimated Cost':>16}")
    print("  " + "-" * 52)
    avg_total_tokens = round(sum(
        sum(r["tokens"].get(a, {}).get("input", 0) + r["tokens"].get(a, {}).get("output", 0) for a in agent_names)
        for r in results
    ) / len(results))

    for volume in [100, 1_000, 10_000, 100_000]:
        total_tokens = avg_total_tokens * volume
        cost         = total_cost * volume
        print(f"  {volume:<20,} {total_tokens:>14,} ${cost:>15.4f}")

    # Save to JSON
    with open("performance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved to performance_results.json")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()