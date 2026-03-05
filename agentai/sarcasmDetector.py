from llm_registry import LLMRegistry

class SarcasmDetector:
    def __init__(self, registry: LLMRegistry):
        self.registry = registry
        print("   SarcasmDetector ready")

    def _build_prompt(self, orig: str, content: str) -> str:
        text_length = len(content.split())

        if text_length <= 16:
            analysis_instruction = (
                "This is a SHORT text. Analyze at the SURFACE/SEMANTIC level:\n"
                "- Look at word choice, punctuation, emojis, and overall emotional tone\n"
                "- Identify the use of slang, specifically noting if it is derogatory or if it causes a contradiction in the text's overall meaning\n"
                "- Detect sentiment mismatch (e.g., positive words paired with negative emojis/punctuation, and vice versa)\n"
                "- If truly impossible to judge, use UNKNOWN"
            )
        else:
            analysis_instruction = (
                "This is a LONGER text. Analyze at the CONTEXTUAL level:\n"
                "- Look at the overall narrative and how the tone shifts\n"
                "- Check if the conclusion contradicts the setup\n"
                "- Detect irony through exaggeration, contradictions, or inconsistent emotional tone"
            )

        return f"""You are a sarcasm detection engine.

    DEFINITIONS:
    - YES      : the literal words mean the OPPOSITE of the true intent
    - NO       : the text means exactly what it says

    TOXICITY (based on TRUE meaning, not literal words):
    - GOOD     : positive, kind, or constructive
    - NEUTRAL  : no harmful or positive intent
    - TOXIC    : hateful, harmful, or offensive

    ANALYSIS APPROACH:
    \"\"\"{analysis_instruction}\"\"\"

    {orig}TEXT TO ANALYZE:
    \"\"\"{content}\"\"\"

    Reply in EXACTLY this format with no extra text:
    IS_SARCASTIC: [YES/NO/UNKNOWN]
    TOXICITY: [GOOD/NEUTRAL/TOXIC]
    TRUE_MEANING: [true meaning if YES, otherwise repeat the original text]"""

    def detect(self, orig: str, content: str) -> dict:
        prompt = self._build_prompt(orig, content)
        raw_response = self.registry.llm_sarcasm.invoke(prompt)
        raw = raw_response.content if hasattr(raw_response, "content") else raw_response
        raw = raw.strip().upper()

        # strip <think> block if present (reasoning models)
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        is_sarcasm = "no"
        toxicity   = "NEUTRAL"
        meaning    = content

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("IS_SARCASTIC:"):
                val = line.replace("IS_SARCASTIC:", "").strip().upper()
                if val == "YES":
                    is_sarcasm = "sarcastic"
                elif val == "UNKNOWN":
                    is_sarcasm = "ambiguous"
            elif line.startswith("TOXICITY:"):
                toxicity = line.replace("TOXICITY:", "").strip().upper()
            elif line.startswith("TRUE_MEANING:"):
                meaning = line.replace("TRUE_MEANING:", "").strip() or content

        match is_sarcasm:
            case "sarcastic":
                print("\n" + "-"*60)
                print(f"     SarcasmDetector: SARCASTIC ({toxicity})")
                print(f"     Original: {content[:300]}")
                print(f"     Meaning:  {meaning[:280]}")
                print("-"*60)
            case "ambiguous":
                print("\n" + "-"*60)
                print(f"     SarcasmDetector: AMBIGUOUS ({toxicity})")
                print(f"     Original: {content[:300]}")
                print("-"*60)
            case _:
                print("\n" + "-"*60)
                print(f"     SarcasmDetector: no sarcasm ({toxicity})")
                print("-"*60)

        return {
            "is_sarcasm": is_sarcasm,
            "toxicity":   toxicity,
            "meaning":    meaning,
        }