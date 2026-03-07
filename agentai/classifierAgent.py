from llm_registry import LLMRegistry
import re

class ClassifierAgent:
    def __init__(self, registry: LLMRegistry):
        self.registry = registry
        print("   Classifier ready")

    def _build_prompt(self, orig:str, content: str, sarcasm_result: dict) -> str:
        is_sarcasm = sarcasm_result["is_sarcasm"]
        meaning    = sarcasm_result["meaning"]

        sarcasm_note = ""
        if is_sarcasm == "sarcastic":
            sarcasm_note = (
                f"\nNOTE: The original text was sarcastic.\n"
                f"  Non-Translated Text: \"\"\"{orig}\"\"\"\n"
                f"  Original:     \"{content}\"\n"
                f"  True meaning: \"{meaning}\"\n"
                f"Classify based on the TRUE MEANING, not the literal words.\n"
            )
        elif is_sarcasm == "ambiguous":
            sarcasm_note = (
                "\nNOTE: This text may or may not be sarcastic — context is unavailable.\n"
                "Classify at face value, but be aware the true intent is uncertain.\n"
            )

        text_to_classify = meaning if is_sarcasm == "sarcastic" else content

        return f"""You are a strict content classification engine.

DEFINITIONS:
- TOXIC   : hate speech, threats, harassment, discrimination, personal attacks, obscene language
- NEUTRAL : factual statements, disagreements without hostility, questions, constructive criticism
- GOOD    : supportive, encouraging, appreciative, respectful, constructive communication

VALID LABELS (you must use one of these exactly):
TOXIC   - HATE SPEECH
TOXIC   - THREAT
TOXIC   - HARASSMENT
TOXIC   - PERSONAL ATTACK
TOXIC   - OBSCENE LANGUAGE
NEUTRAL - FACTUAL STATEMENT
NEUTRAL - CONSTRUCTIVE CRITICISM
NEUTRAL - QUESTION
NEUTRAL - DISAGREEMENT
GOOD    - SUPPORTIVE
GOOD    - APPRECIATIVE
GOOD    - ENCOURAGING
GOOD    - RESPECTFUL

CRITICAL CONTEXT RULES:
1. DO NOT classify a text as TOXIC solely because it contains profanity, slurs, or sensitive words.
2. Evaluate the TARGET: Is the word being used as a weapon to attack, demean, or harass someone? If yes, it is TOXIC.
3. Evaluate for RECLAMATION/SELF-REFERENCE: If a user is using a marginalized term to refer to themselves or as a term of endearment within their own community, label it NEUTRAL or GOOD.
4. Evaluate for QUOTING: If the user is clearly quoting a movie, song, or discussing a word educationally without malicious intent, label it NEUTRAL.
{sarcasm_note}
NON-TRANSLATED TEXT:
\"\"\"{orig}\"\"\"

TEXT TO CLASSIFY:
\"\"\"{text_to_classify}\"\"\"

Reply in EXACTLY this format — no extra lines:
LABEL: TOXIC - HATE SPEECH
REASON: [one sentence explaining why, referencing specific words or tone]"""

    def classify(self, orig: str, content: str, sarcasm_result: dict) -> tuple[str, str, str]:
        prompt       = self._build_prompt(orig, content, sarcasm_result)
        raw_response = self.registry.llm_classifier.invoke(prompt)

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

        print(f"     Classifier: {TOXICITY.capitalize()} - {SUB_LABEL.lower()}")
        print(f"     Reason: {REASON[:120]}{'…' if len(REASON) > 120 else ''}")
        return TOXICITY, SUB_LABEL, REASON