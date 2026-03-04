from llm_registry import LLMRegistry

class TranslatorAgent:
    def __init__(self, registry: LLMRegistry):
        self.registry = registry
        print("   Translator ready")

    def _build_prompt(self, content: str) -> str:
        return f"""You are a strict multilingual language detection and translation engine.
Your ONLY job is to detect language and translate. Nothing else.

TASK:
1. Detect the language of the input text
2. If the text is NOT in English, translate it to English WORD-FOR-WORD
3. If the text IS in English, return it as-is
4. Preserve tone, emotion, and intent exactly — do NOT sanitize, soften, or improve

STRICT RULES:
- Translate LITERALLY — choose the closest English word, not the "better" word
- Preserve slang, insults, profanity, and informal language exactly as intended
- Do NOT substitute informal words with formal alternatives
- Do NOT fix grammar unless required for basic English readability
- Do NOT paraphrase — if the source is blunt, the translation must be blunt
- Filipino, Tagalog, Cebuano and other Philippine languages are NOT English — always translate them
- If text mixes languages (code-switching), translate ONLY the non-English parts
- Preserve original punctuation and capitalization as much as possible

OUTPUT RULES:
- Reply ONLY in the format below — no extra lines, no explanation, no commentary
- Do NOT add any text before or after the format

DETECTED_LANGUAGE: <language name>
IS_ENGLISH: <YES or NO>
TRANSLATED: <exact translated text or original if already English>

Input text:
\"\"\"{content}\"\"\""""

    def translate(self, content: str) -> dict:
        prompt = self._build_prompt(content)
        raw_response = self.registry.llm_translator.invoke(prompt)
        raw = raw_response.content if hasattr(raw_response, "content") else raw_response
        raw = raw.strip()

        # strip <think> block if present
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        result = {
            "detected_language": "unknown",
            "is_english":        True,
            "translated":        content,   # fallback to original
        }

        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("DETECTED_LANGUAGE:"):
                result["detected_language"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("IS_ENGLISH:"):
                result["is_english"] = line.split(":", 1)[1].strip().upper() == "YES"
            elif line.upper().startswith("TRANSLATED:"):
                result["translated"] = line.split(":", 1)[1].strip()

        translation_preview = "(English — no translation needed)" if result["is_english"] else f"→ {result['translated'][:80]}"
        print(f"     Translator: [{result['detected_language']}] {translation_preview}")
        return result