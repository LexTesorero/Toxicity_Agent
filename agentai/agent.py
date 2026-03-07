from llm_registry import LLMRegistry
from .classifierAgent import ClassifierAgent
from .responderAgent  import ResponderAgent
from .sarcasmDetector import SarcasmDetector
from .translatorAgent import TranslatorAgent

class ToxicityAgent:
    def __init__(self):
        self.registry = LLMRegistry()

        print("\n  Initialising agents …")
        self.translator = TranslatorAgent(self.registry)
        self.sarcasm    = SarcasmDetector(self.registry)
        self.classifier = ClassifierAgent(self.registry)
        self.responder  = ResponderAgent(self.registry)
        print("  All agents ready!\n")

    def detect_and_respond(self, content: str) -> dict:
        print(f"  PIPELINE START")
        print(f"  Input: {content[:100]}{'…' if len(content) > 100 else ''}\n")

        translation         = self.translator.translate(content)
        working_content     = translation["translated"]
        sarcasm_result      = self.sarcasm.detect(content, working_content)
        toxicity, sub_label, reason = self.classifier.classify(content, working_content, sarcasm_result)
        explanation         = self.responder.respond(working_content, toxicity, sub_label, sarcasm_result, reason)

        print(f"\n  Pipeline complete → {toxicity} (sarcasm: {sarcasm_result['is_sarcasm']})")

        return {
            "classification":    toxicity,
            "reason":            reason,
            "sub_label":         sub_label,
            "explanation":       explanation,
            "is_sarcasm":        sarcasm_result["is_sarcasm"],
            "meaning":           sarcasm_result["meaning"],
            "original":          content,
            "detected_language": translation["detected_language"],
            "translated":        translation["translated"] if not translation["is_english"] else None,
        }

    def display_result(self, result: dict) -> None:
        colors        = {"TOXIC": "\033[91m", "NEUTRAL": "\033[93m", "GOOD": "\033[92m"}
        icons         = {"TOXIC": "🔴", "NEUTRAL": "🟡", "GOOD": "🟢"}
        sarcasm_icons = {"no": "⚪", "ambiguous": "🟡", "sarcastic": "🟠"}
        reset         = "\033[0m"

        c = result["classification"]
        s = result["is_sarcasm"]

        print(f"\n  ANALYSIS RESULT")
        if result["translated"]:
            print(f"  Translated: {result['translated']}")
        print(f"  {icons.get(c, '')} {colors.get(c, reset)}Classification: {c}{reset}")
        print(f"  {sarcasm_icons.get(s, '')} Sarcasm: {s}")

        if s == "sarcastic":
            print(f"  True meaning: {result['meaning']}")
        elif s == "ambiguous":
            print(f"  Note: sarcasm was ambiguous — classified at face value")

        print(f"\n  Responder: {result['explanation']}")

if __name__ == "__main__":
    agent = ToxicityAgent()