from llm_registry import LLMRegistry
from config import ACTIVE_PROVIDER, LLM_GPT, LLM_GPT_SAFE

print("=" * 50)
print(f"  PROVIDER : {ACTIVE_PROVIDER.value.upper()}")
print(f"  model  → {LLM_GPT}, {LLM_GPT_SAFE}")
print("=" * 50)

rag = LLMRegistry()

# Test GPT (sarcasm model)
print("\n  [TEST] GPT / sarcasm model ...")
response = rag.llm_sarcasm.invoke("Reply with only the word: GPT_OK")
print(f"  Response: {response.content}")

# Test GPT Safe (classifier/responder model)
print("\n  [TEST] GPT Safe / responder model ...")
response = rag.llm_responder.invoke("Reply with only the word: GPT_SAFE_OK")
print(f"  Response: {response.content}")

print("\n" + "=" * 50)
print(f"  ✓ Both models responding via {ACTIVE_PROVIDER.value.upper()}")
print("=" * 50)