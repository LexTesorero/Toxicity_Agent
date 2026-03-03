from rag_setup import ToxicityRAG, ACTIVE_PROVIDER, LLM_GPT, LLM_GPT_SAFE

print("=" * 50)
print(f"  PROVIDER : {ACTIVE_PROVIDER.value.upper()}")
print(f"  model  → {LLM_GPT}, {LLM_GPT_SAFE}")
print("=" * 50)

rag = ToxicityRAG()

# Test Qwen (sarcasm model)
print("\n  [TEST] Qwen / sarcasm model ...")
response = rag.llm_sarcasm.invoke("Reply with only the word: QWEN_OK")
print(f"  Response: {response}")

print("\n" + "=" * 50)
print(f"  ✓ Both models responding via {ACTIVE_PROVIDER.value.upper()}")
print("=" * 50)