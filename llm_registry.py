import os
from langchain_ollama import OllamaLLM
from config import (
    ACTIVE_PROVIDER, LLMProvider,
    MODELS, MODEL_CONFIGS,
    LLM_GPT, LLM_GPT_SAFE,
)

class LLMRegistry:
    def __init__(self):
        self._llm_gpt      = None
        self._llm_gpt_safe = None

    @property
    def llm_gpt(self):
        if self._llm_gpt is None:
            self._llm_gpt = self._connect_llm(LLM_GPT)
        return self._llm_gpt

    @property
    def llm_gpt_safe(self):
        if self._llm_gpt_safe is None:
            self._llm_gpt_safe = self._connect_llm(LLM_GPT_SAFE)
        return self._llm_gpt_safe

    # agents
    @property
    def llm_translator(self):
        return self.llm_gpt

    @property
    def llm_sarcasm(self):
        return self.llm_gpt

    @property
    def llm_classifier(self):
        return self.llm_gpt

    @property
    def llm_responder(self):
        return self.llm_gpt_safe

    def _connect_llm(self, model_name: str):
        model_key = next(k for k, v in MODELS[ACTIVE_PROVIDER].items() if v == model_name)
        config    = MODEL_CONFIGS[ACTIVE_PROVIDER][model_key]

        if ACTIVE_PROVIDER == LLMProvider.GROQ:
            from langchain_groq import ChatGroq
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY not found. "
                    "Add it to your .env file: GROQ_API_KEY=your_key_here"
                )
            print(f"   Connecting to Groq ({model_name}) …")
            llm = ChatGroq(model=model_name, api_key=api_key, **config)
            print(f"   ✓ {model_name} connected")
            return llm

        # LOCAL — Ollama
        print(f"   Connecting to Ollama ({model_name}) …")
        llm = OllamaLLM(model=model_name, **config)
        try:
            llm.invoke("ping")
            print(f"   ✓ {model_name} connected")
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama model '{model_name}': {e}\n"
                f"Make sure Ollama is running and the model is pulled:\n"
                f"  ollama pull {model_name}"
            ) from e
        return llm