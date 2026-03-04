from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os
from enum import Enum

load_dotenv()

class LLMProvider(str, Enum):
    LOCAL = "local"
    GROQ  = "groq"

ACTIVE_PROVIDER = LLMProvider.GROQ

LLM_TEMPERATURE = 0.2

MODELS = {
    LLMProvider.LOCAL: {
        "qwen":  "qwen2.5:7b",
        "llama": "llama3.1:8b",
    },
    LLMProvider.GROQ: {
        "gpt":      "openai/gpt-oss-120b",
        "gpt-safe": "openai/gpt-oss-safeguard-20b",
    },
}

MODEL_CONFIGS = {
    LLMProvider.GROQ: {
        "gpt": {
            "temperature":      1,
            "reasoning_effort": "medium",
            "model_kwargs": {
                "max_completion_tokens": 8192,
                "top_p": 1,
            },
        },
        "gpt-safe": {
            "temperature":      1,
            "reasoning_effort": "medium",
            "model_kwargs": {
                "max_completion_tokens": 8192,
                "top_p": 1,
            },
        },
    },
    LLMProvider.LOCAL: {
        "qwen":  {"temperature": LLM_TEMPERATURE},
        "llama": {"temperature": LLM_TEMPERATURE},
    },
}

LLM_GPT      = MODELS[ACTIVE_PROVIDER]["gpt"]
LLM_GPT_SAFE = MODELS[ACTIVE_PROVIDER]["gpt-safe"]

class ToxicityRAG:
    def __init__(self):
        self._llm_gpt      = None  # was _llm_qwen_safe (wrong name)
        self._llm_gpt_safe = None  # was missing

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
        return self.llm_gpt_safe

    @property
    def llm_responder(self):
        return self.llm_gpt_safe

    def _connect_llm(self, model_name: str):
        model_key = next(k for k, v in MODELS[ACTIVE_PROVIDER].items() if v == model_name)
        config = MODEL_CONFIGS[ACTIVE_PROVIDER][model_key]

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