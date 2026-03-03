from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os
from enum import Enum

load_dotenv()

# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    LOCAL = "local"
    GROQ  = "groq"

ACTIVE_PROVIDER = LLMProvider.GROQ

# ---------------------------------------------------------------------------
# Shared settings — must be defined BEFORE MODEL_CONFIGS references them
# ---------------------------------------------------------------------------

LLM_TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Models + configs
# ---------------------------------------------------------------------------

MODELS = {
    LLMProvider.LOCAL: {
        "qwen":  "qwen2.5:7b",
        "llama": "llama3.1:8b",
    },
    LLMProvider.GROQ: {
        "qwen":  "qwen/qwen3-32b",
        "gpt":   "openai/gpt-oss-120b",
    },
}

MODEL_CONFIGS = {
    LLMProvider.GROQ: {
        "qwen": {
            "temperature":      0.3,
            "reasoning_effort": "none",   # disables <think> blocks entirely — structured output is more reliable
            "model_kwargs": {
                "max_completion_tokens": 512,   # classifier/responder never need 4096
                "top_p":                 0.9,
            },
        },
        "gpt": {
            "temperature": 1,
            "model_kwargs": {
                "max_completion_tokens": 8192,
                "top_p": 1,
            },
        }
    },
    LLMProvider.LOCAL: {
        "qwen":  {"temperature": LLM_TEMPERATURE},
        "llama": {"temperature": LLM_TEMPERATURE},
    },
}

# ---------------------------------------------------------------------------
# Resolved model names
# ---------------------------------------------------------------------------

LLM_QWEN  = MODELS[ACTIVE_PROVIDER]["qwen"]
LLM_GPT = MODELS[ACTIVE_PROVIDER]["gpt"]

CTX_WINDOWS = {
    LLM_QWEN:  2048,
}

# ---------------------------------------------------------------------------
# ToxicityRAG
# ---------------------------------------------------------------------------

class ToxicityRAG:
    def __init__(self):
        self._llm_qwen  = None
        self._llm_gpt = None

    # models
    @property
    def llm_qwen(self) -> OllamaLLM:
        if self._llm_qwen is None:
            self._llm_qwen = self._connect_llm(LLM_QWEN)
        return self._llm_qwen

    @property
    def llm_gpt(self) -> OllamaLLM:
        if self._llm_gpt is None:
            self._llm_gpt = self._connect_llm(LLM_GPT)
        return self._llm_gpt

    # agents
    @property
    def llm_translator(self) -> OllamaLLM:
        return self.llm_gpt

    @property
    def llm_sarcasm(self) -> OllamaLLM:
        return self.llm_qwen

    @property
    def llm_classifier(self) -> OllamaLLM:
        return self.llm_qwen

    @property
    def llm_responder(self) -> OllamaLLM:
        return self.llm_qwen

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
            llm = ChatGroq(
                model=model_name,
                api_key=api_key,
                **config,
            )
            print(f"   ✓ {model_name} connected")
            return llm

        # LOCAL — Ollama
        ctx = CTX_WINDOWS.get(model_name, 2048)
        print(f"   Connecting to Ollama ({model_name}, ctx={ctx}) …")
        llm = OllamaLLM(
            model=model_name,
            num_ctx=ctx,
            **config,
        )
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