from enum import Enum
from dotenv import load_dotenv

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

# Resolved model names
LLM_GPT      = MODELS[ACTIVE_PROVIDER]["gpt"]
LLM_GPT_SAFE = MODELS[ACTIVE_PROVIDER]["gpt-safe"]