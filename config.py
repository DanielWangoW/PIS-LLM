'''
Author: danielwangow daomiao.wang@live.com
Description: Centralized configuration management for ProEngOpt.
             All constants, paths, and environment variables are managed here.
-----> VENI VIDI VICI <-----
Copyright (c) 2025 by Daniel.Wang@Fudan University. All Rights Reserved.
'''

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root (local dev)
_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env")

# ── Streamlit Cloud secrets support ───────────────────────────────────────────
# When running on Streamlit Community Cloud, API keys are stored in
# Settings → Secrets as TOML.  We read them here so the rest of the
# codebase can use os.environ uniformly.
try:
    import streamlit as _st
    _sc = _st.secrets.get("api_keys", {})
    for _k, _v in _sc.items():
        if _v and not os.environ.get(_k):
            os.environ[_k] = str(_v)
except Exception:
    pass  # Not running inside Streamlit or secrets not configured yet

# ── Directories ────────────────────────────────────────────────────────────────
UPLOAD_DIR  = _ROOT / "upload_signals"
OUTPUT_DIR  = _ROOT / "outputs"
PROMPTS_DIR = _ROOT / "prompts"
CORE_DIR    = _ROOT / "core"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Supported LLM Models ───────────────────────────────────────────────────────
# Each entry: (display_name, model_id, provider, env_key_name)
LLM_MODELS = [
    # ── Qwen (DashScope) ──────────────────────────────────────────────────────
    ("Qwen-Plus",               "qwen-plus",                    "dashscope", "DASHSCOPE_API_KEY"),
    ("Qwen-Flash",              "qwen-turbo",                   "dashscope", "DASHSCOPE_API_KEY"),
    ("Qwen3.5-Plus",            "qwen3-235b-a22b",              "dashscope", "DASHSCOPE_API_KEY"),
    ("Qwen3.5-Flash",           "qwen3-30b-a3b",                "dashscope", "DASHSCOPE_API_KEY"),
    # ── DeepSeek (DashScope-compatible) ───────────────────────────────────────
    ("DeepSeek-V3.2",           "deepseek-v3",                  "dashscope",  "DASHSCOPE_API_KEY"),
    ("DeepSeek-R1-Distill-32B", "deepseek-r1",                  "dashscope",  "DASHSCOPE_API_KEY"),
    # ── GLM (Zhipu) ───────────────────────────────────────────────────────────
    ("GLM-5",                   "glm-5",                   "dashscope", "DASHSCOPE_API_KEY"),
    # ── Kimi (Moonshot) ───────────────────────────────────────────────────────
    ("Kimi-K2.5",               "kimi-k2.5",             "dashscope",  "DASHSCOPE_API_KEY"),
]

# Display name → (model_id, provider, env_key_name)
LLM_MODEL_MAP = {name: (mid, prov, ekey) for name, mid, prov, ekey in LLM_MODELS}
LLM_MODEL_NAMES = [name for name, *_ in LLM_MODELS]

# Provider base URLs (OpenAI-compatible)
LLM_PROVIDER_URLS = {
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek":  "https://api.deepseek.com/v1",
    "zhipu":     "https://open.bigmodel.cn/api/paas/v4",
    "moonshot":  "https://api.moonshot.cn/v1",
}

# Default API keys from environment
DEFAULT_API_KEYS = {
    "DASHSCOPE_API_KEY": os.environ.get("DASHSCOPE_API_KEY", ""),
    "DEEPSEEK_API_KEY":  os.environ.get("DEEPSEEK_API_KEY",  ""),
    "ZHIPU_API_KEY":     os.environ.get("ZHIPU_API_KEY",     ""),
    "MOONSHOT_API_KEY":  os.environ.get("MOONSHOT_API_KEY",  ""),
}

# ── Legacy single-model fallback (for llm_client.py compatibility) ─────────────
DASHSCOPE_API_KEY: str = os.environ.get("DASHSCOPE_API_KEY", "")
LLM_MODEL: str         = "qwen-plus"

# ── System prompt ──────────────────────────────────────────────────────────────
LLM_SYSTEM_PROMPT: str = (
    "You are a cardiac signal analysis system developed at Fudan University. "
    "Generate structured, factual cardiovascular assessment reports based strictly "
    "on the provided quantitative metrics. Do not fabricate patient identities, "
    "clinical histories, or conclusions unsupported by the data. "
    "Use hedged clinical language and always include a computational disclaimer."
)

# ── Prompt Templates ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE_EN = PROMPTS_DIR / "prompt_en.txt"
PROMPT_TEMPLATE_ZH = PROMPTS_DIR / "prompt_zh.txt"

PROMPT_MAP = {
    "English": PROMPT_TEMPLATE_EN,
    "中文":    PROMPT_TEMPLATE_ZH,
}

# ── TDA Signal Processing Parameters ──────────────────────────────────────────
TDA_PARAMS = {
    "d":         25,
    "tau":       5,
    "q":         50,
    "n_points":  200,
    "n_diag":    2,
    "normalize": True,
    "adaptive":  False,
}

# ── Application Settings ───────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB: int = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 50))
SUPPORTED_EXTENSIONS   = ["csv", "txt"]
APP_VERSION            = "CardioDetector_TDA_v2.1"
