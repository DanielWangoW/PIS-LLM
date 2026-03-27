'''
Author: danielwangow daomiao.wang@live.com
LastEditTime: 2026-03-27 18:28:20
Description: ProEngOpt v2.3 — Physiological Anomaly Analyzer
             - Multi-model dropdown with per-provider API key
             - TDA result cache (run once per file)
             - Single-page fused layout with inline images
             - Light/dark theme with full selectbox color fix
             - Report constrained to Fudan University attribution only
             - Lock API Key option (retain key when switching models)
             - Regenerate Report button: session_state trigger fix
             - Distinct primary/secondary button styles
-----> VENI VIDI VICI <-----
Copyright (c) 2025 by Daniel.Wang@Fudan University. All Rights Reserved.
'''

import hashlib
import json
import time
import uuid
from pathlib import Path

import streamlit as st
import numpy as np

import config
from core.signal_processor import (
    process_signal_file,
    build_llm_prompt,
    SignalProcessingResult,
)

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CardioDetector TDA",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Theme definitions
# ══════════════════════════════════════════════════════════════════════════════
THEMES = {
    "dark": {
        "bg":             "linear-gradient(135deg,#0f0c29,#302b63,#24243e)",
        "sidebar_bg":     "rgba(255,255,255,0.04)",
        "sidebar_border": "rgba(255,255,255,0.08)",
        "card_bg":        "rgba(255,255,255,0.06)",
        "card_border":    "rgba(255,255,255,0.10)",
        "text":           "#e2e8f0",
        "text_muted":     "#a0aec0",
        "text_label":     "#90cdf4",
        "text_value":     "#ffffff",
        "accent":         "#63b3ed",
        "btn_grad":       "linear-gradient(135deg,#667eea,#764ba2)",
        "btn_shadow":     "rgba(102,126,234,0.4)",
        "report_bg":      "rgba(255,255,255,0.04)",
        "report_border":  "rgba(255,255,255,0.10)",
        "divider":        "rgba(255,255,255,0.08)",
        "step_done":      "#68d391",
        "step_active":    "#63b3ed",
        "step_pending":   "#4a5568",
        "upload_border":  "rgba(99,179,237,0.4)",
        "upload_bg":      "rgba(255,255,255,0.03)",
        # selectbox / input overrides
        "input_bg":       "#1e2235",
        "input_text":     "#e2e8f0",
        "input_border":   "rgba(99,179,237,0.35)",
        "icon":           "🌙",
        "label":          "Dark Mode",
    },
    "light": {
        "bg":             "linear-gradient(135deg,#f0f4ff,#e8f0fe,#f5f7ff)",
        "sidebar_bg":     "rgba(255,255,255,0.88)",
        "sidebar_border": "rgba(99,130,201,0.18)",
        "card_bg":        "rgba(255,255,255,0.92)",
        "card_border":    "rgba(99,130,201,0.20)",
        "text":           "#1e293b",
        "text_muted":     "#64748b",
        "text_label":     "#2563eb",
        "text_value":     "#0f172a",
        "accent":         "#3b82f6",
        "btn_grad":       "linear-gradient(135deg,#3b82f6,#6366f1)",
        "btn_shadow":     "rgba(59,130,246,0.35)",
        "report_bg":      "rgba(248,250,252,0.95)",
        "report_border":  "rgba(99,130,201,0.20)",
        "divider":        "rgba(99,130,201,0.15)",
        "step_done":      "#16a34a",
        "step_active":    "#2563eb",
        "step_pending":   "#94a3b8",
        "upload_border":  "rgba(59,130,246,0.45)",
        "upload_bg":      "rgba(239,246,255,0.8)",
        # selectbox / input overrides
        "input_bg":       "#ffffff",
        "input_text":     "#1e293b",
        "input_border":   "rgba(59,130,246,0.40)",
        "icon":           "☀️",
        "label":          "Light Mode",
    },
}

# ── Session state init ─────────────────────────────────────────────────────────
_SS_DEFAULTS = {
    "theme":            "dark",
    "analysis_result":  None,
    "tda_file_hash":    None,   # MD5 of last analysed file bytes
    "tda_params_hash":  None,   # hash of TDA params used
    "llm_report":       "",
    "llm_error":        None,
    "chat_history":     [],
    "session_id":       uuid.uuid4().hex[:10],
    # per-provider API keys (runtime overrides)
    "api_keys":         dict(config.DEFAULT_API_KEYS),
    # locked API key: when True, switching model keeps the same key value
    "lock_api_key":     False,
    "locked_key_value": "",
    # trigger flags for buttons (avoids Streamlit rerun-reset issue)
    "trigger_regen":    False,
    "trigger_full":     False,
    # persist uploaded file info so theme switch doesn't lose results
    "cached_filename":  None,
    "cached_filesize":  None,
    "cached_filefmt":   None,
    "cached_filebytes": None,   # bytes of last uploaded file (for re-hash)
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

T = THEMES[st.session_state.theme]


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic CSS injection
# ══════════════════════════════════════════════════════════════════════════════
def inject_css(t: dict):
    st.markdown(f"""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {{
    background: {t['bg']};
    min-height: 100vh;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {t['sidebar_bg']} !important;
    border-right: 1px solid {t['sidebar_border']};
    backdrop-filter: blur(14px);
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {{ color: {t['text']} !important; }}

/* ── Fix selectbox / text_input / multiselect in ALL contexts ── */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] span,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
.stSelectbox label,
.stTextInput label,
.stSelectbox div[data-baseweb="select"] * {{
    color: {t['input_text']} !important;
    background-color: {t['input_bg']} !important;
}}
div[data-baseweb="select"] > div {{
    border-color: {t['input_border']} !important;
    border-radius: 8px !important;
}}
/* Dropdown menu items */
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {{
    color: {t['input_text']} !important;
    background-color: {t['input_bg']} !important;
}}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover {{
    background-color: {t['accent']}22 !important;
}}
/* Selectbox chevron icon */
div[data-baseweb="select"] svg {{ fill: {t['input_text']} !important; }}

/* ── Metric cards ── */
.metric-card {{
    background: {t['card_bg']};
    border: 1px solid {t['card_border']};
    border-radius: 14px;
    padding: 16px 20px;
    backdrop-filter: blur(8px);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    margin-bottom: 10px;
}}
.metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.18);
}}
.metric-label {{
    font-size: 0.75rem;
    color: {t['text_label']};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 5px;
    font-weight: 600;
}}
.metric-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {t['text_value']};
    line-height: 1.2;
}}
.metric-sub {{
    font-size: 0.73rem;
    color: {t['text_muted']};
    margin-top: 3px;
}}

/* ── Status badges ── */
.badge {{
    display: inline-block;
    padding: 4px 13px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}}
.badge-normal   {{ background:rgba(72,187,120,0.18); color:#16a34a; border:1px solid #16a34a; }}
.badge-mild     {{ background:rgba(234,179,8,0.18);  color:#ca8a04; border:1px solid #ca8a04; }}
.badge-moderate {{ background:rgba(234,88,12,0.18);  color:#ea580c; border:1px solid #ea580c; }}
.badge-severe   {{ background:rgba(220,38,38,0.18);  color:#dc2626; border:1px solid #dc2626; }}

/* ── Section header ── */
.section-header {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {t['text_label']};
    border-left: 3px solid {t['accent']};
    padding-left: 11px;
    margin: 22px 0 12px 0;
    letter-spacing: 0.02em;
}}

/* ── Progress steps ── */
.step-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 9px 14px;
    border-radius: 10px;
    margin-bottom: 6px;
    transition: background 0.2s;
}}
.step-row.active  {{ background:rgba(99,179,237,0.12); border:1px solid {t['step_active']}44; }}
.step-row.done    {{ background:rgba(72,187,120,0.08);  border:1px solid {t['step_done']}33; }}
.step-row.pending {{ background:transparent; border:1px solid transparent; }}
.step-dot {{ width:11px; height:11px; border-radius:50%; flex-shrink:0; }}
.step-dot.active  {{
    background:{t['step_active']};
    box-shadow:0 0 0 3px {t['step_active']}44;
    animation:pulse-dot 1.4s infinite;
}}
.step-dot.done    {{ background:{t['step_done']}; }}
.step-dot.pending {{ background:{t['step_pending']}; }}
@keyframes pulse-dot {{
    0%,100% {{ transform:scale(1); opacity:1; }}
    50%      {{ transform:scale(1.5); opacity:0.6; }}
}}
.step-text.active  {{ color:{t['step_active']}; font-weight:600; font-size:0.9rem; }}
.step-text.done    {{ color:{t['step_done']};   font-weight:500; font-size:0.9rem; }}
.step-text.pending {{ color:{t['step_pending']}; font-size:0.9rem; }}
.step-num {{ font-size:0.7rem; font-weight:700; color:{t['text_muted']}; min-width:18px; }}

/* ── Report box ── */
.report-box {{
    background: {t['report_bg']};
    border: 1px solid {t['report_border']};
    border-radius: 14px;
    padding: 26px 30px;
    color: {t['text']};
    line-height: 1.85;
    font-size: 0.95rem;
}}
.report-box h1, .report-box h2, .report-box h3 {{ color:{t['text_label']}; }}
.report-box strong {{ color:{t['accent']}; }}

/* ── Upload area ── */
[data-testid="stFileUploader"] {{
    border: 2px dashed {t['upload_border']} !important;
    border-radius: 14px !important;
    background: {t['upload_bg']} !important;
    padding: 10px 14px !important;
    transition: border-color 0.3s;
}}
[data-testid="stFileUploader"]:hover {{ border-color:{t['accent']} !important; }}
[data-testid="stFileUploader"] section {{ padding:8px 12px !important; min-height:unset !important; }}

/* ── Primary button (Run Full Analysis) ── */
.btn-primary-wrap .stButton > button {{
    background:{t['btn_grad']} !important;
    color:white !important;
    border:none !important;
    border-radius:10px !important;
    padding:10px 28px !important;
    font-weight:600 !important;
    font-size:0.95rem !important;
    transition:all 0.22s ease !important;
    box-shadow:0 4px 15px {t['btn_shadow']} !important;
}}
.btn-primary-wrap .stButton > button:hover {{
    transform:translateY(-2px) !important;
    box-shadow:0 8px 24px {t['btn_shadow']} !important;
}}

/* ── Secondary button (Regenerate Report) ── */
.btn-secondary-wrap .stButton > button {{
    background: transparent !important;
    color: {t['accent']} !important;
    border: 1.5px solid {t['accent']} !important;
    border-radius:10px !important;
    padding:10px 28px !important;
    font-weight:600 !important;
    font-size:0.95rem !important;
    transition:all 0.22s ease !important;
    box-shadow: none !important;
}}
.btn-secondary-wrap .stButton > button:hover {{
    background: {t['accent']}18 !important;
    transform:translateY(-2px) !important;
    box-shadow: 0 4px 14px {t['btn_shadow']} !important;
}}
.btn-secondary-wrap .stButton > button:disabled {{
    opacity:0.38 !important;
    cursor:not-allowed !important;
    transform:none !important;
}}

/* ── Other buttons (theme toggle, clear chat, download) ── */
.stButton > button {{
    border-radius:10px !important;
    font-weight:600 !important;
    font-size:0.92rem !important;
    transition:all 0.22s ease !important;
}}

/* ── Chat messages ── */
.chat-user {{
    background:{t['card_bg']};
    border:1px solid {t['card_border']};
    border-radius:14px 14px 4px 14px;
    padding:12px 18px;
    margin:8px 0 8px 60px;
    color:{t['text']};
    font-size:0.92rem;
    line-height:1.6;
}}
.chat-assistant {{
    background:{t['report_bg']};
    border:1px solid {t['report_border']};
    border-radius:14px 14px 14px 4px;
    padding:14px 20px;
    margin:8px 60px 8px 0;
    color:{t['text']};
    font-size:0.92rem;
    line-height:1.7;
}}
.chat-role {{ font-size:0.72rem; font-weight:700; letter-spacing:0.06em; margin-bottom:6px; text-transform:uppercase; }}
.chat-role.user      {{ color:{t['accent']}; }}
.chat-role.assistant {{ color:{t['step_done']}; }}

/* ── Inline image caption ── */
.img-caption {{
    text-align:center;
    font-size:0.78rem;
    color:{t['text_muted']};
    margin-top:-6px;
    margin-bottom:14px;
    font-style:italic;
}}

/* ── Divider ── */
hr {{ border-color:{t['divider']} !important; }}

/* ── General text ── */
p, li {{ color:{t['text']}; }}
h1, h2, h3 {{ color:{t['text_value']}; }}
</style>
""", unsafe_allow_html=True)


inject_css(T)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def badge_class(text: str) -> str:
    t = text.lower()
    if "normal" in t or "healthy" in t: return "badge-normal"
    if "mild"   in t:                   return "badge-mild"
    if "moderate" in t:                 return "badge-moderate"
    if "severe" in t:                   return "badge-severe"
    return "badge-mild"


def metric_card(label: str, value: str, sub: str = "", color: str = "") -> str:
    val_style = f"color:{color};" if color else ""
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value' style='{val_style}'>{value}</div>"
        + (f"<div class='metric-sub'>{sub}</div>" if sub else "")
        + "</div>"
    )


def render_step(num: int, icon: str, text: str, state: str) -> str:
    prefix = "✓ " if state == "done" else ("▶ " if state == "active" else "")
    return (
        f"<div class='step-row {state}'>"
        f"<span class='step-num'>{num:02d}</span>"
        f"<div class='step-dot {state}'></div>"
        f"<span class='step-text {state}'>{prefix}{icon} {text}</span>"
        f"</div>"
    )


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def params_hash(params: dict) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


def get_effective_api_key(provider_env_key: str) -> str:
    """Return runtime override key if set, else fall back to env/config."""
    return st.session_state.api_keys.get(provider_env_key, "") or ""


def call_llm_openai_compat(
    messages: list,
    model_id: str,
    provider: str,
    api_key: str,
) -> str:
    """Stream-collect response via OpenAI-compatible API."""
    from openai import OpenAI
    base_url = config.LLM_PROVIDER_URLS[provider]
    client = OpenAI(api_key=api_key, base_url=base_url)
    chunks = []
    stream = client.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=True,
        max_tokens=2048,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)
    return "".join(chunks)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Theme toggle ───────────────────────────────────────────────────────────
    other_theme = "light" if st.session_state.theme == "dark" else "dark"
    other_T = THEMES[other_theme]
    if st.button(f"{other_T['icon']} Switch to {other_T['label']}", use_container_width=True):
        st.session_state.theme = other_theme
        st.rerun()
    st.markdown(
        f"<div style='color:{T['text_muted']};font-size:0.78rem;margin-top:4px;"
        f"text-align:center'>Current: {T['label']}</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        f"<div style='font-size:1.15rem;font-weight:700;color:{T['text_value']}'>🫀 CardioDetector</div>",
        unsafe_allow_html=True,
    )
    # st.markdown(
    #     f"<div style='color:{T['text_muted']};font-size:0.78rem;margin-bottom:12px'>{config.APP_VERSION}</div>",
    #     unsafe_allow_html=True,
    # )

    # ── Language ───────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-weight:600;color:{T['text_label']};margin-bottom:6px'>📄 Report Language</div>",
        unsafe_allow_html=True,
    )
    language = st.selectbox(
        "language_select",
        options=list(config.PROMPT_MAP.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    # ── Model selection ────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-weight:600;color:{T['text_label']};margin-bottom:6px'>🤖 LLM Model</div>",
        unsafe_allow_html=True,
    )
    selected_model_name = st.selectbox(
        "model_select",
        options=config.LLM_MODEL_NAMES,
        index=0,
        label_visibility="collapsed",
    )
    model_id, provider, env_key = config.LLM_MODEL_MAP[selected_model_name]

    # ── API Key for selected provider ──────────────────────────────────────────
    provider_label = {
        "dashscope": "Qwen (DashScope)",
        "deepseek":  "DeepSeek",
        "dashscope": "Zhipu AI (GLM)",
        "moonshot":  "Moonshot (Kimi)",
    }.get(provider, provider)

    # ── Lock API Key toggle ──────────────────────────────────────────────
    lock_api = st.toggle(
        "🔒 Lock API Key",
        value=st.session_state.lock_api_key,
        key="lock_api_toggle",
    )
    if lock_api != st.session_state.lock_api_key:
        st.session_state.lock_api_key = lock_api

    # Resolve which key value to display
    if st.session_state.lock_api_key:
        # When locked: always show the locked value regardless of provider
        display_key = st.session_state.locked_key_value
    else:
        display_key = st.session_state.api_keys.get(env_key, "")

    lock_badge = " <span style='color:#f6ad55;font-size:0.72rem'>🔒 Locked</span>" if st.session_state.lock_api_key else ""
    st.markdown(
        f"<div style='font-size:0.8rem;color:{T['text_muted']};margin-bottom:4px'>"
        f"🔑 Alibaba Model Studio API Key — {provider_label}{lock_badge}</div>",
        unsafe_allow_html=True,
    )
    new_key = st.text_input(
        "api_key_input",
        value=display_key,
        type="password",
        placeholder=f"Enter {provider_label} API key...",
        label_visibility="collapsed",
    )
    if new_key != display_key:
        if st.session_state.lock_api_key:
            st.session_state.locked_key_value = new_key.strip()
        st.session_state.api_keys[env_key] = new_key.strip()
    elif st.session_state.lock_api_key and st.session_state.locked_key_value:
        # Propagate locked key to current provider slot on model switch
        st.session_state.api_keys[env_key] = st.session_state.locked_key_value

    st.divider()

    # ── TDA Parameters ─────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-weight:600;color:{T['text_label']};margin-bottom:6px'>🔬 TDA Parameters</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Advanced Parameters", expanded=False):
        d        = st.slider("Embedding Dimension (d)", 5, 50, config.TDA_PARAMS["d"], step=5)
        tau      = st.slider("Time Delay (τ)", 1, 20, config.TDA_PARAMS["tau"], step=1)
        q        = st.slider("DTM Parameter (q)", 10, 100, config.TDA_PARAMS["q"], step=10)
        n_points = st.slider("Subsampling Points", 50, 500, config.TDA_PARAMS["n_points"], step=50)
        adaptive = st.toggle("Adaptive Parameter Selection", value=config.TDA_PARAMS["adaptive"])

    tda_params = {
        "d": d, "tau": tau, "q": q,
        "n_points": n_points, "n_diag": config.TDA_PARAMS["n_diag"],
        "normalize": True, "adaptive": adaptive,
    }

    st.divider()
    st.markdown(
        f"<div style='color:{T['text_muted']};font-size:0.73rem;text-align:center'>"
        "© 2025-2026 <a href='https://danielwangow.github.io/' target='_blank' style='color:inherit;text-decoration:underline'>Daomiao Wang</a>@Fudan University </div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Page header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='text-align:center;background:linear-gradient(90deg,#667eea,#f093fb);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "font-size:2.1rem;margin-bottom:2px'>PIS-LLM: Adverse Physiological Event Analyzer</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center;color:{T['text_muted']};font-size:1.0rem;margin-bottom:20px'>"
    "Cardiac Signal Analysis · Label-free Anomaly Detection · Topological Data Analysis · LLM Report Interpretation</p>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# File upload
# ══════════════════════════════════════════════════════════════════════════════
col_up, col_info = st.columns([3, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "Upload PPG/ECG signal file (CSV / TXT, single numeric column)",
        type=config.SUPPORTED_EXTENSIONS,
        label_visibility="visible",
    )
with col_info:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='metric-card' style='padding:10px 14px'>"
        f"<div class='metric-label'>Supported</div>"
        f"<div class='metric-value' style='font-size:0.95rem'>CSV / TXT</div>"
        f"<div class='metric-sub'>Max {config.MAX_UPLOAD_SIZE_MB} MB</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Landing state: show welcome only if no file AND no cached result ──────────────────────
has_cached_result = st.session_state.analysis_result is not None

if uploaded_file is None and not has_cached_result:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("🔬 Feature Engineer", "TDA Cycler",
                                " Persistent Homology"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("🤖 LLM Engine", selected_model_name,
                                f"Provider: {provider_label}"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("💬 Chat Mode", "Enabled",
                                "Multi-turn Q&A · Analysis Context"), unsafe_allow_html=True)
    st.info("👆 Upload a cardiac signal CSV/TXT file to begin analysis.", icon="ℹ️")
    st.stop()

# ── Resolve file bytes: from uploader or from session cache ────────────────────────────────
if uploaded_file is not None:
    file_bytes   = bytes(uploaded_file.getbuffer())
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > config.MAX_UPLOAD_SIZE_MB:
        st.error(f"File size ({file_size_mb:.1f} MB) exceeds limit ({config.MAX_UPLOAD_SIZE_MB} MB).", icon="🚫")
        st.stop()
    # Persist file metadata into session so theme switch can restore it
    st.session_state.cached_filename  = uploaded_file.name
    st.session_state.cached_filesize  = file_size_mb
    st.session_state.cached_filefmt   = uploaded_file.name.split(".")[-1].upper()
    st.session_state.cached_filebytes = file_bytes
else:
    # Theme switch path: no new file, but cached result exists
    file_bytes   = st.session_state.cached_filebytes or b""
    file_size_mb = st.session_state.cached_filesize  or 0.0

# ── Compute hashes for cache check ─────────────────────────────────────────────────────────────
current_file_hash   = file_hash(file_bytes) if file_bytes else ""
current_params_hash = params_hash(tda_params)

# ── File info row ──────────────────────────────────────────────────────────────────
_fname   = st.session_state.cached_filename or (uploaded_file.name if uploaded_file else "(cached)")
_fsize   = f"{file_size_mb:.2f} MB"
_ffmt    = st.session_state.cached_filefmt or "CSV"
_fstatus = "✅" if uploaded_file is not None else "📂 (theme switch — cached)"

fc1, fc2, fc3, fc4 = st.columns([3, 1, 1, 1])
with fc1:
    st.success(f"{_fstatus} **{_fname}** ready", icon="📁")
with fc2:
    st.markdown(metric_card("Size", _fsize), unsafe_allow_html=True)
with fc3:
    st.markdown(metric_card("Format", _ffmt), unsafe_allow_html=True)
with fc4:
    tda_cached = (
        st.session_state.tda_file_hash == current_file_hash
        and st.session_state.tda_params_hash == current_params_hash
        and st.session_state.analysis_result is not None
    )
    cache_label = "Cached ✓" if tda_cached else "Ready"
    st.markdown(metric_card("TDA Status", cache_label), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Action buttons ─────────────────────────────────────────────────────────────
btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 2, 2, 1])
with btn_col2:
    st.markdown("<div class='btn-primary-wrap'>", unsafe_allow_html=True)
    if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True,
                 help="Run TDA + generate report"):
        st.session_state.trigger_full  = True
        st.session_state.trigger_regen = False
    st.markdown("</div>", unsafe_allow_html=True)
with btn_col3:
    st.markdown("<div class='btn-secondary-wrap'>", unsafe_allow_html=True)
    if st.button("🔄 Regenerate Report", use_container_width=True,
                 help="Keep TDA results, regenerate LLM report only",
                 disabled=not tda_cached):
        st.session_state.trigger_regen = True
        st.session_state.trigger_full  = False
    st.markdown("</div>", unsafe_allow_html=True)

# Read triggers and immediately reset to prevent double-fire on next rerun
run_full     = st.session_state.trigger_full
regen_report = st.session_state.trigger_regen
st.session_state.trigger_full  = False
st.session_state.trigger_regen = False

# ══════════════════════════════════════════════════════════════════════════════
# TDA Analysis pipeline (runs only when needed)
# ══════════════════════════════════════════════════════════════════════════════
STEPS = [
    ("📥", "Loading & validating signal"),
    ("🔬", "Building Takens delay embedding"),
    ("🌐", "Computing Weighted Rips complex"),
    ("📊", "Extracting cardiovascular metrics"),
    ("🎨", "Generating high-resolution plots"),
    ("🤖", "Generating LLM clinical report"),
]


def draw_steps(progress_placeholder, current: int, total_done: int):
    html = "<div style='padding:4px 0'>"
    for i, (icon, label) in enumerate(STEPS):
        if i < total_done:
            state = "done"
        elif i == current:
            state = "active"
        else:
            state = "pending"
        html += render_step(i + 1, icon, label, state)
    html += "</div>"
    progress_placeholder.markdown(html, unsafe_allow_html=True)


if run_full:
    session_id = uuid.uuid4().hex[:10]
    st.session_state.session_id = session_id

    # Save uploaded file (use cached bytes if uploader is empty after theme switch)
    _save_name = st.session_state.cached_filename or "signal.csv"
    tmp_path = config.UPLOAD_DIR / f"{session_id}_{_save_name}"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    st.markdown(f"<div class='section-header'>⏳ Analysis Progress</div>", unsafe_allow_html=True)
    progress_box = st.empty()

    draw_steps(progress_box, 0, 0)

    # ── TDA ──────────────────────────────────────────────────────────────────
    draw_steps(progress_box, 1, 1)
    result: SignalProcessingResult = process_signal_file(
        file_path=str(tmp_path),
        output_dir=str(config.OUTPUT_DIR),
        tda_params=tda_params,
        sampling_rate=100,
    )
    draw_steps(progress_box, 4, 4)

    try:
        tmp_path.unlink()
    except Exception:
        pass

    if not result.success:
        progress_box.empty()
        st.error(f"**Analysis failed:** {result.error_message}", icon="❌")
        st.stop()

    # Cache TDA result
    st.session_state.analysis_result  = result
    st.session_state.tda_file_hash    = current_file_hash
    st.session_state.tda_params_hash  = current_params_hash
    st.session_state.chat_history     = []

    # ── LLM report ────────────────────────────────────────────────────────────
    draw_steps(progress_box, 5, 5)
    _run_llm = True
    _result_for_llm = result

elif regen_report and tda_cached:
    _run_llm = True
    _result_for_llm = st.session_state.analysis_result
    st.markdown(f"<div class='section-header'>⏳ Regenerating Report</div>", unsafe_allow_html=True)
    progress_box = st.empty()
    draw_steps(progress_box, 5, 5)
else:
    _run_llm = False
    _result_for_llm = None
    progress_box = None

if _run_llm and _result_for_llm is not None:
    prompt_path = config.PROMPT_MAP.get(language, config.PROMPT_TEMPLATE_EN)
    llm_report = ""
    llm_error  = None
    effective_key = get_effective_api_key(env_key)

    try:
        user_prompt = build_llm_prompt(_result_for_llm, str(prompt_path))
        if not effective_key:
            llm_error = f"No API key configured for {provider_label}. Enter it in the sidebar."
        else:
            messages = [
                {"role": "system",  "content": config.LLM_SYSTEM_PROMPT},
                {"role": "user",    "content": user_prompt},
            ]
            llm_report = call_llm_openai_compat(messages, model_id, provider, effective_key)
    except Exception as e:
        llm_error = str(e)

    if progress_box:
        draw_steps(progress_box, 5, 6)
        time.sleep(0.3)
        progress_box.empty()

    st.session_state.llm_report = llm_report
    st.session_state.llm_error  = llm_error

# ══════════════════════════════════════════════════════════════════════════════
# Results — single-page fused layout with inline images
# ══════════════════════════════════════════════════════════════════════════════
result: SignalProcessingResult = st.session_state.get("analysis_result")
if result is None or not result.success:
    st.stop()

llm_report = st.session_state.get("llm_report", "")
llm_error  = st.session_state.get("llm_error")
summary    = result.summary
basic      = result.basic_signal
topo       = result.topology
anomaly    = result.anomaly
cardio     = result.cardiovascular
severity   = cardio.get("severity_distribution", {})

st.markdown("---")

# ── ① Overall status ──────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📋 Overall Assessment</div>", unsafe_allow_html=True)
sb1, sb2, sb3 = st.columns(3)
with sb1:
    _al_val = summary.get("anomaly_level", "N/A")
    _al_cls = badge_class(_al_val)
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Anomaly Level</div>"
        f"<div class='metric-value' style='font-size:1rem'>"
        f"<span class='badge {_al_cls}'>{_al_val}</span></div></div>",
        unsafe_allow_html=True,
    )
with sb2:
    _cv_val = summary.get("cardiovascular_status", "N/A")
    _cv_cls = badge_class(_cv_val)
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Cardiovascular Status</div>"
        f"<div class='metric-value' style='font-size:1rem'>"
        f"<span class='badge {_cv_cls}'>{_cv_val}</span></div></div>",
        unsafe_allow_html=True,
    )
with sb3:
    recs = summary.get("recommendations", [])
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>Top Recommendation</div>"
        f"<div class='metric-value' style='font-size:0.85rem;line-height:1.4'>"
        f"{recs[0] if recs else 'N/A'}</div></div>",
        unsafe_allow_html=True,
    )

# ── ② Basic signal metrics ────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📡 Basic Signal Metrics</div>", unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(metric_card("Sampling Rate", f"{basic.get('signal_frequency_hz','N/A')} Hz"), unsafe_allow_html=True)
with m2:
    st.markdown(metric_card("Duration", f"{basic.get('signal_duration_seconds',0):.1f} s"), unsafe_allow_html=True)
with m3:
    st.markdown(metric_card("Dominant Frequency", f"{basic.get('dominant_frequency_hz',0):.2f} Hz"), unsafe_allow_html=True)

# ── ③ Signal overview plot (inline) ──────────────────────────────────────────
if result.plot_glance_path:
    st.markdown("<div class='section-header'>📈 Signal Overview & Anomaly Detection</div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{T['text_muted']};font-size:0.88rem;margin-bottom:8px'>"
        "The three-panel chart below shows the raw input signal (top), the TDA-derived anomaly "
        "score at each time point (middle), and the binary anomaly indicator after applying the "
        "90th-percentile threshold (bottom). Highlighted regions correspond to segments where "
        "the topological structure deviates significantly from the baseline cycle geometry.</p>",
        unsafe_allow_html=True,
    )
    st.image(result.plot_glance_path, use_container_width=True)
    st.markdown(
        "<div class='img-caption'>Fig 1 — Input Signal · Anomaly Scores · Detection Indicator</div>",
        unsafe_allow_html=True,
    )

# ── ④ Topological metrics ─────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🌐 Topological Analysis</div>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{T['text_muted']};font-size:0.88rem;margin-bottom:8px'>"
    "Topological Data Analysis decomposes the signal into a sequence of delay-embedded cycles "
    "and measures their geometric persistence. The metrics below quantify how many cycles were "
    "detected and what fraction deviated from the expected topological signature.</p>",
    unsafe_allow_html=True,
)
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown(metric_card("Total Cycles", str(topo.get("total_cycles", 0))), unsafe_allow_html=True)
with t2:
    st.markdown(metric_card("Normal Cycles", str(topo.get("normal_cycles", 0)), color="#16a34a"), unsafe_allow_html=True)
with t3:
    st.markdown(metric_card("Anomaly Cycles", str(topo.get("anomaly_cycles", 0)), color="#dc2626"), unsafe_allow_html=True)
with t4:
    ratio_pct   = topo.get("anomaly_ratio", 0) * 100
    ratio_color = "#16a34a" if ratio_pct < 10 else ("#ca8a04" if ratio_pct < 30 else "#dc2626")
    st.markdown(metric_card("Anomaly Ratio", f"{ratio_pct:.1f}%", color=ratio_color), unsafe_allow_html=True)

# ── ⑤ TDA topology plot (inline) ─────────────────────────────────────────────
if result.plot_tda_path:
    st.markdown("<div class='section-header'>🔬 Topological Structure</div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{T['text_muted']};font-size:0.88rem;margin-bottom:8px'>"
        "The left panel shows the subsampled point cloud projected onto its first three principal "
        "components — a healthy periodic signal forms a closed loop, while anomalies distort or "
        "break this structure. The right panel is the persistence diagram: each point represents "
        "a topological feature (H₁ loop), and its distance above the diagonal indicates how "
        "long it persists — longer persistence implies a more robust, genuine cycle.</p>",
        unsafe_allow_html=True,
    )
    st.image(result.plot_tda_path, use_container_width=True)
    st.markdown(
        "<div class='img-caption'>Fig 2 — PCA Point Cloud (H₁) · Persistence Diagram</div>",
        unsafe_allow_html=True,
    )

# ── ⑥ Anomaly score metrics ───────────────────────────────────────────────────
st.markdown("<div class='section-header'>⚠️ Anomaly Score Distribution</div>", unsafe_allow_html=True)
a1, a2, a3, a4 = st.columns(4)
with a1:
    st.markdown(metric_card("Mean Score", f"{anomaly.get('mean_anomaly_score',0):.4f}"), unsafe_allow_html=True)
with a2:
    st.markdown(metric_card("Max Score", f"{anomaly.get('max_anomaly_score',0):.4f}", color="#dc2626"), unsafe_allow_html=True)
with a3:
    st.markdown(metric_card("Coverage (95th)", f"{anomaly.get('anomaly_coverage_95p_percent',0):.1f}%"), unsafe_allow_html=True)
with a4:
    st.markdown(metric_card("Anomaly Peaks", str(anomaly.get("anomaly_peak_count", 0))), unsafe_allow_html=True)

# ── ⑦ Cardiovascular metrics ─────────────────────────────────────────────────
st.markdown("<div class='section-header'>🫀 Cardiovascular Metrics</div>", unsafe_allow_html=True)
cv1, cv2, cv3, cv4 = st.columns(4)
with cv1:
    hr = cardio.get("estimated_heart_rate_bpm", 0)
    hr_color = "#16a34a" if 60 <= hr <= 100 else "#ca8a04"
    st.markdown(metric_card("Heart Rate", f"{hr:.0f} bpm", color=hr_color), unsafe_allow_html=True)
with cv2:
    st.markdown(metric_card("Mild Anomaly", f"{severity.get('mild_percent',0):.1f}%", color="#ca8a04"), unsafe_allow_html=True)
with cv3:
    st.markdown(metric_card("Moderate Anomaly", f"{severity.get('moderate_percent',0):.1f}%", color="#ea580c"), unsafe_allow_html=True)
with cv4:
    st.markdown(metric_card("Severe Anomaly", f"{severity.get('severe_percent',0):.1f}%", color="#dc2626"), unsafe_allow_html=True)

# ── ⑧ LLM Clinical Report (inline, between metrics and chat) ─────────────────
st.markdown("<div class='section-header'>📝 Clinical Report</div>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{T['text_muted']};font-size:0.88rem;margin-bottom:10px'>"
    f"Generated by <strong style='color:{T['accent']}'>{selected_model_name}</strong> "
    f"· Fudan University CardioDetector TDA System</p>",
    unsafe_allow_html=True,
)

if llm_error:
    st.warning(
        f"**LLM report unavailable:** {llm_error}",
        icon="⚠️",
    )
elif llm_report:
    st.markdown(f"<div class='report-box'>{llm_report}</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    dl1, dl2 = st.columns([1, 4])
    with dl1:
        st.download_button(
            "⬇️ Download Report",
            data=llm_report,
            file_name=f"cardio_report_{st.session_state.session_id}.md",
            mime="text/markdown",
        )
else:
    st.info("Run analysis or click **Regenerate Report** to generate the clinical report.", icon="ℹ️")

# ── Processing metadata ────────────────────────────────────────────────────────
st.markdown(
    f"<div style='color:{T['text_muted']};font-size:0.78rem;text-align:right;margin-top:6px'>"
    f"Daomiao.Wang@Fudan University &nbsp;|&nbsp; "
    f"Model: {selected_model_name} &nbsp;|&nbsp; "
    f"Session: {st.session_state.session_id}"
    f"</div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# Multi-turn Conversation
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<div class='section-header'>💬 Follow-up Conversation</div>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{T['text_muted']};font-size:0.88rem;margin-bottom:12px'>"
    "Ask follow-up questions about the analysis results. The assistant has full context "
    "of all metrics and the clinical report above.</p>",
    unsafe_allow_html=True,
)

ctx_json = json.dumps(result.to_context_dict(), indent=2, ensure_ascii=False)
CONV_SYSTEM = (
    "You are a cardiac signal analysis assistant developed at Fudan University. "
    "Answer questions accurately based on the following analysis context. "
    "Do not fabricate patient identities or unsupported clinical claims.\n\n"
    f"=== Analysis Results ===\n{ctx_json}\n\n"
    + (f"=== Clinical Report ===\n{llm_report}\n" if llm_report else "")
)

# Render chat history
for msg in st.session_state.chat_history:
    role    = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(
            f"<div class='chat-user'>"
            f"<div class='chat-role user'>You</div>{content}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='chat-assistant'>"
            f"<div class='chat-role assistant'>Assistant ({selected_model_name})</div>{content}</div>",
            unsafe_allow_html=True,
        )

# Chat input
chat_c1, chat_c2 = st.columns([5, 1])
with chat_c1:
    user_input = st.text_input(
        "chat_input",
        key="chat_input",
        placeholder="e.g. What does the anomaly ratio mean clinically? Is the heart rate normal?",
        label_visibility="collapsed",
    )
with chat_c2:
    send_btn = st.button("Send ➤", use_container_width=True)

if send_btn and user_input.strip():
    effective_key = get_effective_api_key(env_key)
    if not effective_key:
        st.warning(f"Please configure the {provider_label} API key in the sidebar.", icon="⚠️")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        messages_for_llm = [{"role": "system", "content": CONV_SYSTEM}]
        for msg in st.session_state.chat_history:
            messages_for_llm.append({"role": msg["role"], "content": msg["content"]})

        with st.spinner("Thinking..."):
            try:
                reply = call_llm_openai_compat(messages_for_llm, model_id, provider, effective_key)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"Error: {e}"}
                )
        st.rerun()

if st.session_state.chat_history:
    if st.button("🗑 Clear Conversation", use_container_width=False):
        st.session_state.chat_history = []
        st.rerun()
