import re
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from agent.core import answer, get_last_result

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Samarth AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------
# Styles (light, professional)
# -------------------------------------------------
st.markdown(
    """
    <style>
      :root{
        --bg:#f7f7fb;
        --paper:#ffffff;
        --ink:#0a0f1a;
        --mut:#5b6472;
        --line:#e7e9ee;
        --steel:#334155;
        --accentGrad:linear-gradient(135deg,#1d4ed8,#0ea5e9);
        --chip:#f2f5fb;
      }
      .stApp{ background:var(--bg); color:var(--ink) }
      header{ visibility:hidden }
      .topbar{
        position:sticky; top:0; z-index:1000;
        display:flex; align-items:center; gap:12px;
        padding:12px 18px; background:var(--paper);
        border-bottom:1px solid var(--line);
      }
      .brand{ display:flex; align-items:center; gap:12px; font-weight:800; letter-spacing:.2px; }
      .logo{ width:32px;height:32px;border-radius:10px; display:grid;place-items:center; background:var(--accentGrad); color:#fff; font-weight:900; }
      .subtitle{ color:var(--steel); font-weight:600; font-size:13px; }
      .wrap{ max-width: 1100px; margin: 0 auto; padding: 12px 16px 24px; }
      .hero{ background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:18px 18px 10px; margin-top:12px; }
      .hero h1{ margin:0; font-size:22px }
      .hero p{ margin:8px 0 0; color:var(--steel) }
      .chips{ display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 0 }
      .chip{ padding:8px 12px; border-radius:999px; background:var(--chip); border:1px solid var(--line); cursor:pointer; font-size:12px; color:#0b1220; }
      .chip:hover{ background:#eaf0fc }
      .main{ display:grid; grid-template-columns: 1fr; gap:14px; margin-top:12px; }
      .pane{ background:var(--paper); border:1px solid var(--line); border-radius:16px; }
      .chat{ padding:12px 16px 6px; }
      .msg{ display:flex; gap:12px; padding:14px 0; border-bottom:1px solid var(--line) }
      .msg:last-child{ border-bottom:0 }
      .ava{ width:34px;height:34px;border-radius:10px; display:grid; place-items:center; background:#e9eefc; color:#0a1b3a; font-weight:800; flex-shrink:0 }
      .ava.ai{ background:#e6fbff; color:#072033 }
      .bubble{ background:#f9fbff; border:1px solid var(--line); border-radius:12px; padding:12px 14px; }
      .meta{ font-size:12px; color:var(--mut); margin-top:6px }
      .footer{ text-align:center; color:#495569; padding:10px 0 0; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Top bar + hero
# -------------------------------------------------
st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="logo">S</div>
        <div>Samarth AI</div>
        <div class="subtitle"></div>
      </div>
    </div>
    <div class="wrap">
      <div class="hero">
        <h1>Samarth AI</h1>
        <p>
          Answers agricultural, rainfall, and general knowledge questions instantly with precise data, AI reasoning, and clear unit-based explanations.
        </p>
        <div class="chips">
          <span class="chip" onclick="navigator.clipboard.writeText('Top 5 crops in Davangere in 2015')">Top 5 crops in Davangere in 2015</span>
          <span class="chip" onclick="navigator.clipboard.writeText('Compare production of Rice in Karnataka in 2010 and 2020')">Compare Rice: KA 2010 vs 2020</span>
          <span class="chip" onclick="navigator.clipboard.writeText('Annual rainfall for Davangere district of Karnataka')">Annual rainfall Davangere</span>
          <span class="chip" onclick="navigator.clipboard.writeText('Optimal rainfall for wheat stages')">Wheat rainfall stages</span>
        </div>
      </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Minimal in-memory history (no session panel)
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

SESSION_ID = "ui"  # stable key for last-result cache

# -------------------------------------------------
# Units + safe formatting
# -------------------------------------------------
UNIT_SUFFIX = {
    "annual_rainfall_mm": " mm",
    "rainfall_mm": " mm",
    "area": " ha",
    "production": " t",
    "total_production": " t",
    "yield": " kg/ha",
}

def fmt_with_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in UNIT_SUFFIX:
            suf = UNIT_SUFFIX[c]
            # replace NaN with "—" before formatting
            out[c] = out[c].where(pd.notna(out[c]), other="—")
            # only apply numeric formatting where numeric
            mask_num = pd.to_numeric(out[c], errors="coerce").notna()
            out.loc[mask_num, c] = pd.to_numeric(out.loc[mask_num, c])
            out.loc[mask_num, c] = out.loc[mask_num, c].map(lambda v: f"{v:,.0f}{suf}" if abs(float(v)) >= 1000 else f"{float(v):.2f}{suf}")
            # non-numeric (including "—") just append suffix if it's a bare number-string
            mask_non = ~mask_num
            out.loc[mask_non, c] = out.loc[mask_non, c].astype(str).map(lambda s: s if s == "—" else (s + suf if not s.endswith(suf) else s))
    return out

def looks_structured(text: str) -> bool:
    return text.strip().startswith("[[STRUCTURED_RESULT::")

# -------------------------------------------------
# Chat pane
# -------------------------------------------------
st.markdown('<div class="main"><div class="pane">', unsafe_allow_html=True)
st.markdown('<div class="chat">', unsafe_allow_html=True)

# replay chat history
for m in st.session_state.messages:
    role = m["role"]; content = m["content"]; ts = m.get("ts","")
    ava = "👤" if role == "user" else "Σ"
    ava_cls = "" if role == "user" else " ai"
    st.markdown(f'<div class="msg"><div class="ava{ava_cls}">{ava}</div><div style="flex:1">', unsafe_allow_html=True)
    if role == "assistant" and looks_structured(content):
        st.markdown('<div class="bubble">Structured result:</div>', unsafe_allow_html=True)
        meta = get_last_result(SESSION_ID)
        headers, rows = meta.get("headers", []), meta.get("rows", [])
        if headers and rows:
            df = pd.DataFrame(rows, columns=headers)
            # best-effort numeric coercion without losing "—"
            for c in df.columns:
                try:
                    df[c] = pd.to_numeric(df[c])
                except Exception:
                    pass
            show = fmt_with_units(df)
            st.dataframe(show, use_container_width=True, hide_index=True)
            if "year" in df.columns and len(df) > 1:
                num_cols = [c for c in df.columns if c != "year" and pd.api.types.is_numeric_dtype(df[c])]
                if num_cols:
                    st.line_chart(df.set_index("year")[num_cols])
    else:
        st.markdown(f'<div class="bubble">{content}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="meta">{ts}</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# input
prompt = st.chat_input("Type your message")
if prompt:
    now = datetime.now().strftime("%d %b %Y, %H:%M")

    # show user turn immediately (first-turn visibility)
    with st.chat_message("user"):
        st.markdown(f'<div class="bubble">{prompt}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role":"user","content":prompt,"ts":now})

    # answer
    with st.chat_message("assistant"):
        out = answer(prompt, session_id=SESSION_ID)
        if looks_structured(out):
            st.markdown('<div class="bubble">Structured result:</div>', unsafe_allow_html=True)
            meta = get_last_result(SESSION_ID)
            headers, rows = meta.get("headers", []), meta.get("rows", [])
            if headers and rows:
                df = pd.DataFrame(rows, columns=headers)
                for c in df.columns:
                    try: df[c] = pd.to_numeric(df[c])
                    except Exception: pass
                show = fmt_with_units(df)
                st.dataframe(show, use_container_width=True, hide_index=True)
                if "year" in df.columns and len(df) > 1:
                    num_cols = [c for c in df.columns if c != "year" and pd.api.types.is_numeric_dtype(df[c])]
                    if num_cols:
                        st.line_chart(df.set_index("year")[num_cols])
        else:
            st.markdown(f'<div class="bubble">{out}</div>', unsafe_allow_html=True)

    # store assistant turn
    st.session_state.messages.append({"role":"assistant","content":out,"ts":now})

st.markdown('</div>', unsafe_allow_html=True)  # chat
st.markdown('</div></div>', unsafe_allow_html=True)  # pane + main

st.markdown('<div class="footer">Created by <b>Vashista C V</b></div></div>', unsafe_allow_html=True)
