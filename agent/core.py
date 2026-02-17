import os
import re
import ast
import functools
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import uuid
import yaml
import logging
import time
import sqlite3

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------
# Logging
# ---------------------------------------------
LOG_LEVEL = os.getenv("SAMARTH_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("agent.core")

# ---------------------------------------------
# Env & paths
# ---------------------------------------------
load_dotenv()

# Multiple API keys for automatic rotation
API_KEYS = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
]
# Filter out None values
API_KEYS = [k for k in API_KEYS if k]

if not API_KEYS:
    raise ValueError("No GROQ_API_KEY found in environment")

# Track which key is currently active
current_key_index = 0

DB_PATH = Path("data/processed/samarth_data.db").resolve()
AG_KB_PATH = Path("data/agronomy_kb.yaml")

_LAST_RESULT: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------
# LLM (Groq Native) with Auto-Rotation
# ---------------------------------------------
def get_client():
    """Get Groq client with current API key"""
    global current_key_index
    return Groq(api_key=API_KEYS[current_key_index])

client = get_client()

SYSTEM_CORE = (
    "You are Samarth AI, a world-class Senior Agricultural Intelligence & Data Science Expert. "
    "You provide enterprise-grade, professional, and deeply intelligent assistance. "
    "CORE RULES: "
    "1. CODE REQUESTS: Provide COMPLETE production-grade code with classes, error handling, docs. "
    "2. VISUALIZATION: NEVER give Python/matplotlib code. Output MULTIPLE ```json:chart blocks. "
    "3. ANALYTICS: When data is involved, ALWAYS include rich analysis WITH multiple charts. "
    "4. Tone: Professional, concise, insightful — like a senior data scientist at McKinsey. "
    "5. VISUAL THINKING: Don't just list numbers. PROVE IT with the *right* chart (Radar/Scatter/Composed). "
    "   - If the user asks for a comparison, AUTOMATICALLY generate a Radar Chart + Bar Chart. "
    "   - If the user asks for a trend, AUTOMATICALLY generate a Composed Chart (Line+Bar). "
)

def llm_invoke(messages: List[Dict[str, str]]) -> str:
    """Wrapper for Groq API call with automatic key rotation on rate limit"""
    global current_key_index, client
    
    # Try each API key
    for attempt in range(len(API_KEYS)):
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_tokens=8000,
                timeout=120
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            log.error(f"LLM Error (key {current_key_index + 1}): {error_str}")
            
            # Check if rate limit error
            if "rate_limit" in error_str.lower() or "429" in error_str:
                # Try next key if available
                if len(API_KEYS) > 1:
                    current_key_index = (current_key_index + 1) % len(API_KEYS)
                    client = get_client()
                    log.info(f"Switching to API key {current_key_index + 1}")
                    continue  # Try again with new key
                else:
                    return "⏳ **Rate Limit Reached**\n\nPlease wait 2-3 minutes and try again."
            
            return f"System Error: {error_str}"
    
    # All keys exhausted
    return "⏳ **All API Keys Rate Limited**\n\nAll available API keys have reached their limits. Please wait a few minutes and try again."

# ...


# ---------------------------------------------
# Database (internal only) - Native SQLite3
# ---------------------------------------------
if not DB_PATH.exists():
    log.warning("Database not found at %s. Analytics require ETL.", DB_PATH)

class SimpleSQLDatabase:
    def __init__(self, db_path):
        self.db_path = str(db_path)

    def run(self, sql_query):
        try:
            # Open in Read-Only mode for Vercel (Serverless filesystem is often RO)
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            conn.close()
            return str(results)
        except Exception as e:
            return f"Error: {e}"

db = SimpleSQLDatabase(DB_PATH)

# ---------------------------------------------
# Public API
# ---------------------------------------------

# ... (Previous helper functions for Regex/Planning remain same, just ensure imports match) ...

# [NOTE: Re-inserting the logic functions here to ensure complete file correctness]

def _tc(s: str) -> str:
    return " ".join(w.capitalize() for w in re.findall(r"[A-Za-z]+", s or ""))

def _normalize_place(txt: str) -> str:
    """Normalize 'Davangere District' -> 'Davangere'; trims extra spaces/case."""
    if not txt: return ""
    t = re.sub(r"\bdistrict\b", "", txt, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(w.capitalize() for w in t.split())

def rainfall_cte_for_district(district: str, state: Optional[str] = None) -> str:
    months = ",".join(f"'{m}'" for m in ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
    dq = _normalize_place(district).replace("'", "''")
    state_filter = ""
    if state:
        state_escaped = _normalize_place(state).replace("'", "''")
        state_filter = f" AND lower(trim(state)) = lower(trim('{state_escaped}'))"
    return (
        "norm AS (\n"
        "  SELECT\n"
        "    lower(trim(replace(replace(district,'District',''),'district',''))) AS dnorm,\n"
        "    state, month, rainfall_mm\n"
        "  FROM district_rainfall\n"
        "),\n"
        "rain AS (\n"
        "  SELECT\n"
        "    COALESCE(\n"
        "      MAX(CASE WHEN lower(month)='annual' THEN rainfall_mm END),\n"
        "      NULLIF(SUM(CASE WHEN lower(month) IN ("
        f"{months}"
        ") THEN COALESCE(rainfall_mm,0) END), 0)\n"
        "    ) AS annual_rainfall_mm\n"
        "  FROM norm\n"
        f"  WHERE dnorm = lower(trim('{dq}')){state_filter}\n"
        ")\n"
    )

def _years(text: str) -> Optional[str]:
    ys = re.findall(r"\b(?:19|20)\d{2}\b", text or "")
    if not ys: return None
    uniq = []
    for y in ys:
        if y not in uniq: uniq.append(y)
    return ", ".join(uniq[:8])

# Planners
def plan_district_annual_rainfall(q: str) -> Optional[Tuple[str, List[str]]]:
    m = re.search(r"rainfall.*?\bfor\b\s+([A-Za-z][A-Za-z\s\-]+)(?:\s+district)?(?:\s+of|\s*,\s*|\s+in)\s+([A-Za-z][A-Za-z\s\-]+)", q, re.IGNORECASE)
    if not m:
        m2 = re.search(r"rainfall.*?\bfor\b\s+([A-Za-z][A-Za-z\s\-]+)", q, re.IGNORECASE)
        if not m2: return None
        district = _tc(m2.group(1)); state = None
    else:
        district = _tc(m.group(1)); state = _tc(m.group(2))
    sql = f"WITH {rainfall_cte_for_district(district, state)} SELECT '{_normalize_place(district)}' AS district{(', ' + _normalize_place(state) + ' AS state') if state else ''}, annual_rainfall_mm FROM rain;"
    headers = ["district"] + (["state"] if state else []) + ["annual_rainfall_mm"]
    return sql, headers

def plan_compare_crop_years_with_rain(q: str) -> Optional[Tuple[str, List[str]]]:
    if not re.search(r"\bcompare\b", q, re.IGNORECASE) or not re.search(r"\bproduction\b", q, re.IGNORECASE): return None
    crop_m = re.search(r"production(?:\s+of)?\s+([A-Za-z][A-Za-z\s\-]+?)\s+in\b", q, re.IGNORECASE)
    dist_m = re.search(r"\bin\s+([A-Za-z][A-Za-z\s\-]+?)(?:\s+(?:district|state))?(?:\s+in|\s*,|\.)", q, re.IGNORECASE)
    y = _years(q)
    if not (crop_m and dist_m and y): return None
    crop = _tc(crop_m.group(1)); district = _tc(dist_m.group(1))
    sql = f"WITH prod AS ( SELECT year, SUM(production) AS total_production FROM crop_production WHERE district='{_normalize_place(district)}' AND crop='{_normalize_place(crop)}' AND year IN ({y}) GROUP BY year ), {rainfall_cte_for_district(district)} SELECT p.year, p.total_production, r.annual_rainfall_mm FROM prod p CROSS JOIN rain r ORDER BY p.year;"
    headers = ["year", "total_production", "annual_rainfall_mm"]
    return sql, headers

def plan_top_n_crops_district_year(q: str) -> Optional[Tuple[str, List[str]]]:
    m = re.search(r"\btop\s+(\d+)\s+crops?.*?\bin\s+([A-Za-z][A-Za-z\s\-]+?)\s+in\s+((?:19|20)\d{2})", q, re.IGNORECASE)
    if not m: return None
    n = max(1, min(int(m.group(1)), 20))
    district = _tc(m.group(2)); year = m.group(3)
    sql = f"WITH y AS ( SELECT crop, SUM(production) AS production FROM crop_production WHERE district='{_normalize_place(district)}' AND year={year} GROUP BY crop ) SELECT crop, production FROM y ORDER BY production DESC LIMIT {n};"
    headers = ["crop", "production"]
    return sql, headers

def plan_high_low_crop_state_year(q: str) -> Optional[Tuple[str, List[str]]]:
    m = re.search(r"district\s+in\s+([A-Za-z][A-Za-z\s\-]+)\s+with\s+the\s+highest\s+production\s+of\s+([A-Za-z][A-Za-z\s\-]+)\s+in\s+((?:19|20)\d{2}).*?district\s+in\s+([A-Za-z][A-Za-z\s\-]+)\s+with\s+the\s+lowest\s+non-zero\s+production.*?\3", q, re.IGNORECASE | re.DOTALL)
    if not m: return None
    state_hi = _tc(m.group(1)); crop = _tc(m.group(2)); year = m.group(3); state_lo = _tc(m.group(4))
    sql = f"WITH s AS ( SELECT state, district, SUM(production) AS production FROM crop_production WHERE crop='{_normalize_place(crop)}' AND year={year} AND state IN ('{_normalize_place(state_hi)}', '{_normalize_place(state_lo)}') GROUP BY state, district ), hi AS ( SELECT 'highest' AS which, state, district, production FROM s WHERE state='{_normalize_place(state_hi)}' ORDER BY production DESC LIMIT 1 ), lo AS ( SELECT 'lowest_non_zero' AS which, state, district, production FROM s WHERE state='{_normalize_place(state_lo)}' AND production>0 ORDER BY production ASC LIMIT 1 ) SELECT * FROM hi UNION ALL SELECT * FROM lo;"
    headers = ["which", "state", "district", "production"]
    return sql, headers

PLANNERS = [plan_district_annual_rainfall, plan_compare_crop_years_with_rain, plan_top_n_crops_district_year, plan_high_low_crop_state_year]

def smart_sql_plan(question: str) -> Optional[Tuple[str, List[str]]]:
    q = (question or "").strip()
    for planner in PLANNERS:
        p = planner(q)
        if p: return p
    return None

def _db_run(sql: str) -> str:
    return db.run(sql)

def _parse_rows(rows_str: str) -> List[Tuple]:
    s = rows_str.strip()
    if not s: return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list): return [tuple(x) if isinstance(x, (list, tuple)) else (str(x),) for x in obj]
    except: pass
    return []

def _set_last_result(session_id: str, sql: str, headers: List[str], rows: List[Tuple], latency_ms: int = 0) -> str:
    render_id = f"{session_id}-{uuid.uuid4()}"
    _LAST_RESULT[session_id] = {"sql": sql, "headers": headers, "rows": rows, "render_id": render_id, "latency_ms": latency_ms}
    return render_id

def get_last_result(session_id: str = "default") -> Dict[str, Any]:
    return _LAST_RESULT.get(session_id, {"sql": "", "headers": [], "rows": [], "render_id": f"{session_id}-{uuid.uuid4()}", "latency_ms": 0})

def _structured_is_valid(headers: List[str], rows: List[Tuple]) -> bool:
    if not headers or not rows: return False
    return True # Simplified for now

def _extract_context(q: str) -> Dict[str, str]:
    ctx = {}
    if m := re.search(r"\b(in|of)\s+([A-Za-z][A-Za-z\s\-]+)\s+district", q, re.IGNORECASE): ctx["district"] = _normalize_place(m.group(2))
    if m := re.search(r"\b(?:state\s+of|in)\s+([A-Za-z][A-Za-z\s\-]+)\b", q, re.IGNORECASE): ctx["state"] = _normalize_place(m.group(1))
    if m := re.search(r"\b(19|20)\d{2}\b", q): ctx["year"] = m.group(0)
    if m := re.search(r"\b(?:crop|production|of)\s+([A-Za-z][A-Za-z\s\-]+)\b", q, re.IGNORECASE): ctx["crop"] = _normalize_place(m.group(1))
    return ctx

def _synthesize_direct_answer(question: str, history: List[Dict[str, str]] = []) -> str:
    ctx = _extract_context(question)
    guide = (
        "You are Samarth AI, an elite AI Data Consultant. "
        "Your goal: Transform simple questions into comprehensive, boardroom-ready reports. "
        "\n"
        "RESPONSE STRUCTURE PROTOCOL (Follow Strictly): "
        "\n"
        "1. **EXECUTIVE SUMMARY** (2-3 lines): High-level answer with the most critical number/insight bolded. "
        "\n"
        "2. **DETAILED ANALYSIS** (Text & Tables): "
        "   - Break down the logic. "
        "   - Use Markdown Tables for ANY comparison (e.g., | Metric | Entity A | Entity B |). "
        "   - Use Bullet points with bold headers. "
        "\n"
        "3. STRATEGIC VISUALIZATION (AUTONOMOUS & INTELLIGENT): "
        "   - You have FULL AUTONOMY to decide the best visualization for the data. "
        "   - DO NOT just use 'bar' or 'line'. ANALYZE the data dimensions first: "
        "     * Time-Series + Volume? -> Use 'composed' (Bars for volume, Line for trend). "
        "     * Multi-Metric Comparison? -> Use 'radar' (e.g., comparing 5 districts on 3 metrics). "
        "     * Correlation/Outliers? -> Use 'scatter' (e.g., Rainfall vs Yield). "
        "     * Part-to-Whole? -> Use 'pie' or 'donut'. "
        "     * Performance/Efficiency? -> Use 'gauge' (KPIs). "
        "     * Density/Intensity? -> Use 'heatmap'. "
        "   - GENERATE 3-5 DIVERSE CHARTS that tell a complete story. "
        "   "
        "   CHART JSON FORMAT: "
        "   ```json:chart\n"
        '   {"type": "composed", "title": "Rainfall vs Yield Analysis", "xKey": "year", "data": [{"year": "2020", "rainfall": 1200, "yield": 4.5}]}\n'
        "   ``` "
        "\n"
        "4. **KEY TAKEAWAYS & RECOMMENDATIONS**: "
        "   - Bulleted list of actionable strategic advice. "
        "\n"
        "RULES: "
        "- NEVER output plain text blocks longer than 3 lines. Break it up! "
        "- BE CREATIVE with charts. If you see a correlation, show a Scatter plot. If you see a trend, show an Area chart. "
    )
    usr = f"{guide}\nRequest: {question}\nContext Extracted: {ctx}\n"
    
    # Construct memory-aware prompt
    msgs = [{"role": "system", "content": SYSTEM_CORE}]
    
    # Add history (limit last 10 turns)
    msgs.extend(history[-10:])
    
    msgs.append({"role": "user", "content": usr})
    
    return llm_invoke(msgs)

# ---------------------------------------------
# Public API Entry Point
# ---------------------------------------------
SMALLTALK_PAT = re.compile(r"^(hi|hii+|hello|hey|yo|sup|hola|namaste|namaskar)\W*$", re.IGNORECASE)
def is_smalltalk(q: str): return bool(SMALLTALK_PAT.match((q or "").strip()))

ANALYTICS_PAT = re.compile(r"\b(state|district|crop|season|production|area|rainfall\b|mm\b|year|kharif|rabi|rice|wheat|maize|millet|sugarcane|pulses|oilseed|yield|top\s*\d+|compare|trend|davangere|uttar\s*pradesh|tamil\s*nadu|karnataka|maharashtra|andhra\s*pradesh|anantapur)\b", re.IGNORECASE)
def is_analytics(q: str): return bool(ANALYTICS_PAT.search(q or ""))

def answer(question: str, history: List[Dict[str, str]] = [], session_id: str = "default") -> str:
    q = (question or "").strip()
    
    # Analytics check first (High precision)
    if is_analytics(q):
        plan = smart_sql_plan(q)
        if plan:
            sql, headers = plan
            rows = _parse_rows(_db_run(sql))
            if _structured_is_valid(headers, rows):
                _set_last_result(session_id, sql, headers, rows)
                return f"[[STRUCTURED_RESULT::{session_id}]]"

    # Fallback to synthesis
    return _synthesize_direct_answer(q, history)
