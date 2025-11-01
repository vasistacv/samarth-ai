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
import shutil
import gzip
from zipfile import ZipFile

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase

# ---------------------------------------------
# Logging
# ---------------------------------------------
LOG_LEVEL = os.getenv("SAMARTH_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("agent.core")

# ---------------------------------------------
# Env (no hard dependency on .env in Cloud)
# ---------------------------------------------
load_dotenv()  # harmless if no .env
API_KEY = os.getenv("GROQ_API_KEY")  # Streamlit Secrets inject this

# ---------------------------------------------
# Paths & DB bootstrap (handles .zip / .gz)
# ---------------------------------------------
# We are flexible about where you placed the compressed DB:
#   - agent/data/processed/samarth_data.zip
#   - agent/data/samarth_data.zip
#   - data/processed/samarth_data.zip
#   - data/samarth_data.zip
BASE_DIR = Path(__file__).parent
CANDIDATE_DIRS: List[Path] = [
    BASE_DIR / "data" / "processed",
    BASE_DIR / "data",
    Path("data/processed"),
    Path("data"),
    BASE_DIR,              # last resorts
    Path("."),
]

DB_NAME = "samarth_data.db"
DB_BASENAME = DB_NAME  # final extracted filename

ZIP_CANDIDATES = [
    "samarth_data.zip",
    "samarth_data.db.zip",
]
GZ_CANDIDATES = [
    "samarth_data.gz",
    "samarth_data.db.gz",
]

def _ensure_db_present() -> Path:
    """Return absolute path to samarth_data.db. Inflate from .zip/.gz if needed."""
    found_db: Optional[Path] = None
    # 1) Already extracted?
    for d in CANDIDATE_DIRS:
        p = (d / DB_BASENAME).resolve()
        if p.exists() and p.is_file():
            found_db = p
            break

    if found_db:
        log.info("Using existing DB at %s", found_db)
        return found_db

    # 2) Try to find compressed archive and inflate
    def try_inflate_from_zip(zip_path: Path, target_dir: Path) -> Optional[Path]:
        if not zip_path.exists():
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            # Prefer exact file if present; else extract the first .db we see
            db_member = None
            if DB_BASENAME in names:
                db_member = DB_BASENAME
            else:
                for n in names:
                    if n.lower().endswith(".db"):
                        db_member = n
                        break
            if not db_member:
                raise RuntimeError(f"No .db file inside {zip_path.name}")
            zf.extract(db_member, path=target_dir)
            out = (target_dir / Path(db_member).name).resolve()
            log.info("Inflated DB from %s → %s", zip_path, out)
            return out

    def try_inflate_from_gz(gz_path: Path, target_dir: Path) -> Optional[Path]:
        if not gz_path.exists():
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        out = (target_dir / DB_BASENAME).resolve()
        with gzip.open(gz_path, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        log.info("Inflated DB from %s → %s", gz_path, out)
        return out

    # Search for a .zip/.gz to inflate
    for d in CANDIDATE_DIRS:
        # zip first
        for nm in ZIP_CANDIDATES:
            zp = (d / nm).resolve()
            try:
                out = try_inflate_from_zip(zp, d)
                if out and out.exists():
                    return out
            except Exception as e:
                log.warning("Zip inflate failed from %s: %s", zp, e)

        # gz next
        for nm in GZ_CANDIDATES:
            gp = (d / nm).resolve()
            try:
                out = try_inflate_from_gz(gp, d)
                if out and out.exists():
                    return out
            except Exception as e:
                log.warning("GZ inflate failed from %s: %s", gp, e)

    # 3) Still missing
    raise FileNotFoundError(
        "SQLite database not found. Place one of these in your repo:\n"
        f"- data/processed/{DB_BASENAME}\n"
        f"- data/{DB_BASENAME}\n"
        f"- agent/data/processed/{DB_BASENAME}\n"
        f"- agent/data/{DB_BASENAME}\n"
        "Or upload a compressed file alongside it: samarth_data.zip / samarth_data.db.zip "
        "or samarth_data.gz / samarth_data.db.gz (the app will auto-extract)."
    )

DB_PATH = _ensure_db_present()

# Keep a small in-memory cache for last structured result
_LAST_RESULT: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------
# LLM (friendly if key is missing)
# ---------------------------------------------
def _make_llm():
    if not API_KEY:
        class _Fallback:
            def invoke(self, msgs):
                # Short, neutral fallback if key isn’t set
                return type("Obj", (), {
                    "content": (
                        "Answer: Please add your GROQ_API_KEY in Streamlit → Settings → Secrets to enable AI answers.\n"
                        "- Why 1: The LLM requires an API key to generate explanations.\n"
                        "- Why 2: Keys are stored securely in Streamlit Secrets.\n"
                        "- Why 3: No .env is needed in the public repo."
                    )
                })()
        log.warning("GROQ_API_KEY not found in environment; using fallback stub.")
        return _Fallback()

    last_err = None
    for model in ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"]:
        try:
            return ChatGroq(
                model_name=model,
                temperature=0.15,
                max_tokens=900,
                request_timeout=45,
                groq_api_key=API_KEY,
            )
        except Exception as e:
            last_err = e
            time.sleep(0.4)
    raise RuntimeError(f"LLM init failed: {last_err}")

llm = _make_llm()

SYSTEM_CORE = (
    "Answer the user directly. Use concrete numbers with units where sensible. "
    "Prefer compact bullets or a short paragraph. Avoid negative phrasing or apologies. "
    "If exact figures are unavailable, provide an informed estimate with units and a one-line rationale. "
    "Keep answers crisp and helpful."
)

# ---------------------------------------------
# Database handle (now that DB_PATH exists)
# ---------------------------------------------
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# ---------------------------------------------
# Units (for synthesis hints; UI also formats)
# ---------------------------------------------
UNIT_SUFFIX = {
    "annual_rainfall_mm": " mm",
    "rainfall_mm": " mm",
    "area": " ha",
    "production": " t",
    "total_production": " t",
    "yield": " kg/ha",
}

# ---------------------------------------------
# Lightweight agronomy KB (optional file)
# ---------------------------------------------
AG_KB_PATH = Path("data/agronomy_kb.yaml")

@functools.lru_cache(maxsize=1)
def load_ag_kb() -> Dict[str, Any]:
    if not AG_KB_PATH.exists():
        return {"meta": {}, "crops": {}}
    with open(AG_KB_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {"meta": {}, "crops": {}}
        return data

def kb_lookup_rainfall(crop_name: str) -> Dict[str, Any]:
    kb = load_ag_kb().get("crops", {})
    key = None
    q = (crop_name or "").strip().lower()
    for k, v in kb.items():
        names = [k] + [*(v.get("common_names", []) or [])]
        if any(q == n.lower() for n in names):
            key = k
            break
    if not key:
        for k, v in kb.items():
            names = [k] + [*(v.get("common_names", []) or [])]
            if any(q in n.lower() or n.lower() in q for n in names):
                key = k
                break
    return {"crop": key, "data": kb.get(key, {})}

# ---------------------------------------------
# Router (small-talk / analytics / practice / GK)
# ---------------------------------------------
SMALLTALK_PAT = re.compile(r"^(hi|hii+|hello|hey|yo|sup|hola|namaste|namaskar)\W*$", re.IGNORECASE)
ANALYTICS_PAT = re.compile(
    r"\b(state|district|crop|season|production|area|rainfall\b|mm\b|year|kharif|rabi|"
    r"rice|wheat|maize|millet|sugarcane|pulses|oilseed|yield|top\s*\d+|compare|trend|davangere|uttar\s*pradesh|tamil\s*nadu|karnataka|maharashtra|andhra\s*pradesh|anantapur)\b",
    re.IGNORECASE,
)
PRACTICE_PAT = re.compile(
    r"\b(require(d)?|optimum|optimal|minimum|recommended|needed|rainfed|irrigated|"
    r"temperature|soil|sowing|spacing|fertiliz(e|er)|package of practices|variet(y|ies))\b",
    re.IGNORECASE,
)

def is_smalltalk(q: str) -> bool:
    return bool(SMALLTALK_PAT.match((q or "").strip()))

def is_analytics(q: str) -> bool:
    return bool(ANALYTICS_PAT.search(q or ""))

def is_practice(q: str) -> bool:
    return bool(PRACTICE_PAT.search(q or ""))

# ---------------------------------------------
# DB helpers
# ---------------------------------------------
def _db_run(sql: str) -> str:
    return db.run(sql)

def _parse_rows(rows_str: str) -> List[Tuple]:
    s = rows_str.strip()
    if not s:
        return []
    if s.startswith("[") or s.startswith("("):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, tuple):
                obj = [obj]
            out = []
            for r in obj:
                if isinstance(r, (list, tuple)):
                    out.append(tuple(r))
                else:
                    out.append((str(r),))
            return out
        except Exception:
            pass
    out = []
    for ln in [x for x in s.splitlines() if x.strip()]:
        m = re.match(r"\((.*)\)", ln.strip())
        if m:
            parts = [p.strip() for p in m.group(1).split(",")]
            out.append(tuple([p.strip("' ").strip('" ') for p in parts]))
        else:
            parts = [p.strip() for p in re.split(r"\t|\|", ln)]
            out.append(tuple(parts))
    return out

def _set_last_result(session_id: str, sql: str, headers: List[str], rows: List[Tuple], latency_ms: int = 0) -> str:
    render_id = f"{session_id}-{uuid.uuid4()}"
    _LAST_RESULT[session_id] = {"sql": sql, "headers": headers, "rows": rows, "render_id": render_id, "latency_ms": latency_ms}
    return render_id

def get_last_result(session_id: str) -> Dict[str, Any]:
    return _LAST_RESULT.get(session_id, {"sql": "", "headers": [], "rows": [], "render_id": f"{session_id}-{uuid.uuid4()}", "latency_ms": 0})

# ---------------------------------------------
# SQL helpers
# ---------------------------------------------
def _normalize_place(txt: str) -> str:
    """Normalize 'Davangere District' -> 'Davangere'; trims extra spaces/case."""
    if not txt:
        return ""
    t = re.sub(r"\bdistrict\b", "", txt, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(w.capitalize() for w in t.split())

def rainfall_cte_for_district(district: str, state: Optional[str] = None) -> str:
    """Match by normalized district; prefer annual, otherwise sum months."""
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

def _tc(s: str) -> str:
    return " ".join(w.capitalize() for w in re.findall(r"[A-Za-z]+", s or ""))

def _years(text: str) -> Optional[str]:
    ys = re.findall(r"\b(?:19|20)\d{2}\b", text or "")
    if not ys:
        return None
    uniq = []
    for y in ys:
        if y not in uniq:
            uniq.append(y)
    return ", ".join(uniq[:8])

# ---------------------------------------------
# Deterministic planners
# ---------------------------------------------
def plan_district_annual_rainfall(q: str) -> Optional[Tuple[str, List[str]]]:
    m = re.search(r"rainfall.*?\bfor\b\s+([A-Za-z][A-Za-z\s\-]+)(?:\s+district)?(?:\s+of|\s*,\s*|\s+in)\s+([A-Za-z][A-Za-z\s\-]+)", q, re.IGNORECASE)
    if not m:
        m2 = re.search(r"rainfall.*?\bfor\b\s+([A-Za-z][A-Za-z\s\-]+)", q, re.IGNORECASE)
        if not m2:
            return None
        district = _tc(m2.group(1)); state = None
    else:
        district = _tc(m.group(1)); state = _tc(m.group(2))
    sql = f"""
WITH {rainfall_cte_for_district(district, state)}
SELECT '{_normalize_place(district)}' AS district{(", '" + _normalize_place(state) + "' AS state" ) if state else ""},
       annual_rainfall_mm
FROM rain;
""".strip()
    headers = ["district"] + (["state"] if state else []) + ["annual_rainfall_mm"]
    return sql, headers

def plan_compare_crop_years_with_rain(q: str) -> Optional[Tuple[str, List[str]]]:
    if not re.search(r"\bcompare\b", q, re.IGNORECASE):
        return None
    if not re.search(r"\bproduction\b", q, re.IGNORECASE):
        return None
    crop_m = re.search(r"production(?:\s+of)?\s+([A-Za-z][A-Za-z\s\-]+?)\s+in\b", q, re.IGNORECASE)
    dist_m = re.search(r"\bin\s+([A-Za-z][A-Za-z\s\-]+?)(?:\s+(?:district|state))?(?:\s+in|\s*,|\.)", q, re.IGNORECASE)
    y = _years(q)
    if not (crop_m and dist_m and y):
        return None
    crop = _tc(crop_m.group(1)); district = _tc(dist_m.group(1))
    sql = f"""
WITH prod AS (
  SELECT year, SUM(production) AS total_production
  FROM crop_production
  WHERE district='{_normalize_place(district)}' AND crop='{_normalize_place(crop)}' AND year IN ({y})
  GROUP BY year
),
{rainfall_cte_for_district(district)}
SELECT p.year AS year,
       p.total_production AS total_production,
       r.annual_rainfall_mm AS annual_rainfall_mm
FROM prod p
CROSS JOIN rain r
ORDER BY p.year;
""".strip()
    headers = ["year", "total_production", "annual_rainfall_mm"]
    return sql, headers

def plan_top_n_crops_district_year(q: str) -> Optional[Tuple[str, List[str]]]:
    m = re.search(r"\btop\s+(\d+)\s+crops?.*?\bin\s+([A-Za-z][A-Za-z\s\-]+?)\s+in\s+((?:19|20)\d{2})", q, re.IGNORECASE)
    if not m:
        return None
    n = max(1, min(int(m.group(1)), 20))
    district = _tc(m.group(2)); year = m.group(3)
    sql = f"""
WITH y AS (
  SELECT crop, SUM(production) AS production
  FROM crop_production
  WHERE district='{_normalize_place(district)}' AND year={year}
  GROUP BY crop
)
SELECT crop, production
FROM y
ORDER BY production DESC
LIMIT {n};
""".strip()
    headers = ["crop", "production"]
    return sql, headers

def plan_high_low_crop_state_year(q: str) -> Optional[Tuple[str, List[str]]]:
    m = re.search(
        r"district\s+in\s+([A-Za-z][A-Za-z\s\-]+)\s+with\s+the\s+highest\s+production\s+of\s+([A-Za-z][A-Za-z\s\-]+)\s+in\s+((?:19|20)\d{2}).*?district\s+in\s+([A-Za-z][A-Za-z\s\-]+)\s+with\s+the\s+lowest\s+non-zero\s+production.*?\3",
        q, re.IGNORECASE | re.DOTALL
    )
    if not m:
        return None
    state_hi = _tc(m.group(1)); crop = _tc(m.group(2)); year = m.group(3); state_lo = _tc(m.group(4))
    sql = f"""
WITH s AS (
  SELECT state, district, SUM(production) AS production
  FROM crop_production
  WHERE crop='{_normalize_place(crop)}' AND year={year} AND state IN ('{_normalize_place(state_hi)}', '{_normalize_place(state_lo)}')
  GROUP BY state, district
),
hi AS (
  SELECT 'highest' AS which, state, district, production
  FROM s WHERE state='{_normalize_place(state_hi)}'
  ORDER BY production DESC LIMIT 1
),
lo AS (
  SELECT 'lowest_non_zero' AS which, state, district, production
  FROM s WHERE state='{_normalize_place(state_lo)}' AND production>0
  ORDER BY production ASC LIMIT 1
)
SELECT * FROM hi
UNION ALL
SELECT * FROM lo;
""".strip()
    headers = ["which", "state", "district", "production"]
    return sql, headers

PLANNERS = [
    plan_district_annual_rainfall,
    plan_compare_crop_years_with_rain,
    plan_top_n_crops_district_year,
    plan_high_low_crop_state_year,
]

def smart_sql_plan(question: str) -> Optional[Tuple[str, List[str]]]:
    q = (question or "").strip()
    for planner in PLANNERS:
        p = planner(q)
        if p:
            return p
    return None

# ---------------------------------------------
# Structured rows sanitizer (avoid NaN leaks)
# ---------------------------------------------
def _structured_is_valid(headers: List[str], rows: List[Tuple]) -> bool:
    if not headers or not rows:
        return False
    # If rainfall present, ensure at least one non-null numeric
    if "annual_rainfall_mm" in headers:
        idx = headers.index("annual_rainfall_mm")
        has_val = any(
            (r[idx] is not None) and (str(r[idx]).strip().lower() not in {"", "null", "none"})
            for r in rows
        )
        if not has_val:
            return False
    # If production present, ensure some non-null
    for key in ("production", "total_production"):
        if key in headers:
            idx = headers.index(key)
            if not any((r[idx] is not None) and (str(r[idx]).strip().lower() not in {"", "null", "none"}) for r in rows):
                return False
    return True

# ---------------------------------------------
# AI synthesis helpers
# ---------------------------------------------
def _extract_context(q: str) -> Dict[str, str]:
    ctx = {}
    m = re.search(r"\b(in|of)\s+([A-Za-z][A-Za-z\s\-]+)\s+district", q, re.IGNORECASE)
    if m: ctx["district"] = _normalize_place(m.group(2))
    m = re.search(r"\b(?:state\s+of|in)\s+([A-Za-z][A-Za-z\s\-]+)\b", q, re.IGNORECASE)
    if m: ctx["state"] = _normalize_place(m.group(1))
    m = re.search(r"\b(19|20)\d{2}\b", q)
    if m: ctx["year"] = m.group(0)
    m = re.search(r"\b(?:crop|production|of)\s+([A-Za-z][A-Za-z\s\-]+)\b", q, re.IGNORECASE)
    if m: ctx["crop"] = _normalize_place(m.group(1))
    return ctx

def _synthesize_direct_answer(question: str) -> str:
    ctx = _extract_context(question)
    guide = (
        "Produce a direct answer first line. Then 2–4 concise bullets that justify the number or statement. "
        "Use SI units (mm, ha, t, kg/ha, °C). If quantity uncertain, give a tight plausible range with units."
    )
    usr = (
        f"{guide}\n"
        f"Question: {question}\n"
        f"Context: {ctx}\n"
        "Output format:\n"
        "Answer: <one line with numbers+units>\n"
        "- Why 1\n- Why 2\n- Why 3"
    )
    msgs = [{"role":"system","content":SYSTEM_CORE}, {"role":"user","content":usr}]
    return llm.invoke(msgs).content

# ---------------------------------------------
# Public API
# ---------------------------------------------
def answer(question: str, session_id: str) -> str:
    q = (question or "").strip()

    # 1) Small-talk
    if is_smalltalk(q):
        return (
            "Hey! I’m **Samarth AI** — ask me agriculture, rainfall, or GK. "
            "Try: *“Annual rainfall for Davangere district”* or *“Top 5 crops in Davangere in 2015.”*"
        )

    # 2) Practice KB (rainfall guidance for a crop)
    if is_practice(q):
        return _synthesize_direct_answer(q)

    # 3) Analytics (tables first, then synth fallback)
    if is_analytics(q):
        plan = smart_sql_plan(q)
        if plan:
            sql, headers = plan
            rows = _parse_rows(_db_run(sql))

            # Guard: if key columns are null-ish, force synth instead of NaN in UI
            if not _structured_is_valid(headers, rows):
                return _synthesize_direct_answer(q)

            _set_last_result(session_id, sql, headers, rows, latency_ms=0)
            return f"[[STRUCTURED_RESULT::{session_id}]]"

        # No planner hit → synth
        return _synthesize_direct_answer(q)

    # 4) General knowledge synthesis
    return _synthesize_direct_answer(q)
