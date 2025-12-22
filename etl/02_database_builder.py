#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_database_builder.py
Builds SQLite DB with clean tables, indexes, and an annual rainfall view.
- Reads CSVs from data/raw/
- Normalizes text (strip/space collapse, title case for district/state, lower for month)
- Creates indexes
- Creates v_rainfall_annual view (safe to re-create)
Run:
    python etl/02_database_builder.py
"""

import csv
import os
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
DB = PROC / "samarth_data.db"

CROP_CSV = RAW / "crop_production.csv"
RAIN_CSV = None

# Accept either name user shared
cand = [
    RAW / "district_rainfall.csv",
    RAW / "district wise rainfall normal.csv",
    RAW / "district_wise_rainfall_normal.csv"
]
for c in cand:
    if c.exists():
        RAIN_CSV = c
        break

MONTHS = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

def _tclean(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = " ".join(s.split())
    return s

def _title(s: str) -> str:
    s = _tclean(s)
    if not s:
        return s
    # Keep initials like "N & M Andaman" usable: title-case but preserve & and acronyms shape.
    return " ".join([w.capitalize() if w.lower() not in {"&"} else w for w in s.split()])

def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    # tables
    cur.executescript("""
CREATE TABLE IF NOT EXISTS crop_production (
    state TEXT,
    district TEXT,
    year BIGINT,
    season TEXT,
    crop TEXT,
    area FLOAT,
    production FLOAT
);
CREATE TABLE IF NOT EXISTS district_rainfall (
    state TEXT,
    district TEXT,
    month TEXT,
    rainfall_mm FLOAT,
    source TEXT
);
CREATE TABLE IF NOT EXISTS __data_sources__ (
    table_name TEXT PRIMARY KEY,
    source_title TEXT NOT NULL,
    source_link TEXT NOT NULL
);
""")
    conn.commit()

def load_crop(conn: sqlite3.Connection):
    if not CROP_CSV.exists():
        raise FileNotFoundError(f"Missing {CROP_CSV}")
    cur = conn.cursor()
    cur.execute("DELETE FROM crop_production;")
    with open(CROP_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            state = _title(r.get("State_Name") or r.get("state") or "")
            district = _title(r.get("District_Name") or r.get("district") or "")
            year = r.get("Crop_Year") or r.get("year") or None
            season = _tclean(r.get("Season") or r.get("season") or "")
            crop = _title(r.get("Crop") or r.get("crop") or "")
            area = r.get("Area") or r.get("area") or None
            production = r.get("Production") or r.get("production") or None
            try:
                yv = int(year) if year not in (None, "") else None
            except Exception:
                yv = None
            try:
                av = float(area) if area not in (None, "") else None
            except Exception:
                av = None
            try:
                pv = float(production) if production not in (None, "") else None
            except Exception:
                pv = None
            rows.append((state, district, yv, season, crop, av, pv))
        cur.executemany(
            "INSERT INTO crop_production(state,district,year,season,crop,area,production) VALUES (?,?,?,?,?,?,?)",
            rows
        )
    conn.commit()

def load_rain(conn: sqlite3.Connection):
    if not RAIN_CSV or not RAIN_CSV.exists():
        raise FileNotFoundError("Missing district rainfall CSV (looked for district_rainfall.csv or 'district wise rainfall normal.csv')")
    cur = conn.cursor()
    cur.execute("DELETE FROM district_rainfall;")
    # Two possible layouts:
    # 1) Normalized (state,district,month,rainfall_mm,source)
    # 2) Wide CSV with 12 month columns and maybe 'ANNUAL' column
    with open(RAIN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldset = {c.lower() for c in reader.fieldnames or []}

        if {"state","district","month","rainfall_mm"}.issubset(fieldset):
            # Already normalized
            rows = []
            for r in reader:
                state = _title(r.get("state",""))
                district = _title(r.get("district",""))
                month = _tclean((r.get("month","") or "").lower())
                rainfall = r.get("rainfall_mm")
                src = _tclean(r.get("source","District Rainfall Normals (CSV)"))
                try:
                    val = float(rainfall) if rainfall not in (None,"") else None
                except Exception:
                    val = None
                rows.append((state, district, month, val, src))
            cur.executemany(
                "INSERT INTO district_rainfall(state,district,month,rainfall_mm,source) VALUES (?,?,?,?,?)",
                rows
            )
        else:
            # Wide format: expect columns like State, District, Jan, Feb, ..., Dec, Annual (optional)
            # try to find matches
            def find_col(*names):
                names_l = [n.lower() for n in names]
                for c in reader.fieldnames or []:
                    if c.lower() in names_l:
                        return c
                return None

            c_state = find_col("state","state name","state_name")
            c_district = find_col("district","district name","district_name")
            c_annual = None
            for cand in ["annual", "ann", "total", "sum", "year","year_total"]:
                if cand in fieldset:
                    c_annual = cand
                    break

            month_map = {}
            for m in MONTHS:
                # match many variants e.g. "JAN", "Jan.", "January"
                hit = None
                for col in reader.fieldnames or []:
                    l = col.lower().strip().replace(".","")
                    if l == m or l.startswith(m) or l in {m, m+"uary"}:
                        hit = col
                        break
                if hit:
                    month_map[m] = hit

            rows = []
            for r in reader:
                state = _title(r.get(c_state,"") if c_state else "")
                district = _title(r.get(c_district,"") if c_district else "")

                # monthly rows
                for m, col in month_map.items():
                    v = r.get(col)
                    try:
                        val = float(v) if v not in (None,"") else None
                    except Exception:
                        val = None
                    rows.append((state, district, m, val, "District Rainfall Normals (wide CSV)"))

                # annual row if present
                if c_annual:
                    v = r.get(c_annual)
                    try:
                        val = float(v) if v not in (None,"") else None
                    except Exception:
                        val = None
                    rows.append((state, district, "annual", val, "District Rainfall Normals (wide CSV)"))

            cur.executemany(
                "INSERT INTO district_rainfall(state,district,month,rainfall_mm,source) VALUES (?,?,?,?,?)",
                rows
            )
    conn.commit()

def create_indexes(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
CREATE INDEX IF NOT EXISTS idx_cp_district_year_crop ON crop_production(district,year,crop);
CREATE INDEX IF NOT EXISTS idx_cp_state_year_crop ON crop_production(state,year,crop);
CREATE INDEX IF NOT EXISTS idx_cp_crop_year ON crop_production(crop,year);
CREATE INDEX IF NOT EXISTS idx_rf_district_month ON district_rainfall(district,month);
CREATE INDEX IF NOT EXISTS idx_rf_state_district ON district_rainfall(state,district);
""")
    conn.commit()

def create_sources(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("DELETE FROM __data_sources__;")
    cur.executemany(
        "INSERT OR REPLACE INTO __data_sources__(table_name,source_title,source_link) VALUES (?,?,?)",
        [
            ("crop_production", "District-wise Crop Production (GitHub mirror of open dataset)",
             "https://raw.githubusercontent.com/srinivas-com/Data-Science-Projects/master/Crop_production_data/crop_production_data.csv"),
            ("district_rainfall", "District-wise Rainfall Normal (GitHub mirror of open dataset)",
             "https://raw.githubusercontent.com/airwarriorg/rainfall-analysis/master/district_rainfall_normal.csv"),
        ]
    )
    conn.commit()

def create_view(conn: sqlite3.Connection):
    """
    v_rainfall_annual: prefer an 'annual' record if present; else sum Jan..Dec.
    """
    cur = conn.cursor()
    cur.execute("DROP VIEW IF EXISTS v_rainfall_annual;")
    cur.execute(f"""
CREATE VIEW v_rainfall_annual AS
WITH monthly AS (
  SELECT state, district,
         SUM(CASE WHEN lower(month) IN ({",".join("'%s'"%m for m in MONTHS)}) THEN COALESCE(rainfall_mm,0) ELSE 0 END) AS monthly_sum,
         MAX(CASE WHEN lower(month)='annual' THEN rainfall_mm END) AS annual_row
  FROM district_rainfall
  GROUP BY state, district
)
SELECT state, district,
       COALESCE(annual_row, monthly_sum) AS annual_rainfall_mm
FROM monthly;
""")
    conn.commit()

def main():
    DB.unlink(missing_ok=True)
    conn = sqlite3.connect(DB)
    try:
        ensure_schema(conn)
        load_crop(conn)
        load_rain(conn)
        create_indexes(conn)
        create_sources(conn)
        create_view(conn)
        print(f"Built {DB}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
