"""
Builds sql_load/: SQL Server-ready CSVs + T-SQL scripts (create, bulk load, checks)
from one schema definition, and validates every value against its declared type.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal

SRC = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "sql_load"
(OUT / "data").mkdir(parents=True, exist_ok=True)
DB = "SaleAnalytics"
SCHEMA = "raw"

# table -> (columns [(name, sql_type, nullable)], primary key, [indexes])
T = {
 "fx_rates": ([("currency", "CHAR(3)", 0), ("rate_to_eur", "DECIMAL(9,4)", 0)], ["currency"], []),
 "sales_reps": ([("rep_id", "VARCHAR(18)", 0), ("rep_name", "NVARCHAR(100)", 0), ("role", "VARCHAR(40)", 0),
                 ("segment", "VARCHAR(20)", 0), ("region", "VARCHAR(30)", 0), ("manager_id", "VARCHAR(18)", 1),
                 ("hire_date", "DATE", 0), ("termination_date", "DATE", 1), ("is_active", "BIT", 0)], ["rep_id"], []),
 "quotas": ([("rep_id", "VARCHAR(18)", 0), ("fiscal_quarter", "CHAR(7)", 0), ("quota_start_date", "DATE", 0),
             ("ramp_pct", "DECIMAL(4,2)", 0), ("quota_eur", "DECIMAL(18,2)", 0)], ["rep_id", "fiscal_quarter"], []),
 "accounts": ([("account_id", "VARCHAR(18)", 0), ("account_name", "NVARCHAR(200)", 0), ("website_domain", "VARCHAR(100)", 0),
               ("industry", "NVARCHAR(100)", 1), ("employee_count", "INT", 1), ("segment", "VARCHAR(20)", 0),
               ("billing_country", "NVARCHAR(60)", 0), ("region", "VARCHAR(30)", 0), ("annual_revenue_eur", "DECIMAL(18,2)", 0),
               ("created_date", "DATE", 0)], ["account_id"], []),
 "workspaces": ([("workspace_id", "VARCHAR(20)", 0), ("account_id", "VARCHAR(18)", 1), ("signup_date", "DATE", 0),
                 ("hosting_type", "VARCHAR(30)", 0), ("signup_source", "VARCHAR(40)", 0), ("telemetry_enabled", "BIT", 0),
                 ("owner_email_domain", "VARCHAR(100)", 0), ("is_active", "BIT", 0), ("churned_month", "DATE", 1)],
                ["workspace_id"], [["account_id"]]),
 "workspace_usage_monthly": ([("month", "DATE", 0), ("workspace_id", "VARCHAR(20)", 0), ("plan", "VARCHAR(30)", 0),
                              ("executions", "BIGINT", 1), ("active_workflows", "INT", 1), ("active_users", "INT", 1),
                              ("ai_node_executions", "BIGINT", 1), ("distinct_integrations_used", "INT", 1),
                              ("failed_execution_rate", "DECIMAL(6,4)", 1), ("self_serve_mrr_eur", "DECIMAL(10,2)", 0)],
                             ["workspace_id", "month"], [["month"]]),
 "leads": ([("lead_id", "VARCHAR(18)", 0), ("created_date", "DATE", 0), ("lead_source", "VARCHAR(50)", 0),
            ("account_id", "VARCHAR(18)", 1), ("workspace_id", "VARCHAR(20)", 1), ("email_domain", "VARCHAR(100)", 1),
            ("pql_rule_v1_flag", "BIT", 0), ("status", "VARCHAR(50)", 0), ("converted_opportunity_id", "VARCHAR(18)", 1),
            ("converted_date", "DATE", 1), ("owner_id", "VARCHAR(18)", 1)],
           ["lead_id"], [["account_id"], ["workspace_id"], ["converted_opportunity_id"]]),
 "opportunities": ([("opportunity_id", "VARCHAR(18)", 0), ("opportunity_name", "NVARCHAR(200)", 0), ("account_id", "VARCHAR(18)", 0),
                    ("opportunity_type", "VARCHAR(30)", 0), ("lead_source", "VARCHAR(50)", 1), ("owner_id", "VARCHAR(18)", 0),
                    ("created_date", "DATE", 0), ("close_date", "DATE", 0), ("stage_name", "VARCHAR(40)", 0),
                    ("forecast_category", "VARCHAR(20)", 1), ("amount", "DECIMAL(18,2)", 1), ("currency", "CHAR(3)", 0),
                    ("is_closed", "BIT", 0), ("is_won", "BIT", 0), ("loss_reason", "VARCHAR(60)", 1),
                    ("primary_competitor", "VARCHAR(40)", 1), ("primary_workspace_id", "VARCHAR(20)", 1),
                    ("next_step", "NVARCHAR(200)", 1), ("last_activity_date", "DATE", 0)],
                   ["opportunity_id"], [["account_id"], ["owner_id"], ["close_date"]]),
 "opportunity_field_history": ([("history_id", "VARCHAR(18)", 0), ("opportunity_id", "VARCHAR(18)", 0), ("field", "VARCHAR(40)", 0),
                                ("old_value", "NVARCHAR(255)", 1), ("new_value", "NVARCHAR(255)", 1), ("created_by", "VARCHAR(18)", 0),
                                ("created_at", "DATETIME2(0)", 0)],
                               ["history_id"], [["opportunity_id", "field", "created_at"]]),
 "forecast_submissions": ([("submission_week", "DATE", 0), ("fiscal_quarter", "CHAR(7)", 0), ("rep_id", "VARCHAR(18)", 0),
                           ("closed_won_qtd_eur", "DECIMAL(18,2)", 0), ("commit_call_eur", "DECIMAL(18,2)", 0),
                           ("best_case_call_eur", "DECIMAL(18,2)", 0), ("open_pipeline_in_quarter_eur", "DECIMAL(18,2)", 0)],
                          ["rep_id", "submission_week"], []),
 "arr_monthly": ([("month", "DATE", 0), ("account_id", "VARCHAR(18)", 0), ("arr_eur", "DECIMAL(18,2)", 0),
                  ("new_arr_eur", "DECIMAL(18,2)", 0), ("expansion_arr_eur", "DECIMAL(18,2)", 0),
                  ("contraction_arr_eur", "DECIMAL(18,2)", 0), ("churned_arr_eur", "DECIMAL(18,2)", 0),
                  ("usage_overage_revenue_eur", "DECIMAL(18,2)", 0), ("total_executions", "BIGINT", 0)],
                 ["account_id", "month"], [["month"]]),
 "marketing_spend": ([("month", "DATE", 0), ("channel", "VARCHAR(50)", 0), ("region", "VARCHAR(30)", 0),
                      ("spend_eur", "DECIMAL(18,2)", 0)], ["month", "channel", "region"], []),
}

LOAD_ORDER = list(T)
counts = {}


def check_and_format(df, cols, table):
    out = pd.DataFrame(index=df.index)
    for name, typ, nullable in cols:
        s = df[name]
        nn = s.notna()
        if not nullable and (~nn).any():
            raise ValueError(f"{table}.{name}: {(~nn).sum()} NULLs in NOT NULL column")
        if typ == "BIT":
            v = s.map({True: "1", False: "0", "True": "1", "False": "0"})
            assert v[nn].notna().all(), f"{table}.{name} bad bool"
        elif typ in ("INT", "BIGINT"):
            f = pd.to_numeric(s)
            assert ((f[nn] % 1) == 0).all(), f"{table}.{name} non-integer"
            lim = 2**31 - 1 if typ == "INT" else 2**63 - 1
            assert (f[nn].abs() <= lim).all(), f"{table}.{name} overflow"
            v = f.astype("Int64").astype(str).where(nn, None)
        elif typ.startswith("DECIMAL"):
            p, sc = map(int, typ[8:-1].split(","))
            f = pd.to_numeric(s)
            v = f.map(lambda x: None if pd.isna(x) else format(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-sc)), "f"))
            ints = v[nn].str.lstrip("-").str.split(".").str[0].str.len()
            assert (ints <= p - sc).all(), f"{table}.{name} exceeds {typ}"
            assert (pd.to_numeric(v[nn]).round(sc) == f[nn].round(sc)).all(), f"{table}.{name} rounding"
        elif typ == "DATE":
            d = pd.to_datetime(s[nn], format="%Y-%m-%d")
            v = s.where(nn, None)
        elif typ.startswith("DATETIME2"):
            d = pd.to_datetime(s[nn], format="%Y-%m-%d %H:%M:%S")
            v = s.where(nn, None)
        else:  # (N)VARCHAR / CHAR
            n = int(typ[typ.index("(") + 1:-1])
            v = s.where(nn, None).astype(object)
            L = v[nn].astype(str).str.len()
            assert (L <= n).all(), f"{table}.{name}: max len {L.max()} > {n}"
            if typ.startswith("CHAR"):
                assert (L == n).all(), f"{table}.{name} CHAR length mismatch"
            bad = v[nn].astype(str).str.contains(r"[\r\n]")
            assert not bad.any(), f"{table}.{name} has line breaks"
            v = v.where(~(nn & (v.astype(str) == "")), None)
        out[name] = v
    return out


for t, (cols, pk, idx) in T.items():
    df = pd.read_csv(SRC / f"{t}.csv", dtype=str, keep_default_na=False, na_values=[""])
    assert list(df.columns) == [c[0] for c in cols], f"{t}: column mismatch {list(df.columns)}"
    # typed checks need numeric/bool parsing from strings
    out = check_and_format(df, cols, t)
    # PK uniqueness
    assert not out.duplicated(subset=pk).any(), f"{t}: duplicate primary key"
    out.to_csv(OUT / "data" / f"{t}.csv", index=False, lineterminator="\n", encoding="utf-8", na_rep="")
    counts[t] = len(out)
    print(f"ok  {t:28s} {len(out):>8,d}")

# referential checks (not enforced as FKs in raw layer, but verified so joins work)
def ids(t, c):
    return set(pd.read_csv(OUT / "data" / f"{t}.csv", usecols=[c], dtype=str, keep_default_na=False, na_values=[""])[c].dropna())
refs = [("opportunities", "account_id", "accounts", "account_id"), ("opportunities", "owner_id", "sales_reps", "rep_id"),
        ("opportunity_field_history", "opportunity_id", "opportunities", "opportunity_id"),
        ("leads", "converted_opportunity_id", "opportunities", "opportunity_id"), ("leads", "account_id", "accounts", "account_id"),
        ("leads", "workspace_id", "workspaces", "workspace_id"), ("workspaces", "account_id", "accounts", "account_id"),
        ("workspace_usage_monthly", "workspace_id", "workspaces", "workspace_id"), ("arr_monthly", "account_id", "accounts", "account_id"),
        ("quotas", "rep_id", "sales_reps", "rep_id"), ("forecast_submissions", "rep_id", "sales_reps", "rep_id"),
        ("opportunities", "currency", "fx_rates", "currency")]
for a, ac, b, bc in refs:
    miss = ids(a, ac) - ids(b, bc)
    assert not miss, f"{a}.{ac} -> {b}.{bc}: {len(miss)} orphans"
print("ok  all references resolve")

# ------------------------------------------------------------------ SQL scripts
hdr = "-- Generated by build_sql_package.py. Synthetic data, snapshot 2026-09-21.\n"
s1 = [hdr, f"""-- 01: create database, schema and tables (safe to re-run: drops and recreates the raw tables)
IF DB_ID(N'{DB}') IS NULL CREATE DATABASE [{DB}];
GO
USE [{DB}];
GO
IF SCHEMA_ID(N'{SCHEMA}') IS NULL EXEC(N'CREATE SCHEMA [{SCHEMA}]');
GO
"""]
for t in reversed(LOAD_ORDER):
    s1.append(f"DROP TABLE IF EXISTS [{SCHEMA}].[{t}];\n")
s1.append("GO\n\n")
for t in LOAD_ORDER:
    cols, pk, idx = T[t]
    lines = [f"    [{n}] {typ} {'NULL' if nl else 'NOT NULL'}" for n, typ, nl in cols]
    lines.append(f"    CONSTRAINT [PK_{t}] PRIMARY KEY CLUSTERED ({', '.join(f'[{c}]' for c in pk)})")
    s1.append(f"CREATE TABLE [{SCHEMA}].[{t}] (\n" + ",\n".join(lines) + "\n);\n")
    for i, ic in enumerate(idx):
        s1.append(f"CREATE NONCLUSTERED INDEX [IX_{t}_{'_'.join(ic)}] ON [{SCHEMA}].[{t}] ({', '.join(f'[{c}]' for c in ic)});\n")
    s1.append("GO\n\n")
(OUT / "01_create_tables.sql").write_text("".join(s1))

s2 = [hdr, f"""-- 02: bulk load the CSVs.
-- The CSV files must be INSIDE the SQL Server container, in the folder below
-- (copy them there with the docker cp command in LOAD_INSTRUCTIONS.md).
USE [{DB}];
GO
SET NOCOUNT ON;
DECLARE @folder NVARCHAR(400) = N'/var/opt/mssql/import/sale_analytics/';   -- change only if you copied them elsewhere
DECLARE @t SYSNAME, @sql NVARCHAR(MAX);
DECLARE @tables TABLE (ord INT IDENTITY, name SYSNAME);
INSERT @tables (name) VALUES {', '.join(f"(N'{t}')" for t in LOAD_ORDER)};

DECLARE c CURSOR LOCAL FAST_FORWARD FOR SELECT name FROM @tables ORDER BY ord;
OPEN c; FETCH NEXT FROM c INTO @t;
WHILE @@FETCH_STATUS = 0
BEGIN
    SET @sql = N'TRUNCATE TABLE [{SCHEMA}].' + QUOTENAME(@t) + N';
BULK INSERT [{SCHEMA}].' + QUOTENAME(@t) + N'
FROM ''' + @folder + @t + N'.csv''
WITH (FORMAT = ''CSV'', FIRSTROW = 2, FIELDTERMINATOR = '','', FIELDQUOTE = ''"'',
      ROWTERMINATOR = ''0x0a'', CODEPAGE = ''65001'', KEEPNULLS, TABLOCK);';
    EXEC sp_executesql @sql;
    PRINT CONCAT(@t, N': loaded');
    FETCH NEXT FROM c INTO @t;
END
CLOSE c; DEALLOCATE c;
GO
"""]
(OUT / "02_load_data.sql").write_text("".join(s2))

union = "\nUNION ALL ".join(f"SELECT N'{t}', {counts[t]}, (SELECT COUNT_BIG(*) FROM [{SCHEMA}].[{t}])" for t in LOAD_ORDER)
s3 = [hdr, f"""-- 03: verify the load. Every row should say OK.
USE [{DB}];
GO
SELECT table_name, expected_rows, loaded_rows,
       CASE WHEN expected_rows = loaded_rows THEN 'OK' ELSE 'MISMATCH' END AS status
FROM (
SELECT {union.replace("SELECT N'", "N'", 1)}
) x (table_name, expected_rows, loaded_rows)
ORDER BY table_name;

-- quick sanity query: bookings by quarter in EUR (closed won, new business + expansion)
SELECT CONCAT(YEAR(o.close_date), '-Q', DATEPART(QUARTER, o.close_date)) AS fiscal_quarter,
       COUNT(*) AS deals_won,
       CAST(SUM(o.amount * fx.rate_to_eur) AS DECIMAL(18,0)) AS bookings_eur
FROM [{SCHEMA}].[opportunities] o
JOIN [{SCHEMA}].[fx_rates] fx ON fx.currency = o.currency
WHERE o.is_won = 1
GROUP BY YEAR(o.close_date), DATEPART(QUARTER, o.close_date)
ORDER BY 1;
GO
"""]
(OUT / "03_check_load.sql").write_text("".join(s3))

# expected output of the sanity query, for the instructions
o = pd.read_csv(OUT / "data" / "opportunities.csv", parse_dates=["close_date"])
fx = pd.read_csv(OUT / "data" / "fx_rates.csv").set_index("currency").rate_to_eur
w = o[o.is_won == 1].copy()
w["eur"] = w.amount * w.currency.map(fx)
w["q"] = w.close_date.dt.year.astype(str) + "-Q" + w.close_date.dt.quarter.astype(str)
print(w.groupby("q").agg(deals=("eur", "size"), eur=("eur", "sum")).round(0).tail(3))

# ------------------------------------------------------------------ PostgreSQL version
PG = Path(__file__).parent / "sql_postgres"
PG.mkdir(exist_ok=True)
PG_DB = "sale_analytics"
PG_DIR = "/tmp/sale_analytics"


def pg_type(t):
    if t == "BIT":
        return "BOOLEAN"
    if t.startswith("NVARCHAR"):
        return "VARCHAR" + t[8:]
    if t.startswith("DATETIME2"):
        return "TIMESTAMP(0)"
    if t.startswith("DECIMAL"):
        return "NUMERIC" + t[7:]
    return t  # INT, BIGINT, DATE, CHAR(n), VARCHAR(n)


p = [f"""-- Generated by build_sql_package.py. Synthetic data, snapshot 2026-09-21.
-- Creates database {PG_DB}, schema raw, 12 tables; loads the CSVs from {PG_DIR}; checks row counts.
-- Run with psql (see LOAD_INSTRUCTIONS.md). Safe to re-run.
\\set ON_ERROR_STOP on
SELECT 'CREATE DATABASE {PG_DB}' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{PG_DB}')\\gexec
\\c {PG_DB}
SET client_min_messages = warning;

CREATE SCHEMA IF NOT EXISTS raw;
"""]
for t in reversed(LOAD_ORDER):
    p.append(f"DROP TABLE IF EXISTS raw.{t};\n")
p.append("\n")
for t in LOAD_ORDER:
    cols, pk, idx = T[t]
    lines = [f'    "{n}" {pg_type(typ)}{"" if nl else " NOT NULL"}' for n, typ, nl in cols]
    lines.append(f"    CONSTRAINT pk_{t} PRIMARY KEY ({', '.join(pk)})")
    p.append(f"CREATE TABLE raw.{t} (\n" + ",\n".join(lines) + "\n);\n")
    p.append(f"COPY raw.{t} FROM '{PG_DIR}/{t}.csv' WITH (FORMAT csv, HEADER true);\n")
    for ic in idx:
        p.append(f"CREATE INDEX ix_{t}_{'_'.join(ic)} ON raw.{t} ({', '.join(ic)});\n")
    p.append("\n")
p.append("ANALYZE;\n\n-- check: every row should say OK\nSELECT table_name, expected_rows, loaded_rows,\n"
         "       CASE WHEN expected_rows = loaded_rows THEN 'OK' ELSE 'MISMATCH' END AS status\nFROM (\n")
p.append("\nUNION ALL ".join(f"  SELECT '{t}' AS table_name, {counts[t]} AS expected_rows, (SELECT count(*) FROM raw.{t}) AS loaded_rows"
                             if i == 0 else f"  SELECT '{t}', {counts[t]}, (SELECT count(*) FROM raw.{t})"
                             for i, t in enumerate(LOAD_ORDER)))
p.append("\n) x ORDER BY table_name;\n\n-- sanity: won bookings by quarter in EUR\n"
         "SELECT to_char(o.close_date, 'YYYY') || '-Q' || to_char(o.close_date, 'Q') AS fiscal_quarter,\n"
         "       count(*) AS deals_won, round(sum(o.amount * fx.rate_to_eur)) AS bookings_eur\n"
         "FROM raw.opportunities o JOIN raw.fx_rates fx ON fx.currency = o.currency\n"
         "WHERE o.is_won GROUP BY 1 ORDER BY 1;\n")
(PG / "load_sale_analytics.sql").write_text("".join(p))
print("ok  wrote sql_postgres/load_sale_analytics.sql")
