"""
Sale_Analytics | Step 3: forecast backtest
------------------------------------------
Question: how accurate are the reps' forecast calls at different points in the
quarter, and can a data-driven forecast do better?

For every Monday of every completed quarter we pretend it is that day and
predict the quarter's total bookings (closed won, New Business + Expansion, EUR)
with three methods:

  1. Rep call        sum of the reps' weekly "commit" calls
  2. Stage-weighted  won so far + open deals closing this quarter x a fixed
                     probability per stage (standard CRM default)
  3. Model           won so far + every open deal x P(won this quarter), from a
                     logistic regression trained ONLY on earlier quarters
                     (walk-forward, so no peeking at the future)
                     + expected bookings from deals not created yet
                     (same week in earlier quarters, as a share of quota)

Then compare each prediction with what the quarter actually booked.

Reads from PostgreSQL through `docker exec` (no open port needed).
Needs: pandas, scikit-learn, matplotlib.   Run:  python 30_forecast_backtest.py
Outputs go to ./output/
"""
import io
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# How to reach the database. Default: the Docker container called "pg".
PSQL_CMD = os.environ.get(
    "SALE_PSQL",
    "docker exec -i pg sh -c 'psql -U \"${POSTGRES_USER:-postgres}\" -d sale_analytics -X -q'",
)


def query(sql: str) -> pd.DataFrame:
    copy = f"COPY ({sql}) TO STDOUT WITH CSV HEADER;"
    r = subprocess.run(PSQL_CMD, input=copy, shell=True, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"Query failed. Is the container running and have steps 1-2 been run?\n{r.stderr}")
    return pd.read_csv(io.StringIO(r.stdout))


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
snap = query("""
    SELECT s.snapshot_week, s.fiscal_quarter, s.week_of_quarter, s.opportunity_id, s.owner_id,
           s.opportunity_type, s.lead_source, s.segment, s.stage_number, s.forecast_category,
           s.amount_eur, s.close_date, s.closes_this_quarter, s.is_past_due,
           s.deal_age_days, s.days_in_stage, s.close_date_pushes, w.quarter_end
    FROM marts.fct_pipeline_snapshot_weekly s
    JOIN marts.dim_week w USING (snapshot_week)
    WHERE s.is_open
""")
opps = query("""
    SELECT opportunity_id, is_won, close_date, amount_eur
    FROM staging.stg_opportunities WHERE is_closed
""")
won_qtd = query("""
    SELECT snapshot_week, sum(amount_eur) AS won_qtd_eur
    FROM marts.fct_pipeline_snapshot_weekly
    WHERE is_closed_won AND closes_this_quarter
    GROUP BY 1
""")
weeks = query("SELECT snapshot_week, fiscal_quarter, week_of_quarter FROM marts.dim_week")
rep_calls = query("SELECT submission_week AS snapshot_week, rep_id, commit_call_eur FROM raw.forecast_submissions")

for df, cols in [(snap, ["snapshot_week", "close_date", "quarter_end"]), (opps, ["close_date"]),
                 (won_qtd, ["snapshot_week"]), (weeks, ["snapshot_week"]), (rep_calls, ["snapshot_week"])]:
    for c in cols:
        df[c] = pd.to_datetime(df[c])
for c in ["closes_this_quarter", "is_past_due"]:
    snap[c] = snap[c].astype(str).str.lower().isin(["t", "true", "1"])
opps["is_won"] = opps.is_won.astype(str).str.lower().isin(["t", "true", "1"])

# actual bookings per quarter, from field history (same source as "won so far"; the current
# opportunities table has some blanked amounts, the history keeps them)
won_final = query("""
    SELECT s.opportunity_id, s.owner_id, o.created_date, s.close_date, s.amount_eur
    FROM marts.fct_pipeline_snapshot_weekly s
    JOIN staging.stg_opportunities o USING (opportunity_id)
    WHERE s.snapshot_week = (SELECT max(snapshot_week) FROM marts.dim_week) AND s.is_closed_won
""")
for c in ["created_date", "close_date"]:
    won_final[c] = pd.to_datetime(won_final[c])
won_final["fiscal_quarter"] = won_final.close_date.dt.year.astype(str) + "-Q" + won_final.close_date.dt.quarter.astype(str)
actual = won_final.groupby("fiscal_quarter").amount_eur.sum().rename("actual_eur")
opps["fiscal_quarter"] = opps.close_date.dt.year.astype(str) + "-Q" + opps.close_date.dt.quarter.astype(str)
quota = query("SELECT fiscal_quarter, sum(quota_eur) AS quota_eur FROM raw.quotas GROUP BY 1").set_index("fiscal_quarter").quota_eur

# label: did this open deal close won in the SAME quarter as the snapshot?
snap = snap.merge(opps[["opportunity_id", "is_won", "close_date"]].rename(columns={"close_date": "final_close"}),
                  on="opportunity_id", how="left")
snap["won_this_quarter"] = (snap.is_won.fillna(False).astype(bool)
                            & (snap.final_close <= snap.quarter_end)
                            & (snap.final_close >= snap.snapshot_week))
snap["amount_eur"] = snap.amount_eur.fillna(0)
snap["days_to_quarter_end"] = (snap.quarter_end - snap.snapshot_week).dt.days
snap["days_to_close_date"] = (snap.close_date - snap.snapshot_week).dt.days.clip(-120, 400)
snap["log_amount"] = np.log1p(snap.amount_eur)
snap["is_plg"] = snap.lead_source.eq("Product-led (PQL)")

# ---------------------------------------------------------------------------
# 2. Method 1: rep calls (a rep who missed a week keeps last week's call)
# ---------------------------------------------------------------------------
rc = rep_calls.merge(weeks, on="snapshot_week")
grid = rc[["rep_id", "fiscal_quarter"]].drop_duplicates().merge(weeks, on="fiscal_quarter")
rc = grid.merge(rc, on=["rep_id", "fiscal_quarter", "week_of_quarter", "snapshot_week"], how="left")
rc = rc.sort_values(["rep_id", "snapshot_week"])
rc["commit_call_eur"] = rc.groupby(["rep_id", "fiscal_quarter"]).commit_call_eur.ffill()
pred_rep = rc.groupby("snapshot_week").commit_call_eur.sum().rename("pred_rep_call")

# ---------------------------------------------------------------------------
# 3. Method 2: stage-weighted pipeline (standard CRM stage probabilities)
# ---------------------------------------------------------------------------
STAGE_PROB = {1: 0.10, 2: 0.20, 3: 0.40, 4: 0.60, 5: 0.80}
sw = snap[snap.closes_this_quarter].copy()
sw["weighted"] = sw.amount_eur * sw.stage_number.map(STAGE_PROB)
pred_stage = sw.groupby("snapshot_week").weighted.sum().rename("pred_stage_open")

# ---------------------------------------------------------------------------
# 4. Method 3: model, walk-forward by quarter
# ---------------------------------------------------------------------------
NUM = ["stage_number", "days_in_stage", "deal_age_days", "close_date_pushes", "days_to_quarter_end",
       "days_to_close_date", "log_amount"]
BIN = ["closes_this_quarter", "is_past_due", "is_plg"]
CAT = ["forecast_category", "segment", "opportunity_type"]


def make_model():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("bin", "passthrough", BIN),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])


# Bookings that come from deals which do not exist yet on the snapshot Monday.
# For each quarter and week: won amount (closed in that quarter) from deals created on/after that Monday,
# expressed as a share of the quarter's quota so it scales with company growth.
wk_q = weeks.merge(won_final, on="fiscal_quarter")
wk_q = wk_q[wk_q.created_date >= wk_q.snapshot_week]
not_yet = wk_q.groupby(["fiscal_quarter", "week_of_quarter"]).amount_eur.sum()
not_yet = weeks.set_index(["fiscal_quarter", "week_of_quarter"]).join(not_yet).fillna({"amount_eur": 0})
not_yet["share_of_quota"] = not_yet.amount_eur / not_yet.index.get_level_values(0).map(quota)
not_yet = not_yet.reset_index()

quarters = sorted(snap.fiscal_quarter.unique())
completed = [q for q in quarters if q < "2026-Q3"]
test_quarters = [q for q in completed if q >= "2025-Q3"]        # need at least one earlier quarter to learn from
model_preds, aucs = [], []
for q in test_quarters + ["2026-Q3"]:
    train = snap[snap.fiscal_quarter < q]
    test = snap[snap.fiscal_quarter == q].copy()
    m = make_model().fit(train[NUM + BIN + CAT].astype({c: float for c in BIN}), train.won_this_quarter)
    test["p_win_qtr"] = m.predict_proba(test[NUM + BIN + CAT].astype({c: float for c in BIN}))[:, 1]
    if q != "2026-Q3":
        aucs.append((q, roc_auc_score(test.won_this_quarter, test.p_win_qtr)))
    test["expected"] = test.p_win_qtr * test.amount_eur
    exp_open = test.groupby("snapshot_week").expected.sum()
    # + expected bookings from deals not created yet (average share of quota in earlier quarters, same week)
    hist = not_yet[(not_yet.fiscal_quarter < q) & (not_yet.fiscal_quarter >= "2025-Q2")]
    share = hist.groupby("week_of_quarter").share_of_quota.mean()
    wq = weeks[weeks.fiscal_quarter == q].set_index("snapshot_week").week_of_quarter
    exp_new = wq.map(share).fillna(0) * quota[q]
    model_preds.append(exp_open.add(exp_new, fill_value=0))
pred_model = pd.concat(model_preds).rename("pred_model_open")

# final model on all completed quarters, for the coefficient table
full = snap[snap.fiscal_quarter < "2026-Q3"]
final = make_model().fit(full[NUM + BIN + CAT].astype({c: float for c in BIN}), full.won_this_quarter)
names = final.named_steps["pre"].get_feature_names_out()
coefs = pd.DataFrame({"feature": names, "coefficient": final.named_steps["clf"].coef_[0]}) \
    .assign(abs=lambda d: d.coefficient.abs()).sort_values("abs", ascending=False).drop(columns="abs")

# ---------------------------------------------------------------------------
# 5. Put it together
# ---------------------------------------------------------------------------
bt = weeks.set_index("snapshot_week").join([won_qtd.set_index("snapshot_week"), pred_rep, pred_stage, pred_model])
bt["won_qtd_eur"] = bt.won_qtd_eur.fillna(0)
bt["pred_stage_weighted"] = bt.won_qtd_eur + bt.pred_stage_open.fillna(0)
bt["pred_model"] = bt.won_qtd_eur + bt.pred_model_open
bt = bt.join(actual, on="fiscal_quarter").reset_index()
methods = {"pred_rep_call": "Rep calls", "pred_stage_weighted": "Stage-weighted", "pred_model": "Model"}
for c in methods:
    bt[c.replace("pred_", "err_")] = (bt[c] - bt.actual_eur) / bt.actual_eur

evalq = bt[bt.fiscal_quarter.isin(test_quarters)]
err_cols = [c.replace("pred_", "err_") for c in methods]

# accuracy by week of quarter (mean absolute % error across test quarters)
by_week = evalq.groupby("week_of_quarter")[err_cols].apply(lambda d: d.abs().mean()).reset_index()
by_week.columns = ["week_of_quarter"] + list(methods.values())

# bias (signed error): do methods over- or under-call?
bias = evalq.groupby("week_of_quarter")[err_cols].mean().reset_index()
bias.columns = ["week_of_quarter"] + list(methods.values())

# rep-level: call vs what that rep actually closed, week 6
rep_actual = won_final.groupby(["owner_id", "fiscal_quarter"]).amount_eur.sum().rename("rep_actual_eur")
rep6 = rc[(rc.week_of_quarter == 6) & rc.fiscal_quarter.isin(completed)] \
    .join(rep_actual, on=["rep_id", "fiscal_quarter"]).fillna({"rep_actual_eur": 0})
rep_bias = rep6.groupby("rep_id")[["commit_call_eur", "rep_actual_eur"]].sum()
rep_bias = rep_bias[rep_bias.rep_actual_eur > 0]
rep_bias["call_vs_actual"] = (rep_bias.commit_call_eur / rep_bias.rep_actual_eur).round(2)
rep_bias = rep_bias.sort_values("call_vs_actual")

# current quarter outlook
cur = bt[bt.fiscal_quarter == "2026-Q3"].sort_values("snapshot_week").iloc[-1]

# ---------------------------------------------------------------------------
# 6. Save + print
# ---------------------------------------------------------------------------
bt.to_csv(OUT / "backtest_weekly.csv", index=False)
by_week.to_csv(OUT / "accuracy_by_week.csv", index=False)
bias.to_csv(OUT / "bias_by_week.csv", index=False)
rep_bias.to_csv(OUT / "rep_forecast_bias.csv")
coefs.to_csv(OUT / "model_coefficients.csv", index=False)


# write results back into PostgreSQL (schema marts) so Metabase can chart them
def write_table(df: pd.DataFrame, table: str):
    def pg_type(s):
        if pd.api.types.is_datetime64_any_dtype(s):
            return "DATE"
        if pd.api.types.is_numeric_dtype(s):
            return "NUMERIC"
        return "TEXT"
    cols = ", ".join(f'"{c}" {pg_type(df[c])}' for c in df.columns)
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    csv = out.to_csv(index=False, header=False, lineterminator="\n")
    script = (f"\\set ON_ERROR_STOP on\nDROP TABLE IF EXISTS marts.{table};\nCREATE TABLE marts.{table} ({cols});\n"
              f"COPY marts.{table} FROM STDIN WITH CSV;\n{csv}\\.\n")
    r = subprocess.run(PSQL_CMD, input=script, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Could not write marts.{table}:\n{r.stderr}")


acc_long = by_week.melt(id_vars="week_of_quarter", var_name="method", value_name="avg_abs_error")
acc_long["avg_abs_error_pct"] = (acc_long.avg_abs_error * 100).round(1)
write_table(acc_long[["week_of_quarter", "method", "avg_abs_error_pct"]], "forecast_accuracy_by_week")
bias_long = bias.melt(id_vars="week_of_quarter", var_name="method", value_name="bias")
bias_long["bias_pct"] = (bias_long.bias * 100).round(1)
write_table(bias_long[["week_of_quarter", "method", "bias_pct"]], "forecast_bias_by_week")
write_table(bt[["snapshot_week", "fiscal_quarter", "week_of_quarter", "won_qtd_eur", "pred_rep_call",
                "pred_stage_weighted", "pred_model", "actual_eur"]].round(0), "forecast_backtest_weekly")
write_table(rep_bias.reset_index().rename(columns={"rep_id": "rep_id"}), "rep_forecast_bias")

pd.set_option("display.width", 140)
print("\nModel ranking quality (AUC, 0.5 = coin flip, 1.0 = perfect), per test quarter:")
for q, a in aucs:
    print(f"  {q}: {a:.3f}")
print("\nAverage absolute error vs actual bookings, by week of quarter (test quarters "
      f"{test_quarters[0]} to {test_quarters[-1]}):")
print((by_week.set_index("week_of_quarter") * 100).round(1).astype(str).add("%").to_string())
print("\nBias (+ = over-forecast, - = under-forecast), selected weeks:")
print((bias.set_index("week_of_quarter").loc[[2, 6, 10, 13]] * 100).round(1).astype(str).add("%").to_string())
print("\nRep call vs what the rep actually closed (week 6, completed quarters):")
print(rep_bias[["call_vs_actual"]].T.to_string())
print(f"\nCurrent quarter 2026-Q3 as of {cur.snapshot_week.date()} (week {int(cur.week_of_quarter)}):")
print(f"  won so far        EUR {cur.won_qtd_eur:,.0f}")
for c, n in methods.items():
    print(f"  {n:16s}  EUR {cur[c]:,.0f}")
wk1 = not_yet[(not_yet.week_of_quarter == 1) & not_yet.fiscal_quarter.isin(completed)].set_index("fiscal_quarter")
wk1_share = (wk1.amount_eur / wk1.index.map(actual)).round(3)
print("\nShare of each quarter's bookings from deals that did not exist in week 1:")
print("  " + ", ".join(f"{q}: {v:.0%}" for q, v in wk1_share.items()))
print("\nTop model drivers (standardised coefficients):")
print(coefs.head(8).round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Chart: forecast error by week of quarter
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Rep calls": "#2a78d6", "Stage-weighted": "#eb6834", "Model": "#1baf7a"}
fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
last = {}
for name, col in COLORS.items():
    y = by_week[name] * 100
    ax.plot(by_week.week_of_quarter, y, color=col, lw=2, marker="o", ms=4.5, label=name,
            markeredgecolor=SURFACE, markeredgewidth=1)
    last[name] = y.iloc[-1]
# end-of-line labels, nudged apart so they never overlap
ymax = max(60, by_week[list(COLORS)].max().max() * 100 * 1.1)
min_gap = ymax * 0.045
placed = {}
for name, v in sorted(last.items(), key=lambda kv: kv[1]):
    pos = v if not placed else max(v, max(placed.values()) + min_gap)
    placed[name] = pos
for name, pos in placed.items():
    ax.annotate(f"{name}  {last[name]:.0f}%", xy=(13, last[name]), xytext=(13.35, pos),
                va="center", fontsize=9, color=INK,
                arrowprops=None if abs(pos - last[name]) < 0.5 else dict(arrowstyle="-", color=GRID, lw=0.8))
ax.set_xlim(0.5, 16.3)
ax.set_ylim(0, ymax)
ax.set_xticks(range(1, 14))
ax.set_xlabel("Week of quarter", color=INK2, fontsize=9)
ax.set_ylabel("Average forecast error vs actual bookings", color=INK2, fontsize=9)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8)
ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK, bbox_to_anchor=(0.8, 1.0))
fig.subplots_adjust(top=0.84, left=0.09, right=0.98, bottom=0.11)
fig.text(0.09, 0.945, "How far off each forecast method is, by week of the quarter", fontsize=12.5, color=INK)
fig.text(0.09, 0.9, f"Mean absolute % error vs actual bookings, {len(test_quarters)} completed quarters "
         f"({test_quarters[0]} to {test_quarters[-1]}). Lower is better.", fontsize=8.5, color=INK2)
fig.savefig(OUT / "forecast_error_by_week.png", facecolor=SURFACE)
print(f"\nSaved results and chart to {OUT}/")
