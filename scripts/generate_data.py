"""
Synthetic GTM dataset for a PLG + enterprise workflow-automation SaaS
(modelled loosely on a company like n8n: self-serve cloud plans, a free
self-hosted community edition, usage (execution) based pricing, and an
enterprise sales team working product-qualified accounts).

All data is synthetic. Snapshot date ("as of") is 2026-09-21.

Run:  python generate_data.py  -> writes CSVs to ./data
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

START = pd.Timestamp("2024-01-01")
AS_OF = pd.Timestamp("2026-09-21")
SF_MIGRATION = pd.Timestamp("2025-01-15")
MONTHS = pd.date_range(START, AS_OF, freq="MS")  # month starts
NM = len(MONTHS)
FX_TO_EUR = {"EUR": 1.0, "USD": 0.92, "GBP": 1.17}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# 1. Accounts
# ---------------------------------------------------------------------------
N_ACC = 9000
syl_a = ["Nord", "Vali", "Kest", "Bram", "Orio", "Tern", "Lumo", "Quar", "Sella", "Vex",
         "Halo", "Mira", "Stra", "Cobal", "Fjel", "Aster", "Brio", "Delta", "Evo", "Gran",
         "Iven", "Juno", "Kora", "Lyra", "Mosa", "Nexa", "Opal", "Pira", "Rune", "Solv",
         "Tala", "Umbra", "Vera", "Wend", "Xeno", "Yara", "Zeph", "Alto", "Borea", "Cira"]
syl_b = ["vik", "tech", "ware", "labs", "data", "flow", "line", "point", "logic", "works",
         "sys", "net", "grid", "core", "path", "stack", "wave", "mark", "bridge", "field"]
suffix_by_country = {
    "Germany": ["GmbH", "AG", "SE"], "Austria": ["GmbH"], "Switzerland": ["AG", "GmbH"],
    "United Kingdom": ["Ltd", "plc"], "Ireland": ["Ltd"], "France": ["SAS", "SA"],
    "Netherlands": ["B.V.", "N.V."], "Spain": ["S.L.", "S.A."], "Italy": ["S.r.l.", "S.p.A."],
    "Sweden": ["AB"], "Finland": ["Oy", "Oyj"], "Denmark": ["A/S", "ApS"], "Norway": ["AS"],
    "Poland": ["Sp. z o.o."], "United States": ["Inc.", "LLC", "Corp."], "Canada": ["Inc.", "Ltd"],
    "Brazil": ["Ltda"], "India": ["Pvt Ltd"], "Australia": ["Pty Ltd"], "Singapore": ["Pte Ltd"],
    "Japan": ["K.K."],
}
country_w = {
    "Germany": 14, "Austria": 2, "Switzerland": 3, "United Kingdom": 9, "Ireland": 1.5,
    "France": 6, "Netherlands": 4, "Spain": 3, "Italy": 3, "Sweden": 2.5, "Finland": 1.5,
    "Denmark": 1.5, "Norway": 1.2, "Poland": 2.5, "United States": 26, "Canada": 3.5,
    "Brazil": 3, "India": 5, "Australia": 2.5, "Singapore": 1.2, "Japan": 1.5,
}
region_of = {
    "Germany": "EMEA - DACH", "Austria": "EMEA - DACH", "Switzerland": "EMEA - DACH",
    "United Kingdom": "EMEA - UK&I", "Ireland": "EMEA - UK&I",
    "France": "EMEA - South", "Spain": "EMEA - South", "Italy": "EMEA - South",
    "Netherlands": "EMEA - North", "Sweden": "EMEA - North", "Finland": "EMEA - North",
    "Denmark": "EMEA - North", "Norway": "EMEA - North", "Poland": "EMEA - North",
    "United States": "North America", "Canada": "North America",
    "Brazil": "LATAM", "India": "APAC", "Australia": "APAC", "Singapore": "APAC", "Japan": "APAC",
}
currency_of = {c: ("USD" if region_of[c] in ("North America", "LATAM", "APAC") else "EUR") for c in country_w}
currency_of["United Kingdom"] = "GBP"

industries = ["Software & SaaS", "Financial Services", "E-commerce & Retail", "Manufacturing",
              "Healthcare & Life Sciences", "Professional Services", "Media & Marketing",
              "Logistics & Transport", "Education", "Public Sector", "Telecommunications", "Energy & Utilities"]
ind_w = np.array([22, 10, 11, 9, 7, 12, 9, 6, 4, 3, 3, 4], float)
# messy CRM labels for the same industry (hygiene issue)
ind_variants = {
    "Software & SaaS": ["Software", "SaaS", "software", "Computer Software"],
    "Financial Services": ["Finance", "Fintech", "Banking"],
    "E-commerce & Retail": ["Retail", "Ecommerce", "E-Commerce"],
    "Professional Services": ["Consulting", "Agency"],
}

countries = list(country_w)
cw = np.array([country_w[c] for c in countries]); cw /= cw.sum()
acc_country = rng.choice(countries, N_ACC, p=cw)
employees = np.round(np.exp(rng.normal(4.6, 1.55, N_ACC))).astype(int) + 2
employees = np.clip(employees, 2, 250000)
segment = np.where(employees >= 2000, "Enterprise", np.where(employees >= 250, "Mid-Market", "Commercial"))
industry = rng.choice(industries, N_ACC, p=ind_w / ind_w.sum())
names = set()
acc_names = []
for c in acc_country:
    while True:
        n = rng.choice(syl_a) + rng.choice(syl_b)
        n = n[0].upper() + n[1:]
        if rng.random() < 0.3:
            n += " " + rng.choice(["Group", "Solutions", "Digital", "Partners", "Systems", "Holding"])
        full = f"{n} {rng.choice(suffix_by_country[c])}"
        if full not in names:
            names.add(full)
            acc_names.append(full)
            break
acc_created = START - pd.to_timedelta(rng.integers(0, 900, N_ACC), unit="D")
accounts = pd.DataFrame({
    "account_id": [f"001A{i:06d}" for i in range(N_ACC)],
    "account_name": acc_names,
    "website_domain": [n.split(" ")[0].lower() + rng.choice([".com", ".io", ".de", ".co", ".eu", ".net"]) for n in acc_names],
    "industry": industry,
    "employee_count": employees,
    "segment": segment,
    "billing_country": acc_country,
    "region": [region_of[c] for c in acc_country],
    "annual_revenue_eur": np.round(employees * np.exp(rng.normal(np.log(160000), 0.6, N_ACC)), -3),
    "created_date": acc_created.date,
    "tech_stack_competitor": rng.choice(["None", "Zapier", "Make", "Workato", "Power Automate", "Tray.io", "Custom scripts"],
                                        N_ACC, p=[0.34, 0.2, 0.14, 0.07, 0.12, 0.03, 0.10]),
})
# latent account fit (hidden, drives behaviour)
acc_fit = rng.normal(0, 1, N_ACC) + 0.15 * (accounts["industry"].isin(["Software & SaaS", "E-commerce & Retail", "Media & Marketing"]).values)
acc_idx = {a: i for i, a in enumerate(accounts.account_id)}

# ---------------------------------------------------------------------------
# 2. Reps (AEs, SDRs, managers) with hiring over time
# ---------------------------------------------------------------------------
first = ["Anna", "Lukas", "Sofia", "Jonas", "Emma", "Felix", "Mia", "Noah", "Lea", "Elias", "Clara", "Ben",
         "Hannah", "Paul", "Laura", "Max", "Julia", "Tom", "Sara", "David", "Nina", "Oliver", "Maya", "Leo",
         "Chloe", "Ethan", "Priya", "Arjun", "Isabel", "Marco", "Ines", "Kenji", "Olivia", "Liam", "Grace", "Samuel"]
last = ["Schmidt", "Weber", "Fischer", "Wagner", "Becker", "Hoffmann", "Koch", "Richter", "Klein", "Wolf",
        "Neumann", "Braun", "Keller", "Lange", "Walker", "Bennett", "Murphy", "Collins", "Rossi", "Garcia",
        "Martin", "Dubois", "Jansen", "Nilsson", "Virtanen", "Nowak", "Silva", "Sharma", "Tanaka", "Olsen"]
territories = [  # (region, segment)
    ("EMEA - DACH", "Enterprise"), ("EMEA - DACH", "Mid-Market"), ("EMEA - DACH", "Commercial"),
    ("EMEA - UK&I", "Enterprise"), ("EMEA - UK&I", "Mid-Market"), ("EMEA - UK&I", "Commercial"),
    ("EMEA - North", "Mid-Market"), ("EMEA - North", "Enterprise"), ("EMEA - North", "Commercial"),
    ("EMEA - South", "Mid-Market"), ("EMEA - South", "Commercial"), ("EMEA - South", "Enterprise"),
    ("North America", "Enterprise"), ("North America", "Mid-Market"), ("North America", "Commercial"),
    ("APAC", "Mid-Market"), ("LATAM", "Mid-Market"),
]
reps = []
rid = 0
used_names = set()


def new_name():
    while True:
        n = f"{rng.choice(first)} {rng.choice(last)}"
        if n not in used_names:
            used_names.add(n)
            return n


managers = {}
for mreg in ["EMEA", "North America", "APAC & LATAM"]:
    managers[mreg] = f"005M{len(managers):04d}"
    reps.append(dict(rep_id=managers[mreg], rep_name=new_name(), role="Sales Manager", segment="All",
                     region=mreg, manager_id=None, hire_date=pd.Timestamp("2023-06-01"),
                     termination_date=pd.NaT))


def mgr_for(region):
    return managers["EMEA"] if region.startswith("EMEA") else managers["North America"] if region == "North America" else managers["APAC & LATAM"]


# AE hiring: first 8 territories staffed at start, others over time; some territories get a 2nd/3rd AE
ae_plan = []
for i, (reg, seg) in enumerate(territories):
    hire = pd.Timestamp("2023-09-01") if i < 8 else START + pd.Timedelta(days=int(rng.integers(60, 700)))
    ae_plan.append((reg, seg, hire))
extra = [("EMEA - DACH", "Mid-Market"), ("North America", "Mid-Market"), ("North America", "Enterprise"), ("EMEA - UK&I", "Mid-Market"),
         ("North America", "Commercial"), ("EMEA - DACH", "Enterprise"), ("North America", "Mid-Market"), ("EMEA - DACH", "Commercial")]
for reg, seg in extra:
    ae_plan.append((reg, seg, START + pd.Timedelta(days=int(rng.integers(200, 900)))))
for reg, seg, hire in ae_plan:
    reps.append(dict(rep_id=f"005A{rid:04d}", rep_name=new_name(), role="Account Executive", segment=seg,
                     region=reg, manager_id=mgr_for(reg), hire_date=hire, termination_date=pd.NaT))
    rid += 1
for k, reg in enumerate(["EMEA - DACH", "EMEA - UK&I", "EMEA - North", "North America", "North America", "EMEA - South", "APAC"]):
    reps.append(dict(rep_id=f"005S{k:04d}", rep_name=new_name(), role="Sales Development Rep", segment="All",
                     region=reg, manager_id=mgr_for(reg), hire_date=START + pd.Timedelta(days=int(rng.integers(0, 400))),
                     termination_date=pd.NaT))
reps = pd.DataFrame(reps)
ae_mask = reps.role == "Account Executive"
# attrition: 4 AEs leave
leavers = rng.choice(reps[ae_mask].index, 4, replace=False)
for ix in leavers:
    h = reps.at[ix, "hire_date"]
    reps.at[ix, "termination_date"] = min(AS_OF - pd.Timedelta(days=20), max(h + pd.Timedelta(days=300), START + pd.Timedelta(days=int(rng.integers(300, 950)))))
rep_skill = pd.Series(rng.normal(0, 0.5, len(reps)), index=reps.rep_id)
# forecasting personality: multiplier applied to what they call as commit
rep_bias = pd.Series(rng.choice([0.8, 0.95, 1.0, 1.1, 1.3], len(reps), p=[0.2, 0.2, 0.25, 0.2, 0.15]), index=reps.rep_id)
rep_optimist_stage3_commit = pd.Series(rep_bias.values >= 1.1, index=reps.rep_id)


def pick_owner(region, seg, when):
    """Round-robin-ish owner among active AEs in territory; fall back to region, then any."""
    active = reps[ae_mask & (reps.hire_date <= when) & (reps.termination_date.isna() | (reps.termination_date > when))]
    cands = active[(active.region == region) & (active.segment == seg)]
    if cands.empty:
        cands = active[active.region == region]
    if cands.empty:
        reg_group = "North America" if region in ("North America", "LATAM") else "EMEA - DACH" if region.startswith("EMEA") else region
        cands = active[active.region == reg_group]
    if cands.empty:
        cands = active
    return cands.rep_id.values[rng.integers(0, len(cands))]


# ---------------------------------------------------------------------------
# 3. Workspaces + monthly product usage
# ---------------------------------------------------------------------------
N_WS = 24000
# signups accelerate over time
sw = np.linspace(1, 3.2, NM) ** 1.3
sw /= sw.sum()
ws_signup_m = rng.choice(NM, N_WS, p=sw)
ws_signup = MONTHS[ws_signup_m] + pd.to_timedelta(rng.integers(0, 28, N_WS), unit="D")
linked = rng.random(N_WS) < 0.72
acc_pick_w = (accounts.employee_count.values ** 0.35) * np.exp(0.4 * acc_fit)
acc_pick_w /= acc_pick_w.sum()
ws_acc = np.where(linked, rng.choice(N_ACC, N_WS, p=acc_pick_w), -1)
hosting = rng.choice(["cloud", "self_hosted_community", "self_hosted_business"], N_WS, p=[0.62, 0.33, 0.05])
emp_ws = np.where(ws_acc >= 0, accounts.employee_count.values[np.maximum(ws_acc, 0)], rng.integers(1, 30, N_WS))
fit_ws = np.where(ws_acc >= 0, acc_fit[np.maximum(ws_acc, 0)], 0) * 0.6 + rng.normal(0, 0.8, N_WS)
telemetry = np.where(hosting == "self_hosted_community", rng.random(N_WS) < 0.65, True)
signup_src = rng.choice(["Organic search", "GitHub / Docs", "Community forum", "YouTube / Tutorials", "Paid search",
                         "Referral", "Template library", "Direct"], N_WS, p=[0.22, 0.2, 0.08, 0.14, 0.1, 0.08, 0.08, 0.1])

base = np.exp(rng.normal(6.2 + 0.45 * fit_ws + 0.22 * np.log10(emp_ws), 1.0))
growth = 0.02 + 0.045 * fit_ws + rng.normal(0, 0.04, N_WS)
churn_h = sigmoid(-3.1 - 0.9 * fit_ws + 0.3 * (hosting == "cloud"))
alive = np.zeros(N_WS, bool)
churn_month = np.full(N_WS, -1)
plan_rank = np.zeros(N_WS, int)  # 0 Starter,1 Pro,2 Business,3 Enterprise
PLANS = ["Starter", "Pro", "Business", "Enterprise"]
PLAN_MRR = {"Starter": 24.0, "Pro": 60.0, "Business": 800.0, "Enterprise": 0.0}
PLAN_LIMIT = {"Starter": 2500, "Pro": 10000, "Business": 40000, "Enterprise": 10**9}
users_base = 1 + rng.poisson(0.4 + 0.5 * np.log10(emp_ws + 1) + 0.4 * np.maximum(fit_ws, 0))
enterprise_from = np.full(N_WS, 10**6)  # month index when account became enterprise customer (filled later)
rows = []
exec_hist = np.zeros((N_WS, NM))
users_hist = np.zeros((N_WS, NM))
for m in range(NM):
    starting = ws_signup_m == m
    alive |= starting
    age = m - ws_signup_m
    live = alive & (age >= 0)
    ramp = np.where(age == 0, 0.25, np.where(age == 1, 0.6, 1.0))
    seas = 1 + 0.06 * np.sin(2 * np.pi * (m % 12) / 12) - 0.12 * ((m % 12) == 7)  # August dip
    ex = base * np.exp(growth * np.clip(age, 0, 30)) * ramp * seas * np.exp(rng.normal(0, 0.28, N_WS))
    ex = np.where(live, np.round(ex), 0)
    exec_hist[:, m] = ex
    us = np.where(live, np.maximum(1, np.round(users_base * (1 + 0.04 * np.clip(age, 0, 24) * (fit_ws > 0)) + rng.normal(0, 0.6, N_WS))), 0)
    users_hist[:, m] = us
    # plan upgrades for cloud: follow usage with lag, some stay under-licensed
    need = np.select([ex > 40000, ex > 10000, ex > 2500], [2, 2, 1], 0)
    need = np.where(ex > 10000, 2, need)
    upgrade = (need > plan_rank) & (rng.random(N_WS) < 0.45)
    plan_rank = np.where(live & upgrade, np.minimum(need, 2), plan_rank)
    # month-level churn after month 1
    ch = live & (age >= 1) & (rng.random(N_WS) < churn_h * np.where(ex < 300, 2.2, 1.0))
    ch &= ~(enterprise_from <= m)
    churn_month = np.where(ch & (churn_month < 0), m, churn_month)
    for_rows = np.where(live)[0]
    rows.append((m, for_rows, ex[for_rows], us[for_rows], plan_rank[for_rows].copy()))
    alive &= ~ch

# ---------------------------------------------------------------------------
# 4. Leads (PQL / MQL / outbound / partner) and opportunity creation events
# ---------------------------------------------------------------------------
# helpers on usage
def acct_usage_at(ai, m):
    """account-level usage features at month m (sum over linked workspaces)."""
    w = np.where(ws_acc == ai)[0]
    if len(w) == 0:
        return 0.0, 0.0, 0
    e_now = exec_hist[w, max(m - 1, 0)].sum() + exec_hist[w, m].sum()
    e_prev = exec_hist[w, max(m - 4, 0)].sum() + exec_hist[w, max(m - 3, 0)].sum()
    g = np.log((e_now + 2000) / (e_prev + 2000))  # smoothed so brand-new workspaces don't look like 'hyper-growth'
    u = users_hist[w, m].sum()
    return e_now / 2, g, u


leads = []
lead_id = 0


def add_lead(created, src, acc_i, ws_i=None, pql_flag=False, email_domain=None):
    global lead_id
    lid = f"00QL{lead_id:07d}"
    lead_id += 1
    leads.append(dict(lead_id=lid, created_date=created, lead_source=src,
                      account_id=accounts.account_id[acc_i] if acc_i is not None and acc_i >= 0 else None,
                      workspace_id=f"ws_{ws_i:06d}" if ws_i is not None else None,
                      pql_rule_v1_flag=pql_flag, email_domain=email_domain))
    return lid


# PQL rule v1 (the crude current definition): execs in month >= 15k OR >= 5 active users OR Business plan
pql_candidates = []
for m, idx, ex, us, pr in rows:
    fire = (ex >= 15000) | (us >= 5) | (pr >= 2)
    for w in idx[fire]:
        pql_candidates.append((w, m))
pql_first = {}
for w, m in pql_candidates:
    if w not in pql_first:
        pql_first[w] = m

opp_requests = []  # (created_date, account_idx, source, lead_id, workspace)
acc_has_ws = np.zeros(N_ACC, bool)
acc_has_ws[ws_acc[ws_acc >= 0]] = True

for w, m in pql_first.items():
    if hosting[w] == "self_hosted_community" and not telemetry[w]:
        continue
    created = MONTHS[m] + pd.Timedelta(days=int(rng.integers(1, 27)))
    created = max(created, ws_signup[w] + pd.Timedelta(days=2))  # a PQL can't precede the workspace sign-up
    if created > AS_OF:
        continue
    ai = ws_acc[w]
    lid = add_lead(created, "Product (PQL)", ai if ai >= 0 else None, w, True,
                   email_domain=None if ai >= 0 else rng.choice(["gmail.com", "outlook.com", "proton.me", "yahoo.com"]))
    if ai < 0:
        continue
    e, g, u = acct_usage_at(ai, m)
    x = (-4.0 + 1.6 * np.clip(g, -1.5, 2) + 0.4 * np.log2(u + 1) + 0.6 * np.log10(accounts.employee_count[ai])
         + 0.5 * acc_fit[ai] + 0.25 * np.log10(e + 1) + 0.4 * (hosting[w] == "self_hosted_business") - 0.3 * (hosting[w] == "self_hosted_community"))
    if rng.random() < sigmoid(x):
        opp_requests.append((created + pd.Timedelta(days=int(rng.integers(3, 30))), ai, "Product-led (PQL)", lid, w))

# PLG-influenced inbound demo requests: accounts with growing usage but not flagged by rule
for m, idx, ex, us, pr in rows:
    if m < 2:
        continue
    cand = idx[(ws_acc[idx] >= 0)]
    g = np.log((exec_hist[cand, m] + 2000) / (exec_hist[cand, m - 2] + 2000))
    p = sigmoid(-5.6 + 0.9 * np.clip(g, -2, 3) + 0.35 * np.log10(emp_ws[cand] + 1) + 0.3 * fit_ws[cand])
    hit = cand[rng.random(len(cand)) < p]
    for w in hit:
        created = MONTHS[m] + pd.Timedelta(days=int(rng.integers(0, 27)))
        created = max(created, ws_signup[w] + pd.Timedelta(days=2))
        if created > AS_OF:
            continue
        ai = ws_acc[w]
        lid = add_lead(created, "Website - Demo request", ai, w, False)
        if rng.random() < sigmoid(-0.9 + 0.4 * acc_fit[ai] + 0.3 * np.log10(accounts.employee_count[ai])):
            opp_requests.append((created + pd.Timedelta(days=int(rng.integers(2, 20))), ai, "Inbound", lid, w))

# marketing MQLs (content, webinars, events, paid) - most never convert
mkt_sources = {"Webinar": 70, "Content download": 140, "Event / Trade show": 45, "Paid social": 60, "Paid search - Enterprise LP": 35}
growth_curve = np.linspace(0.7, 1.6, NM)
for m in range(NM):
    for src, rate in mkt_sources.items():
        n = rng.poisson(rate * growth_curve[m])
        ais = rng.choice(N_ACC, n, p=(accounts.employee_count.values ** 0.5) / (accounts.employee_count.values ** 0.5).sum())
        for ai in ais:
            created = MONTHS[m] + pd.Timedelta(days=int(rng.integers(0, 28)))
            if created > AS_OF:
                continue
            nomatch = rng.random() < 0.08  # lead-to-account matching failed
            lid = add_lead(created, src, None if nomatch else ai, None, False)
            if nomatch:
                continue
            e, g, u = acct_usage_at(ai, m)
            has_usage = e > 500
            x = -4.3 + 1.6 * has_usage + 0.5 * acc_fit[ai] + 0.25 * np.log10(accounts.employee_count[ai]) + \
                (0.5 if src in ("Event / Trade show", "Paid search - Enterprise LP") else 0)
            if rng.random() < sigmoid(x):
                opp_requests.append((created + pd.Timedelta(days=int(rng.integers(5, 45))), ai, "Marketing (MQL)", lid, None))

# SDR outbound into larger accounts
sdrs = reps[reps.role == "Sales Development Rep"]
for m in range(NM):
    n = rng.poisson(110 * growth_curve[m])
    big = np.where(accounts.employee_count.values >= 200)[0]
    ais = rng.choice(big, n)
    for ai in ais:
        created = MONTHS[m] + pd.Timedelta(days=int(rng.integers(0, 28)))
        if created > AS_OF:
            continue
        lid = add_lead(created, "SDR Outbound", ai, None, False)
        e, g, u = acct_usage_at(ai, m)
        x = -3.9 + 1.2 * (e > 500) + 0.35 * acc_fit[ai]
        if rng.random() < sigmoid(x):
            opp_requests.append((created + pd.Timedelta(days=int(rng.integers(7, 40))), ai, "Outbound", lid, None))

# partner referrals
for m in range(NM):
    for ai in rng.choice(N_ACC, rng.poisson(9 * growth_curve[m])):
        created = MONTHS[m] + pd.Timedelta(days=int(rng.integers(0, 28)))
        if created > AS_OF:
            continue
        lid = add_lead(created, "Partner referral", ai, None, False)
        if rng.random() < 0.38:
            opp_requests.append((created + pd.Timedelta(days=int(rng.integers(3, 25))), ai, "Partner", lid, None))

leads = pd.DataFrame(leads)
opp_requests.sort(key=lambda r: r[0])

# ---------------------------------------------------------------------------
# 5. Opportunities: outcome + full timeline (stage, close date, amount, forecast category)
# ---------------------------------------------------------------------------
STAGES = ["1 - Discovery", "2 - Qualification", "3 - Technical Validation", "4 - Proposal", "5 - Negotiation"]
SEG_CYCLE = {"Commercial": 38, "Mid-Market": 72, "Enterprise": 128}
SEG_ACV = {"Commercial": 11000, "Mid-Market": 32000, "Enterprise": 105000}
LOSS_REASONS = ["No decision / status quo", "Chose self-hosted Community edition", "Competitor - Zapier",
                "Competitor - Make", "Competitor - Workato", "Competitor - Power Automate", "Security / compliance review",
                "Price / budget", "Timing", "Champion left", "Missing feature / integration", "Unresponsive"]

acct_open_until = {}  # account -> date until which a new-business opp is open/active
acct_customer_since = {}
opps = []
events = []  # (opp_id, ts, field, old, new, by)
lead_conv = {}
oid = 0


def fc_for(stage_i, owner):
    if stage_i <= 1:
        return "Pipeline"
    if stage_i == 2:
        return "Commit" if rep_optimist_stage3_commit[owner] and rng.random() < 0.5 else "Best Case"
    if stage_i == 3:
        return "Commit" if rep_optimist_stage3_commit[owner] else ("Best Case" if rng.random() < 0.6 else "Commit")
    return "Commit"


def simulate_opp(opp_type, created, ai, source, owner, usage_z, ws=None, base_amt=None):
    global oid
    seg = accounts.segment[ai]
    cur = currency_of[accounts.billing_country[ai]]
    opp_id = f"006O{oid:07d}"
    oid += 1
    skill = rep_skill[owner]
    comp = accounts.tech_stack_competitor[ai]
    # --- outcome
    x = (-2.25 + 0.75 * (source == "Product-led (PQL)") + 0.35 * (source == "Partner") - 0.55 * (source == "Outbound")
         + 0.7 * usage_z + 0.55 * skill + 0.35 * acc_fit[ai] - 0.35 * (seg == "Enterprise")
         - 0.35 * (comp in ("Workato", "Power Automate")) + (1.5 if opp_type == "Expansion" else 0)
         + rng.normal(0, 0.5))
    tenure_days = (created - reps.set_index("rep_id").at[owner, "hire_date"]).days
    x -= 0.5 * (tenure_days < 120)
    won = rng.random() < sigmoid(x)
    # stage reached
    if won:
        exit_stage = 5
    else:
        exit_stage = int(rng.choice([1, 2, 3, 4, 5], p=[0.36, 0.26, 0.18, 0.12, 0.08] if x < -1 else [0.18, 0.24, 0.24, 0.2, 0.14]))
    # --- durations
    cyc = SEG_CYCLE[seg] * (0.55 if opp_type == "Expansion" else 1) * np.exp(-0.25 * skill) * (0.85 if source == "Product-led (PQL)" else 1)
    fr = np.array([0.15, 0.2, 0.25, 0.2, 0.2])
    durs = rng.gamma(2.2, cyc * fr / 2.2)
    t = created
    stage_times = []
    for s in range(exit_stage):
        stage_times.append(t)
        t = t + pd.Timedelta(days=max(1, float(durs[s])))
    end = t
    if not won:
        end = end + pd.Timedelta(days=float(rng.gamma(1.6, 18 + 8 * exit_stage)))  # zombie time before closing out
    # --- amount
    amt = base_amt if base_amt is not None else SEG_ACV[seg] * np.exp(rng.normal(0, 0.55) + 0.25 * usage_z)
    amt = float(np.round(amt / FX_TO_EUR[cur], -2))
    # --- close-date behaviour
    first_close = created + pd.Timedelta(days=float(cyc * rng.uniform(0.45, 0.9)))
    ev = []
    ev.append((created, "Created", None, None))
    ev.append((created, "StageName", None, STAGES[0]))
    ev.append((created, "CloseDate", None, first_close.normalize()))
    ev.append((created, "Amount", None, amt))
    ev.append((created, "ForecastCategoryName", None, "Pipeline"))
    close_d = first_close.normalize()
    fc = "Pipeline"
    skip = rng.random() < 0.1
    for s in range(1, exit_stage):
        ts = stage_times[s] + pd.Timedelta(hours=int(rng.integers(8, 18)))
        # pushes that happen before this stage change
        while close_d < ts - pd.Timedelta(days=2):
            push_at = close_d + pd.Timedelta(days=int(rng.integers(0, 9)), hours=int(rng.integers(8, 18)))
            if push_at > ts:
                break
            new_close = (push_at + pd.Timedelta(days=float(rng.gamma(2, 14)))).normalize()
            ev.append((push_at, "CloseDate", close_d, new_close))
            close_d = new_close
        if skip and s in (1, 2) and exit_stage > 3:
            continue  # stage skipped in CRM
        ev.append((ts, "StageName", STAGES[s - 1] if not (skip and s == 3) else STAGES[0], STAGES[s]))
        nfc = fc_for(s, owner)
        if nfc != fc:
            ev.append((ts, "ForecastCategoryName", fc, nfc))
            fc = nfc
        if s == 3 and rng.random() < 0.35:
            nam = float(np.round(amt * rng.choice([0.75, 0.85, 1.15, 1.3], p=[0.3, 0.35, 0.2, 0.15]), -2))
            ev.append((ts + pd.Timedelta(hours=1), "Amount", amt, nam))
            amt = nam
        if s == 4 and rng.random() < 0.4:
            nam = float(np.round(amt * rng.uniform(0.8, 0.97), -2))  # negotiation discount
            ev.append((ts + pd.Timedelta(days=2), "Amount", amt, nam))
            amt = nam
    # pushes before close
    while close_d < end - pd.Timedelta(days=2):
        push_at = close_d + pd.Timedelta(days=int(rng.integers(0, 12)), hours=int(rng.integers(8, 18)))
        if push_at > end:
            break
        new_close = (push_at + pd.Timedelta(days=float(rng.gamma(2, 14)))).normalize()
        ev.append((push_at, "CloseDate", close_d, new_close))
        close_d = new_close
    end_ts = end + pd.Timedelta(hours=int(rng.integers(9, 19)))
    last_stage = ev_last = [e for e in ev if e[1] == "StageName"][-1][3]
    final_stage = "Closed Won" if won else "Closed Lost"
    ev.append((end_ts, "StageName", last_stage, final_stage))
    ev.append((end_ts, "ForecastCategoryName", fc, "Closed" if won else "Omitted"))
    ev.append((end_ts, "CloseDate", close_d, end.normalize()))
    loss = None
    if not won:
        p = np.ones(len(LOSS_REASONS))
        p[0] += 3 * (usage_z < 0) + 1.5 * (exit_stage <= 2)
        p[1] += 2.5 * (seg != "Enterprise") + 1.5 * (source == "Product-led (PQL)")
        for j, c in [(2, "Zapier"), (3, "Make"), (4, "Workato"), (5, "Power Automate")]:
            p[j] += 5 * (comp == c)
        p[6] += 3.5 * (seg == "Enterprise") + 1.5 * (accounts.industry[ai] in ("Financial Services", "Healthcare & Life Sciences", "Public Sector"))
        p[7] += 1.5
        p[8] += 1.2
        p[9] += 0.6 * (exit_stage >= 3)
        p[10] += 1.0
        p[11] += 1.5 * (exit_stage <= 2)
        loss = rng.choice(LOSS_REASONS, p=p / p.sum())
    opps.append(dict(opportunity_id=opp_id, account_id=accounts.account_id[ai], opportunity_type=opp_type,
                     lead_source=source, owner_id=owner, created_date=created, currency=cur,
                     _won=won, _end=end, _exit_stage=exit_stage, _loss=loss, _ws=ws, _amount_final=amt))
    for (ts, f, o, n) in ev:
        events.append((opp_id, ts, f, o, n, owner))
    return opp_id, won, end, amt


def usage_z_for(ai, when):
    m = int((when.year - START.year) * 12 + when.month - 1)
    m = min(max(m, 0), NM - 1)
    e, g, u = acct_usage_at(ai, m)
    return float(np.clip(0.35 * np.log10(e + 10) - 1.1 + 0.8 * np.clip(g, -1.5, 2.5) + 0.15 * np.log2(u + 1), -2, 2.5))


for created, ai, source, lid, ws in opp_requests:
    if created > AS_OF:
        continue
    if acct_open_until.get(ai, pd.Timestamp("1900-01-01")) > created:
        lead_conv[lid] = ("attached_existing", acct_open_until.get(("oid", ai)))
        continue
    if ai in acct_customer_since:
        lead_conv[lid] = ("existing_customer", None)
        continue
    owner = pick_owner(accounts.region[ai], accounts.segment[ai], created)
    uz = usage_z_for(ai, created)
    o, won, end, amt = simulate_opp("New Business", created, ai, source, owner, uz, ws)
    lead_conv[lid] = ("converted", o)
    acct_open_until[ai] = end + pd.Timedelta(days=150 if not won else 0)
    acct_open_until[("oid", ai)] = o
    if won:
        acct_customer_since[ai] = end

# expansion opportunities for customers, driven by usage growth
for ai, since in list(acct_customer_since.items()):
    t = since + pd.Timedelta(days=int(rng.integers(90, 200)))
    while t < AS_OF:
        uz = usage_z_for(ai, t)
        if rng.random() < sigmoid(-0.8 + 0.9 * uz):
            owner = pick_owner(accounts.region[ai], accounts.segment[ai], t)
            base_amt = SEG_ACV[accounts.segment[ai]] * np.exp(rng.normal(-1.0, 0.5) + 0.2 * uz)
            o, won, end, amt = simulate_opp("Expansion", t, ai, "Account Management", owner, uz, None, base_amt)
            t = end + pd.Timedelta(days=int(rng.integers(90, 240)))
        else:
            t = t + pd.Timedelta(days=int(rng.integers(90, 150)))

opps = pd.DataFrame(opps)
events = pd.DataFrame(events, columns=["opportunity_id", "ts", "field", "old_value", "new_value", "created_by"])
events = events.sort_values(["opportunity_id", "ts"], kind="stable").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Forecast submissions (weekly rep calls, post-Salesforce migration) - uses TRUE timeline
# ---------------------------------------------------------------------------
ev_full = events[events.ts <= AS_OF].copy()
state_fields = ["StageName", "CloseDate", "Amount", "ForecastCategoryName"]
fs_rows = []
weeks = pd.date_range(SF_MIGRATION + pd.Timedelta(days=(7 - SF_MIGRATION.weekday()) % 7), AS_OF, freq="7D")
opp_owner = opps.set_index("opportunity_id").owner_id
opp_cur = opps.set_index("opportunity_id").currency
wide = {}
for f in state_fields:
    sub = ev_full[ev_full.field == f][["opportunity_id", "ts", "new_value"]].rename(columns={"new_value": f})
    wide[f] = sub.sort_values("ts")
for wk in weeks:
    q_start = pd.Timestamp(year=wk.year, month=3 * ((wk.month - 1) // 3) + 1, day=1)
    q_end = q_start + pd.offsets.QuarterEnd(0)
    snap = None
    for f in state_fields:
        s = wide[f][wide[f].ts <= wk].groupby("opportunity_id")[f].last()
        snap = s.to_frame() if snap is None else snap.join(s, how="outer")
    snap = snap.dropna(subset=["StageName"])
    snap["CloseDate"] = pd.to_datetime(snap["CloseDate"])
    snap["amount_eur"] = snap["Amount"].astype(float) * snap.index.map(opp_cur).map(FX_TO_EUR).astype(float)
    snap["owner"] = snap.index.map(opp_owner)
    inq = (snap.CloseDate >= q_start) & (snap.CloseDate <= q_end)
    for owner, g in snap[inq].groupby("owner"):
        won_qtd = g[g.StageName == "Closed Won"].amount_eur.sum()
        commit_open = g[(g.ForecastCategoryName == "Commit") & (~g.StageName.str.startswith("Closed"))].amount_eur.sum()
        best_open = g[(g.ForecastCategoryName == "Best Case") & (~g.StageName.str.startswith("Closed"))].amount_eur.sum()
        pipe_open = g[(g.ForecastCategoryName == "Pipeline") & (~g.StageName.str.startswith("Closed"))].amount_eur.sum()
        bias = rep_bias[owner]
        commit_call = won_qtd + commit_open * bias * rng.uniform(0.62, 0.75)  # reps call a judgement share of their Commit bucket
        fs_rows.append(dict(submission_week=wk.date(), fiscal_quarter=f"{q_start.year}-Q{(q_start.month - 1)//3 + 1}",
                            rep_id=owner, closed_won_qtd_eur=round(won_qtd, 0),
                            commit_call_eur=round(commit_call, -2),
                            best_case_call_eur=round(commit_call + best_open * min(1, bias) * rng.uniform(0.5, 0.8), -2),
                            open_pipeline_in_quarter_eur=round(commit_open + best_open + pipe_open, 0)))
forecast = pd.DataFrame(fs_rows)
# a few missed submissions (hygiene)
forecast = forecast.drop(forecast.sample(frac=0.04, random_state=SEED).index)

# ---------------------------------------------------------------------------
# 7. Apply Salesforce migration quirk + truncate at AS_OF
# ---------------------------------------------------------------------------
events = events[events.ts <= AS_OF].copy()
first_ts = events.groupby("opportunity_id").ts.min()
pre = first_ts[first_ts < SF_MIGRATION].index
keep_post = events[~events.opportunity_id.isin(pre) | (events.ts >= SF_MIGRATION)]
mig_rows = []
for opp_id, g in events[events.opportunity_id.isin(pre) & (events.ts < SF_MIGRATION)].groupby("opportunity_id"):
    last = g[g.field != "Created"].groupby("field").new_value.last()
    for f, v in last.items():
        if f == "ForecastCategoryName":
            continue  # field did not exist in legacy CRM
        mig_rows.append((opp_id, SF_MIGRATION + pd.Timedelta(hours=3), f, None, v, "migration_user"))
events = pd.concat([keep_post, pd.DataFrame(mig_rows, columns=events.columns)]).sort_values(["opportunity_id", "ts"], kind="stable")
events = events[events.field != "Created"]
events["created_by"] = np.where(events.created_by == "migration_user", "migration_user", events.created_by)
events["created_by"] = np.where((events.created_by != "migration_user") & (rng.random(len(events)) < 0.06), "005X0001", events.created_by)  # RevOps admin edits

# current state per opp
cur_state = events.groupby(["opportunity_id", "field"]).new_value.last().unstack()
opps = opps.set_index("opportunity_id")
opps["stage_name"] = cur_state["StageName"]
opps["amount"] = pd.to_numeric(cur_state["Amount"])
opps["close_date"] = pd.to_datetime(cur_state["CloseDate"]).dt.date
opps["forecast_category"] = cur_state.get("ForecastCategoryName")
opps["is_closed"] = opps.stage_name.str.startswith("Closed")
opps["is_won"] = opps.stage_name == "Closed Won"
opps["loss_reason"] = np.where(opps.stage_name == "Closed Lost", opps._loss, None)
opps["primary_competitor"] = opps.account_id.map(accounts.set_index("account_id").tech_stack_competitor).replace("None", None)
opps["primary_competitor"] = np.where(rng.random(len(opps)) < 0.45, None, opps.primary_competitor)  # rarely filled
last_act = events.groupby("opportunity_id").ts.max()
opps["last_activity_date"] = (last_act + pd.to_timedelta(rng.integers(0, 10, len(last_act)), unit="D")).clip(upper=AS_OF).dt.date
opps["created_date"] = pd.to_datetime(opps.created_date).dt.date
opps["opportunity_name"] = [f"{accounts.account_name[acc_idx[a]].split(' ')[0]} - {t}" for a, t in zip(opps.account_id, opps.opportunity_type)]
opps["primary_workspace_id"] = [f"ws_{int(w):06d}" if w is not None and not pd.isna(w) else None for w in opps._ws]
opps["next_step"] = np.where(opps.is_closed, None, rng.choice(["Tech validation call", "Security questionnaire", "Send proposal",
                                                                  "Legal review", "Follow up", None], len(opps)))
# hygiene: missing / zero amounts, loss reason blank, stage-in-past close date on open deals remains naturally
ix = opps.sample(frac=0.025, random_state=1).index
opps.loc[ix, "amount"] = np.where(rng.random(len(ix)) < 0.5, np.nan, 0)
ix = opps[opps.stage_name == "Closed Lost"].sample(frac=0.12, random_state=2).index
opps.loc[ix, "loss_reason"] = np.where(rng.random(len(ix)) < 0.6, None, "Other")
opps.loc[opps.sample(frac=0.03, random_state=3).index, "lead_source"] = None
# open deals left with AEs who have since left the company (owner never reassigned)
leaver_ids = reps.loc[leavers, "rep_id"].values
ix = opps[~opps.is_closed].sample(n=9, random_state=8).index
opps.loc[ix, "owner_id"] = rng.choice(leaver_ids, len(ix))
opps = opps.reset_index()
opp_cols = ["opportunity_id", "opportunity_name", "account_id", "opportunity_type", "lead_source", "owner_id",
            "created_date", "close_date", "stage_name", "forecast_category", "amount", "currency", "is_closed",
            "is_won", "loss_reason", "primary_competitor", "primary_workspace_id", "next_step", "last_activity_date"]
opps_out = opps[opp_cols]

events_out = events.rename(columns={"ts": "created_at"})
events_out["history_id"] = [f"017H{i:08d}" for i in range(len(events_out))]
events_out["created_at"] = events_out.created_at.dt.strftime("%Y-%m-%d %H:%M:%S")
for c in ("old_value", "new_value"):
    events_out[c] = events_out[c].map(lambda v: v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v)
events_out = events_out[["history_id", "opportunity_id", "field", "old_value", "new_value", "created_by", "created_at"]]

# ---------------------------------------------------------------------------
# 8. Leads output (status + conversion)
# ---------------------------------------------------------------------------
status, conv_opp, conv_date = [], [], []
opp_created = opps.set_index("opportunity_id").created_date
for lid, created in zip(leads.lead_id, leads.created_date):
    c = lead_conv.get(lid)
    if c and c[0] == "converted":
        status.append("Converted"); conv_opp.append(c[1]); conv_date.append(opp_created[c[1]])
    elif c and c[0] == "attached_existing":
        status.append("Disqualified - Existing opportunity"); conv_opp.append(None); conv_date.append(None)
    elif c and c[0] == "existing_customer":
        status.append("Disqualified - Existing customer"); conv_opp.append(None); conv_date.append(None)
    else:
        age = (AS_OF - created).days
        if age < 30:
            status.append(rng.choice(["Open", "Working"], p=[0.5, 0.5]))
        else:
            status.append(rng.choice(["Nurture", "Disqualified - No fit", "Disqualified - No response", "Disqualified - Student / hobbyist"],
                                     p=[0.45, 0.2, 0.25, 0.1]))
        conv_opp.append(None); conv_date.append(None)
leads["status"] = status
leads["converted_opportunity_id"] = conv_opp
leads["converted_date"] = conv_date
leads["created_date"] = pd.to_datetime(leads.created_date).dt.date
own = []
for src, acc in zip(leads.lead_source, leads.account_id):
    if src == "SDR Outbound" or src.startswith("Paid") or src in ("Webinar", "Content download", "Event / Trade show"):
        reg = accounts.region[acc_idx[acc]] if isinstance(acc, str) else "North America"
        c = sdrs[sdrs.region == reg]
        own.append((c if len(c) else sdrs).rep_id.values[rng.integers(0, len(c) if len(c) else len(sdrs))])
    else:
        own.append(None)
leads["owner_id"] = own
leads = leads.sort_values("created_date").reset_index(drop=True)
leads = leads[["lead_id", "created_date", "lead_source", "account_id", "workspace_id", "email_domain", "pql_rule_v1_flag",
               "status", "converted_opportunity_id", "converted_date", "owner_id"]]

# ---------------------------------------------------------------------------
# 9. Contracts -> monthly ARR movements (+ usage overage) for sales-led customers
# ---------------------------------------------------------------------------
won = opps[opps.is_won].copy()
won["amt_eur"] = won._amount_final * won.currency.map(FX_TO_EUR)
won["end"] = pd.to_datetime(won._end)
arr_rows = []
ent_since_month = {}
for ai_id, g in won.sort_values("end").groupby("account_id"):
    ai = acc_idx[ai_id]
    nb = g[g.opportunity_type == "New Business"]
    if nb.empty:
        continue
    start = nb.end.iloc[0]
    sm = (start.year - START.year) * 12 + start.month - 1 + 1  # billing starts next month
    ent_since_month[ai] = sm
    arr = 0.0
    renew_m = sm + 12
    exp_by_m = {}
    for _, r in g.iterrows():
        m = (r.end.year - START.year) * 12 + r.end.month - 1 + 1
        exp_by_m[m] = exp_by_m.get(m, 0) + r.amt_eur
    churned = False
    for m in range(sm, NM):
        mv = {"new": 0.0, "expansion": 0.0, "contraction": 0.0, "churn": 0.0}
        if m == sm:
            mv["new"] = exp_by_m.pop(m, 0.0)
            arr = mv["new"]
        elif m in exp_by_m:
            mv["expansion"] = exp_by_m.pop(m)
            arr += mv["expansion"]
        if m == renew_m:
            uz = usage_z_for(ai, MONTHS[m])
            p_churn = sigmoid(-1.3 - 1.0 * uz - 0.4 * acc_fit[ai])
            u = rng.random()
            if u < p_churn:
                mv["churn"] = -arr; arr = 0; churned = True
            elif u < p_churn + 0.15:
                c = -round(arr * rng.uniform(0.1, 0.35), -2); mv["contraction"] = c; arr += c
            elif uz > 0.5 and rng.random() < 0.4:
                e = round(arr * rng.uniform(0.05, 0.25), -2); mv["expansion"] += e; arr += e
            renew_m += 12
        # usage overage (executions above contracted allowance)
        w = np.where(ws_acc == ai)[0]
        execs = exec_hist[w, m].sum() if len(w) else 0
        allowance = arr / 12 * 60  # ~60 executions included per EUR of MRR
        overage = max(0.0, execs - allowance) * 0.004 if arr > 0 else 0.0
        arr_rows.append(dict(month=MONTHS[m].date(), account_id=ai_id, arr_eur=round(arr, 2),
                             new_arr_eur=round(mv["new"], 2), expansion_arr_eur=round(mv["expansion"], 2),
                             contraction_arr_eur=round(mv["contraction"], 2), churned_arr_eur=round(mv["churn"], 2),
                             usage_overage_revenue_eur=round(overage, 2), total_executions=int(execs)))
        if churned:
            break
arr_monthly = pd.DataFrame(arr_rows)

# ---------------------------------------------------------------------------
# 10. Workspace tables
# ---------------------------------------------------------------------------
ws_acc_id = np.array([accounts.account_id[a] if a >= 0 else None for a in ws_acc], dtype=object)
usage_rows = []
for m, idx, ex, us, pr in rows:
    if MONTHS[m] > AS_OF:
        continue
    ent = np.array([ent_since_month.get(a, 10**6) <= m if a >= 0 else False for a in ws_acc[idx]])
    plan = np.where(hosting[idx] == "cloud", np.array(PLANS)[pr], np.where(hosting[idx] == "self_hosted_business", "Business (self-hosted)", "Community (free)"))
    plan = np.where(ent, "Enterprise", plan)
    tel = telemetry[idx]
    ai_share = np.clip(sigmoid(-2.6 + 0.09 * m + 0.4 * fit_ws[idx]) + rng.normal(0, 0.04, len(idx)), 0, 0.95)
    wf = np.maximum(1, np.round(0.9 * np.power(ex + 1, 0.42) + rng.normal(0, 1.5, len(idx))))
    integ = np.maximum(1, np.round(1.5 + 1.3 * np.log1p(ex / 200) + rng.normal(0, 1, len(idx))))
    mrr = np.array([PLAN_MRR.get(p, 0.0) for p in plan])
    df = pd.DataFrame({
        "month": MONTHS[m].date(), "workspace_id": [f"ws_{i:06d}" for i in idx], "plan": plan,
        "executions": np.where(tel, ex, np.nan), "active_workflows": np.where(tel, wf, np.nan),
        "active_users": np.where(tel, us, np.nan), "ai_node_executions": np.where(tel, np.round(ex * ai_share), np.nan),
        "distinct_integrations_used": np.where(tel, integ, np.nan),
        "failed_execution_rate": np.where(tel, np.round(np.clip(rng.beta(1.5, 30, len(idx)) + 0.03 * (fit_ws[idx] < -1), 0, 1), 4), np.nan),
        "self_serve_mrr_eur": mrr,
    })
    usage_rows.append(df)
usage = pd.concat(usage_rows, ignore_index=True)

last_m = np.array([churn_month[i] if churn_month[i] >= 0 else NM - 1 for i in range(N_WS)])
workspaces = pd.DataFrame({
    "workspace_id": [f"ws_{i:06d}" for i in range(N_WS)],
    "account_id": ws_acc_id,
    "signup_date": ws_signup.date,
    "hosting_type": hosting,
    "signup_source": signup_src,
    "telemetry_enabled": telemetry,
    "owner_email_domain": [accounts.website_domain[a] if a >= 0 else rng.choice(["gmail.com", "outlook.com", "proton.me", "icloud.com", "yahoo.com"]) for a in ws_acc],
    "is_active": churn_month < 0,
    "churned_month": [MONTHS[c].date() if c >= 0 else None for c in churn_month],
})
workspaces["signup_date"] = pd.to_datetime(workspaces.signup_date).clip(upper=AS_OF).dt.date  # keep every workspace; cap sign-up at snapshot

# ---------------------------------------------------------------------------
# 11. Quotas (per AE per quarter, ramped) - calibrated to realistic attainment
# ---------------------------------------------------------------------------
quarters = pd.period_range("2024Q1", "2026Q4", freq="Q")
bk = won.copy()
bk["q"] = bk.end.dt.to_period("Q")
quarters = quarters[1:]  # 2024-Q1 is pipeline warm-up (data starts Jan 2024), no quotas set
bk_rep_q = bk.groupby(["owner_id", "q"]).amt_eur.sum()
qrows = []
for _, r in reps[ae_mask].iterrows():
    territory_difficulty = rng.uniform(0.9, 1.35)  # some territories are over-assigned
    rq = []
    for q in quarters:
        qs, qe = q.start_time, q.end_time
        if r.hire_date > qe or (pd.notna(r.termination_date) and r.termination_date < qs):
            continue
        months_in = max(0, (qs - r.hire_date).days / 30.4)
        ramp = 0.25 if months_in < 1 else 0.5 if months_in < 3 else 0.75 if months_in < 5 else 1.0
        rq.append((q, qs, ramp))
    # quota set annually from territory potential (smoothed bookings capacity) with a stretch factor
    for yr in sorted({q.year for q, _, _ in rq}):
        yq = [(q, qs, rp) for q, qs, rp in rq if q.year == yr]
        obs = [(bk_rep_q.get((r.rep_id, q), 0.0), rp) for q, qs, rp in yq if q.end_time <= AS_OF]
        prev = [(bk_rep_q.get((r.rep_id, q), 0.0), rp) for q, qs, rp in rq if q.year == yr - 1]
        pool = obs + prev if obs else prev
        cap = sum(b for b, _ in pool) / max(sum(rp for _, rp in pool), 0.25) if pool else 0
        if cap <= 0:
            cap = {"Commercial": 90000, "Mid-Market": 200000, "Enterprise": 380000}[r.segment]
        for q, qs, rp in yq:
            qrows.append(dict(rep_id=r.rep_id, fiscal_quarter=f"{q.year}-Q{q.quarter}", quota_start_date=qs.date(),
                              ramp_pct=rp, quota_eur=round(cap * territory_difficulty * rp * rng.uniform(0.97, 1.03), -3)))
quotas = pd.DataFrame(qrows)

# ---------------------------------------------------------------------------
# 12. Marketing spend
# ---------------------------------------------------------------------------
ms = []
chan_cost = {"Webinar": 110, "Content download": 60, "Event / Trade show": 900, "Paid social": 180, "Paid search - Enterprise LP": 260,
             "Paid search - Self-serve": None, "Community sponsorships": None, "Partner MDF": None}
lead_m = leads.assign(month=pd.to_datetime(leads.created_date).dt.to_period("M"))
lead_m["region"] = lead_m.account_id.map(accounts.set_index("account_id").region).fillna("Unknown")
cnt = lead_m.groupby(["month", "lead_source", "region"]).size()
for (mo, src, reg), n in cnt.items():
    if src in chan_cost and chan_cost[src]:
        ms.append(dict(month=mo.start_time.date(), channel=src, region=reg, spend_eur=round(n * chan_cost[src] * rng.uniform(0.8, 1.25), 0)))
for m in MONTHS:
    for reg in ["EMEA - DACH", "EMEA - UK&I", "EMEA - North", "EMEA - South", "North America", "APAC", "LATAM"]:
        w = {"North America": 1.6, "EMEA - DACH": 1.2}.get(reg, 0.6)
        ms.append(dict(month=m.date(), channel="Paid search - Self-serve", region=reg, spend_eur=round(18000 * w * rng.uniform(0.8, 1.2) * (1 + 0.02 * ((m.year - 2024) * 12 + m.month)), 0)))
        ms.append(dict(month=m.date(), channel="Community sponsorships", region=reg, spend_eur=round(4000 * w * rng.uniform(0.5, 1.5), 0)))
        ms.append(dict(month=m.date(), channel="Partner MDF", region=reg, spend_eur=round(2500 * w * rng.uniform(0.5, 1.5), 0)))
mspend = pd.DataFrame(ms).sort_values(["month", "channel", "region"])

# ---------------------------------------------------------------------------
# 13. Account hygiene issues: duplicates + messy industry labels
# ---------------------------------------------------------------------------
acc_out = accounts.copy()
dups = acc_out.sample(n=140, random_state=4).copy()
dups["account_id"] = [f"001D{i:06d}" for i in range(len(dups))]
dups["account_name"] = [n.upper() if i % 3 == 0 else n.replace(" ", "  ", 1) if i % 3 == 1 else n.rsplit(" ", 1)[0] for i, n in enumerate(dups.account_name)]
dups["employee_count"] = np.where(rng.random(len(dups)) < 0.5, np.nan, dups.employee_count)
dups["created_date"] = [SF_MIGRATION.date()] * len(dups)
acc_out = pd.concat([acc_out, dups], ignore_index=True)
# re-point ~40 opportunities to duplicate records
dmap = dict(zip(dups.index.map(lambda i: accounts.account_id[i]), dups.account_id))
cand = opps_out[opps_out.account_id.isin(dmap)].sample(frac=0.5, random_state=5).index
opps_out.loc[cand, "account_id"] = opps_out.loc[cand, "account_id"].map(dmap)
for canon, variants in ind_variants.items():
    ix = acc_out[acc_out.industry == canon].sample(frac=0.25, random_state=6).index
    acc_out.loc[ix, "industry"] = rng.choice(variants, len(ix))
acc_out.loc[acc_out.sample(frac=0.04, random_state=7).index, "industry"] = None
acc_out = acc_out.drop(columns=["tech_stack_competitor"])
acc_out["employee_count"] = acc_out.employee_count.astype("Int64")

# fx table
fx = pd.DataFrame([{"currency": k, "rate_to_eur": v} for k, v in FX_TO_EUR.items()])

reps_out = reps.copy()
reps_out["hire_date"] = reps_out.hire_date.dt.date
reps_out["termination_date"] = reps_out.termination_date.dt.date
reps_out["is_active"] = reps_out.termination_date.isna()
reps_out = pd.concat([reps_out, pd.DataFrame([dict(rep_id="005X0001", rep_name="RevOps Admin", role="Sales Operations",
                                                   segment="All", region="Global", manager_id=None,
                                                   hire_date=pd.Timestamp("2024-11-01").date(), termination_date=None, is_active=True)])])

# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
tables = {
    "accounts": acc_out, "sales_reps": reps_out, "quotas": quotas, "workspaces": workspaces,
    "workspace_usage_monthly": usage, "leads": leads, "opportunities": opps_out,
    "opportunity_field_history": events_out, "forecast_submissions": forecast,
    "arr_monthly": arr_monthly, "marketing_spend": mspend, "fx_rates": fx,
}
for n, df in tables.items():
    df.to_csv(OUT / f"{n}.csv", index=False)
    print(f"{n:28s} {len(df):>8,d} rows")
