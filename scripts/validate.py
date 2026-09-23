import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
D="data/"
L=pd.read_csv(D+"leads.csv",parse_dates=["created_date"]); o=pd.read_csv(D+"opportunities.csv",parse_dates=["created_date","close_date"])
a=pd.read_csv(D+"accounts.csv"); fx=dict(pd.read_csv(D+"fx_rates.csv").values)
L["conv"]=L.converted_opportunity_id.notna()
print(L.groupby("lead_source").conv.agg(["mean","size"]).round(3))
print(o.opportunity_type.value_counts().to_dict(), "open:", (~o.is_closed).sum())
c=o[o.is_closed]; print("win:",c.groupby("opportunity_type").is_won.mean().round(3).to_dict())
print(c.groupby("lead_source").is_won.agg(["mean","size"]).round(3))
p=L[(L.lead_source=="Product (PQL)")&L.workspace_id.notna()&L.account_id.notna()&~L.status.str.contains("Existing")].copy()
p["month"]=p.created_date.dt.to_period("M").dt.to_timestamp()
u=pd.read_csv(D+"workspace_usage_monthly.csv",parse_dates=["month"]).sort_values(["workspace_id","month"])
u["ex_lag3"]=u.groupby("workspace_id").executions.shift(3)
d=p.merge(u,on=["workspace_id","month"],how="left").merge(a[["account_id","employee_count"]],on="account_id",how="left")
d["growth3"]=np.log((d.executions+2000)/(d.ex_lag3.fillna(0)+2000)); d["lemp"]=np.log10(d.employee_count)
for col in ["growth3","active_users","lemp","executions"]:
    print(col, d.groupby(pd.qcut(d[col].rank(method="first"),4)).conv.mean().round(3).values)
dd=d.dropna(subset=["growth3","active_users","lemp","executions"])
X=np.c_[dd.growth3.clip(-3,3),np.log2(dd.active_users+1),dd.lemp,np.log10(dd.executions+1)]
pr=cross_val_predict(LogisticRegression(max_iter=500),X,dd.conv,cv=5,method="predict_proba")[:,1]
print("PQL eligible n",len(dd),"conv",round(dd.conv.mean(),3),"CV AUC",round(roc_auc_score(dd.conv,pr),3), "| rule-only baseline = all flagged")
f=pd.read_csv(D+"forecast_submissions.csv",parse_dates=["submission_week"])
o["amt_eur"]=o.amount*o.currency.map(fx); o["fq"]=o.close_date.dt.year.astype(str)+"-Q"+o.close_date.dt.quarter.astype(str)
act=o[o.is_won].groupby(["owner_id","fq"]).amt_eur.sum().rename("actual")
f=f.join(act,on=["rep_id","fiscal_quarter"]).fillna({"actual":0}); f=f[f.fiscal_quarter<"2026-Q3"]
f["wk"]=f.groupby(["rep_id","fiscal_quarter"]).submission_week.rank()
g=f.groupby("wk")[["commit_call_eur","actual"]].sum(); print("commit/actual by week:", (g.commit_call_eur/g.actual).round(2).to_dict())
r=f[f.wk==6].groupby("rep_id")[["commit_call_eur","actual"]].sum(); print("rep-level call/actual wk6:", (r.commit_call_eur/r.actual).round(2).sort_values().values)
print("amount null/zero:", o.amount.isna().sum(), (o.amount==0).sum(), "| opps on dup accounts:", o.account_id.str.startswith("001D").sum())
print("open past close:", ((~o.is_closed)&(o.close_date<"2026-09-21")).sum())
r=pd.read_csv(D+"sales_reps.csv"); print("open opps owned by inactive reps:", o[~o.is_closed].owner_id.isin(r[~r.is_active].rep_id).sum())
h=pd.read_csv(D+"opportunity_field_history.csv"); print("history migration/admin share:", h.created_by.isin(["migration_user","005X0001"]).mean().round(3))
print("FK ok:", o.account_id.isin(a.account_id).mean(), h.opportunity_id.isin(o.opportunity_id).mean(), L.converted_opportunity_id.dropna().isin(o.opportunity_id).mean())
arr=pd.read_csv(D+"arr_monthly.csv"); print("ARR last months:", arr.groupby("month").arr_eur.sum().tail(3).round(-3).to_dict())
print("12m movements:", arr[arr.month>"2025-09-01"][["new_arr_eur","expansion_arr_eur","contraction_arr_eur","churned_arr_eur","usage_overage_revenue_eur"]].sum().round(-3).to_dict())
q=pd.read_csv(D+"quotas.csv"); b=o[o.is_won].groupby(["owner_id","fq"]).amt_eur.sum().rename("b")
q=q.join(b,on=["rep_id","fiscal_quarter"]).fillna({"b":0}); q=q[q.fiscal_quarter<"2026-Q3"]; q["att"]=q.b/q.quota_eur
print("attainment median", q.att.median().round(2), "share>=100%", (q.att>=1).mean().round(2))
