# Sale_Analytics: synthetic GTM dataset (PLG + enterprise SaaS)

Synthetic data for a workflow-automation SaaS company with two commercial motions:
a self-serve product (cloud plans plus a free self-hosted Community edition, priced on executions)
and an enterprise sales team that works the accounts that grow out of it. The shape is modelled loosely
on a company like n8n. All companies, people and numbers are invented.

- Snapshot date ("today"): **2026-09-21**
- History: **Jan 2024 to Sep 2026** (2024-Q1 is a warm-up quarter with little closed business, so no quotas are set for it)
- Salesforce migration: **2025-01-15** (see "Known data quirks")
- Regions: EMEA - DACH, EMEA - UK&I, EMEA - North, EMEA - South, North America, LATAM, APAC
- CSV format: UTF-8, comma separated, header row, empty field = NULL, booleans as 1/0, amounts with 2 decimals
- Loading into PostgreSQL: see the main README
- Generated with `scripts/generate_data.py` (numpy + pandas, fixed seed 42), then typed and checked with `scripts/build_sql_package.py`

## Tables

| File | Rows | Grain | What it is |
|---|---|---|---|
| accounts.csv | 9,140 | account | Salesforce Account: firmographics, segment, region |
| sales_reps.csv | 36 | user | AEs, SDRs, managers, one RevOps admin; hire and exit dates |
| quotas.csv | 187 | AE x quarter | Quarterly bookings quota (EUR), with ramp % for new hires |
| workspaces.csv | 24,000 | workspace | Product sign-ups, linked to an account where the email domain matched |
| workspace_usage_monthly.csv | 212,783 | workspace x month | Executions, workflows, users, AI node usage, plan, self-serve MRR |
| leads.csv | 26,367 | lead | PQLs, demo requests, marketing MQLs, SDR outbound, partner referrals |
| opportunities.csv | 4,200 | opportunity | Current state of each opportunity (New Business and Expansion) |
| opportunity_field_history.csv | 43,045 | field change | Salesforce-style history of Stage, CloseDate, Amount, ForecastCategory |
| forecast_submissions.csv | 1,617 | AE x week | Weekly rep forecast calls (commit, best case) since the migration |
| arr_monthly.csv | 9,114 | account x month | Contracted ARR and movements for sales-led customers, plus usage overage revenue |
| marketing_spend.csv | 1,983 | month x channel x region | GTM spend in EUR |
| fx_rates.csv | 3 | currency | Conversion rates to EUR (USD 0.92, GBP 1.17) |

## Key columns

**accounts**: `account_id`, `account_name`, `website_domain`, `industry`, `employee_count`, `segment` (Commercial <250, Mid-Market 250-1,999, Enterprise 2,000+), `billing_country`, `region`, `annual_revenue_eur`, `created_date`

**sales_reps**: `rep_id`, `rep_name`, `role` (Account Executive / Sales Development Rep / Sales Manager / Sales Operations), `segment`, `region` (the rep's territory), `manager_id`, `hire_date`, `termination_date`, `is_active`

**quotas**: `rep_id`, `fiscal_quarter` (calendar quarters, e.g. 2025-Q3), `quota_start_date`, `ramp_pct` (0.25 / 0.5 / 0.75 / 1.0), `quota_eur`. Quota covers New Business + Expansion bookings (closed won amount in EUR).

**workspaces**: `workspace_id`, `account_id` (blank for personal emails), `signup_date`, `hosting_type` (cloud / self_hosted_community / self_hosted_business), `signup_source`, `telemetry_enabled`, `owner_email_domain`, `is_active`, `churned_month`

**workspace_usage_monthly**: `month`, `workspace_id`, `plan` (Starter / Pro / Business / Enterprise / Community (free) / Business (self-hosted)), `executions`, `active_workflows`, `active_users`, `ai_node_executions`, `distinct_integrations_used`, `failed_execution_rate`, `self_serve_mrr_eur` (0 once the account is on an Enterprise contract, billed through arr_monthly instead)

**leads**: `lead_id`, `created_date`, `lead_source`, `account_id`, `workspace_id` (PQLs and demo requests), `email_domain`, `pql_rule_v1_flag`, `status`, `converted_opportunity_id`, `converted_date`, `owner_id` (SDR, where assigned)

**opportunities**: `opportunity_id`, `opportunity_name`, `account_id`, `opportunity_type` (New Business / Expansion), `lead_source`, `owner_id`, `created_date`, `close_date`, `stage_name` (1 - Discovery, 2 - Qualification, 3 - Technical Validation, 4 - Proposal, 5 - Negotiation, Closed Won, Closed Lost), `forecast_category` (Pipeline / Best Case / Commit / Closed / Omitted), `amount` (in `currency`), `currency`, `is_closed`, `is_won`, `loss_reason`, `primary_competitor`, `primary_workspace_id`, `next_step`, `last_activity_date`

**opportunity_field_history**: `history_id`, `opportunity_id`, `field` (StageName / CloseDate / Amount / ForecastCategoryName), `old_value`, `new_value`, `created_by` (rep id, `migration_user`, or RevOps admin `005X0001`), `created_at`. The first row per field has an empty `old_value` (value at creation).

**forecast_submissions**: `submission_week` (Mondays), `fiscal_quarter`, `rep_id`, `closed_won_qtd_eur`, `commit_call_eur`, `best_case_call_eur`, `open_pipeline_in_quarter_eur`

**arr_monthly**: `month`, `account_id`, `arr_eur` (end of month), `new_arr_eur`, `expansion_arr_eur`, `contraction_arr_eur`, `churned_arr_eur`, `usage_overage_revenue_eur` (executions above the contracted allowance), `total_executions`

## How the tables join

```
accounts ─┬─< workspaces ─< workspace_usage_monthly
          ├─< leads >── workspaces (workspace_id, PQLs / demo requests)
          ├─< opportunities ─< opportunity_field_history
          │        └── sales_reps (owner_id) ─< quotas, forecast_submissions
          └─< arr_monthly
leads.converted_opportunity_id → opportunities.opportunity_id
opportunities.currency → fx_rates.currency
```

## Behaviour built into the data (things you can discover)

- Product-led opportunities win more often and close faster than marketing or outbound ones.
- Usage momentum (execution growth over the last ~3 months), company size and number of active users drive whether a PQL turns into a real opportunity. The current `pql_rule_v1_flag` (15k+ executions in a month, 5+ active users, or Business plan) flags far more workspaces than sales can work, so there is room for a better score.
- Marketing MQLs and outbound leads convert much better when the account already has active product usage.
- Deals whose close date has been pushed several times are much less likely to be won. Lost deals tend to sit open for a while before being closed out.
- Reps differ in forecasting behaviour: some consistently over-call, some sandbag. Commit calls drift up mid-quarter and then slip at quarter end.
- Loss reasons have structure: "Chose self-hosted Community edition" is common in smaller accounts, security/compliance reviews in Enterprise and regulated industries, and competitor losses line up with the tool the account already used.
- Renewal churn and expansion depend on usage trends.
- Territories are not equally loaded, so quota attainment varies by rep for reasons beyond skill.

## Known data quirks (deliberate, like a real CRM)

- **Salesforce migration (2025-01-15):** opportunities created before the migration only have history from the migration date (one `migration_user` row per field). `ForecastCategoryName` did not exist in the legacy CRM, and weekly forecast submissions start after migration.
- **Multi-currency:** amounts are in the deal currency (EUR, USD, GBP). Convert with `fx_rates.csv` before summing.
- **Missing or zero amounts** on about 2.5% of opportunities.
- **Stage skipping:** about 10% of deals jump over a stage in the history.
- **Open deals with a close date in the past**, and a few open deals still owned by AEs who have left.
- **Duplicate accounts:** 140 accounts have a duplicate record (ids starting `001D`, names in different case/spacing or without the legal suffix); some opportunities are attached to the duplicate.
- **Inconsistent industry labels** ("Software", "SaaS", "Computer Software" for the same thing) and some blanks.
- **Blank or "Other" loss reasons** on about 12% of lost deals; `primary_competitor` is rarely filled in.
- **Lead-to-account matching:** some leads have no `account_id`; PQLs from personal email domains cannot be matched.
- **Self-hosted telemetry:** about a third of Community (self-hosted) workspaces have no usage telemetry (blank metrics).
- **History edits by RevOps admin** (`005X0001`) mixed in with rep edits.
- **Missed forecast submissions:** about 4% of rep-weeks are missing.
- **last_activity_date** on deals closed before the migration shows the migration date, because that is the last time the record was touched in Salesforce.
- 2026-Q3 is still in progress on the snapshot date, so its bookings are partial.
