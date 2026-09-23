-- =====================================================================
-- Sale_Analytics | Step 1: staging layer + weekly pipeline snapshot
-- Database: sale_analytics (PostgreSQL). Safe to re-run.
--
-- Layers (same idea as dbt):
--   raw.*       loaded CSVs, untouched
--   staging.*   cleaned, typed, EUR-converted, one model per source
--   marts.*     business tables: the weekly pipeline snapshot
--
-- Snapshot definition: the state of every opportunity as of Monday 00:00
-- each week, rebuilt from Salesforce field history. Weeks start
-- 2025-01-20, the first Monday after the Salesforce migration
-- (history before the migration was not carried over).
-- =====================================================================
\set ON_ERROR_STOP on
SET client_min_messages = warning;

CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS marts;

-- ---------------------------------------------------------------------
-- staging.stg_accounts
-- Finds duplicate account records and maps each to its original (the
-- older record), and standardises industry labels.
-- Many different companies share a website domain here, so a duplicate
-- must match on domain + country + segment + a near-identical name
-- (same name ignoring case/spacing, or the same name without the legal
-- suffix), and must not contradict on employee count.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS staging.stg_accounts CASCADE;
CREATE TABLE staging.stg_accounts AS
WITH named AS (
    SELECT a.*, regexp_replace(lower(trim(account_name)), '\s+', ' ', 'g') AS name_norm
    FROM raw.accounts a
),
dup_match AS (
    SELECT DISTINCT ON (d.account_id) d.account_id, o.account_id AS master_account_id
    FROM named d
    JOIN named o
      ON  o.website_domain  = d.website_domain
      AND o.billing_country = d.billing_country
      AND o.segment         = d.segment
      AND o.created_date    < d.created_date
      AND o.account_id     <> d.account_id
      AND (o.name_norm = d.name_norm OR o.name_norm LIKE d.name_norm || ' %')
      AND (d.employee_count IS NULL OR o.employee_count = d.employee_count)
    ORDER BY d.account_id, o.created_date, o.account_id
)
SELECT
    a.account_id,
    COALESCE(m.master_account_id, a.account_id)                AS master_account_id,
    m.master_account_id IS NOT NULL                            AS is_duplicate_record,
    a.account_name,
    a.website_domain,
    CASE
        WHEN a.industry IS NULL                                          THEN 'Unknown'
        WHEN lower(a.industry) IN ('software', 'saas', 'computer software') THEN 'Software & SaaS'
        WHEN a.industry IN ('Finance', 'Fintech', 'Banking')             THEN 'Financial Services'
        WHEN a.industry IN ('Retail', 'Ecommerce', 'E-Commerce')         THEN 'E-commerce & Retail'
        WHEN a.industry IN ('Consulting', 'Agency')                      THEN 'Professional Services'
        ELSE a.industry
    END                                                         AS industry,
    a.employee_count,
    a.segment,
    a.billing_country,
    a.region
FROM raw.accounts a
LEFT JOIN dup_match m USING (account_id);

-- ---------------------------------------------------------------------
-- staging.stg_opportunities: current state, EUR amounts, master account
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS staging.stg_opportunities CASCADE;
CREATE TABLE staging.stg_opportunities AS
SELECT
    o.opportunity_id,
    o.opportunity_name,
    a.master_account_id                                AS account_id,
    o.opportunity_type,
    COALESCE(o.lead_source, 'Unknown')                 AS lead_source,
    o.owner_id,
    o.created_date,
    o.close_date,
    o.stage_name,
    o.forecast_category,
    NULLIF(o.amount, 0)                                AS amount_local,
    o.currency,
    round(NULLIF(o.amount, 0) * fx.rate_to_eur, 2)     AS amount_eur,
    o.is_closed,
    o.is_won,
    o.loss_reason,
    o.last_activity_date,
    a.segment,
    a.region,
    a.industry
FROM raw.opportunities o
JOIN raw.fx_rates fx            ON fx.currency = o.currency
JOIN staging.stg_accounts a     ON a.account_id = o.account_id;

-- ---------------------------------------------------------------------
-- staging.stg_opportunity_history: one row per field change, typed,
-- with the time window during which each value was valid.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS staging.stg_opportunity_history CASCADE;
CREATE TABLE staging.stg_opportunity_history AS
SELECT
    h.history_id,
    h.opportunity_id,
    h.field,
    h.old_value,
    h.new_value,
    h.created_by,
    h.created_by = 'migration_user'                                   AS is_migration_row,
    h.created_at                                                      AS valid_from,
    COALESCE(
        lead(h.created_at) OVER (PARTITION BY h.opportunity_id, h.field ORDER BY h.created_at, h.history_id),
        'infinity'::timestamp)                                        AS valid_to
FROM raw.opportunity_field_history h;

CREATE INDEX ON staging.stg_opportunity_history (field, opportunity_id, valid_from, valid_to);

-- ---------------------------------------------------------------------
-- marts.dim_week: Mondays from first post-migration week to snapshot date
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS marts.dim_week CASCADE;
CREATE TABLE marts.dim_week AS
SELECT
    d::date                                                         AS snapshot_week,
    to_char(d, 'YYYY') || '-Q' || to_char(d, 'Q')                   AS fiscal_quarter,
    date_trunc('quarter', d)::date                                  AS quarter_start,
    (date_trunc('quarter', d) + interval '3 months - 1 day')::date  AS quarter_end,
    ((d::date - date_trunc('quarter', d)::date) / 7) + 1            AS week_of_quarter
FROM generate_series('2025-01-20'::date, '2026-09-21'::date, interval '7 days') d;

-- ---------------------------------------------------------------------
-- marts.fct_pipeline_snapshot_weekly
-- Grain: one row per opportunity per week, for every opportunity that
-- existed at that point (open or already closed).
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS marts.fct_pipeline_snapshot_weekly CASCADE;
CREATE TABLE marts.fct_pipeline_snapshot_weekly AS
WITH state AS (
    SELECT
        w.snapshot_week, w.fiscal_quarter, w.quarter_start, w.quarter_end, w.week_of_quarter,
        o.opportunity_id,
        st.new_value                       AS stage_name,
        st.valid_from                      AS stage_entered_at,
        am.new_value::numeric              AS amount_local,
        cd.new_value::date                 AS close_date,
        fc.new_value                       AS forecast_category_raw
    FROM marts.dim_week w
    JOIN staging.stg_opportunities o
      ON o.created_date < w.snapshot_week
    JOIN staging.stg_opportunity_history st
      ON st.opportunity_id = o.opportunity_id AND st.field = 'StageName'
     AND st.valid_from < w.snapshot_week AND st.valid_to >= w.snapshot_week
    LEFT JOIN staging.stg_opportunity_history am
      ON am.opportunity_id = o.opportunity_id AND am.field = 'Amount'
     AND am.valid_from < w.snapshot_week AND am.valid_to >= w.snapshot_week
    LEFT JOIN staging.stg_opportunity_history cd
      ON cd.opportunity_id = o.opportunity_id AND cd.field = 'CloseDate'
     AND cd.valid_from < w.snapshot_week AND cd.valid_to >= w.snapshot_week
    LEFT JOIN staging.stg_opportunity_history fc
      ON fc.opportunity_id = o.opportunity_id AND fc.field = 'ForecastCategoryName'
     AND fc.valid_from < w.snapshot_week AND fc.valid_to >= w.snapshot_week
),
pushes AS (
    -- close-date pushes (moved later) made before each snapshot week
    SELECT w.snapshot_week, h.opportunity_id, count(*) AS close_date_pushes
    FROM marts.dim_week w
    JOIN staging.stg_opportunity_history h
      ON h.field = 'CloseDate' AND h.valid_from < w.snapshot_week
     AND NOT h.is_migration_row AND h.old_value IS NOT NULL
     AND h.new_value::date > h.old_value::date
    GROUP BY 1, 2
)
SELECT
    s.snapshot_week,
    s.fiscal_quarter,
    s.week_of_quarter,
    s.opportunity_id,
    o.account_id,
    o.owner_id,
    o.opportunity_type,
    o.lead_source,
    o.segment,
    o.region,
    s.stage_name,
    CASE WHEN s.stage_name ~ '^[1-5] ' THEN left(s.stage_name, 1)::int END      AS stage_number,
    s.stage_name NOT IN ('Closed Won', 'Closed Lost')                           AS is_open,
    s.stage_name = 'Closed Won'                                                 AS is_closed_won,
    -- forecast category did not exist before the migration: fill from stage and flag it
    COALESCE(s.forecast_category_raw,
        CASE WHEN s.stage_name = 'Closed Won'  THEN 'Closed'
             WHEN s.stage_name = 'Closed Lost' THEN 'Omitted'
             WHEN s.stage_name ~ '^[12] '      THEN 'Pipeline'
             WHEN s.stage_name ~ '^[34] '      THEN 'Best Case'
             ELSE 'Commit' END)                                                 AS forecast_category,
    s.forecast_category_raw IS NULL                                             AS forecast_category_imputed,
    s.amount_local,
    o.currency,
    round(NULLIF(s.amount_local, 0) * fx.rate_to_eur, 2)                        AS amount_eur,
    s.close_date,
    to_char(s.close_date, 'YYYY') || '-Q' || to_char(s.close_date, 'Q')         AS close_quarter,
    s.close_date BETWEEN s.quarter_start AND s.quarter_end                      AS closes_this_quarter,
    s.close_date < s.snapshot_week
        AND s.stage_name NOT IN ('Closed Won', 'Closed Lost')                   AS is_past_due,
    s.snapshot_week - o.created_date                                            AS deal_age_days,
    s.snapshot_week - s.stage_entered_at::date                                  AS days_in_stage,
    COALESCE(p.close_date_pushes, 0)                                            AS close_date_pushes
FROM state s
JOIN staging.stg_opportunities o USING (opportunity_id)
JOIN raw.fx_rates fx ON fx.currency = o.currency
LEFT JOIN pushes p   ON p.snapshot_week = s.snapshot_week AND p.opportunity_id = s.opportunity_id;

ALTER TABLE marts.fct_pipeline_snapshot_weekly ADD PRIMARY KEY (snapshot_week, opportunity_id);
CREATE INDEX ON marts.fct_pipeline_snapshot_weekly (owner_id, snapshot_week);
ANALYZE staging.stg_accounts, staging.stg_opportunities, staging.stg_opportunity_history,
        marts.dim_week, marts.fct_pipeline_snapshot_weekly;

-- ---------------------------------------------------------------------
-- Quick look: open pipeline closing this quarter, latest 6 weeks
-- ---------------------------------------------------------------------
SELECT snapshot_week, fiscal_quarter, week_of_quarter,
       count(*) FILTER (WHERE is_open)                                         AS open_deals,
       count(*) FILTER (WHERE is_open AND closes_this_quarter)                 AS open_deals_this_qtr,
       round(sum(amount_eur) FILTER (WHERE is_open AND closes_this_quarter))   AS open_pipeline_this_qtr_eur,
       count(*) FILTER (WHERE is_past_due)                                     AS past_due_deals
FROM marts.fct_pipeline_snapshot_weekly
GROUP BY 1, 2, 3
ORDER BY 1 DESC
LIMIT 6;
