-- =====================================================================
-- Sale_Analytics | Step 2: pipeline health metrics
-- Needs 10_pipeline_snapshot.sql to have run first. Safe to re-run.
--
--   marts.pipeline_coverage_weekly   Q1: do we have enough pipeline?
--   marts.stage_conversion           Q2: where do deals drop out?
--   marts.stage_velocity             Q2: how long does each stage take?
--   marts.quarter_start_slippage     Q3: what happens to deals due this quarter?
--   marts.win_rate_by_pushes         Q3: what do close-date pushes do to win rates?
-- =====================================================================
\set ON_ERROR_STOP on
SET client_min_messages = warning;

-- ---------------------------------------------------------------------
-- Q1. Coverage: open pipeline due this quarter vs quota still to close
--     coverage_ratio = open pipeline closing this quarter / (quota - won so far)
--     Company level, every Monday. Rule of thumb: 3x at quarter start.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS marts.pipeline_coverage_weekly;
CREATE TABLE marts.pipeline_coverage_weekly AS
WITH quota AS (
    SELECT fiscal_quarter, sum(quota_eur) AS quota_eur
    FROM raw.quotas GROUP BY 1
),
wk AS (
    SELECT
        snapshot_week, fiscal_quarter, week_of_quarter,
        sum(amount_eur) FILTER (WHERE is_open AND closes_this_quarter)                          AS open_pipeline_eur,
        sum(amount_eur) FILTER (WHERE is_open AND closes_this_quarter AND forecast_category = 'Commit')    AS commit_eur,
        sum(amount_eur) FILTER (WHERE is_open AND closes_this_quarter AND forecast_category = 'Best Case') AS best_case_eur,
        sum(amount_eur) FILTER (WHERE is_closed_won AND closes_this_quarter)                   AS won_qtd_eur,
        count(*)        FILTER (WHERE is_past_due)                                             AS past_due_deals
    FROM marts.fct_pipeline_snapshot_weekly
    GROUP BY 1, 2, 3
),
final AS (   -- what the quarter actually booked (latest snapshot of each quarter + anything closed after it)
    SELECT to_char(close_date, 'YYYY') || '-Q' || to_char(close_date, 'Q') AS fiscal_quarter,
           sum(amount_eur) AS actual_bookings_eur
    FROM staging.stg_opportunities WHERE is_won GROUP BY 1
)
SELECT
    wk.*,
    q.quota_eur,
    greatest(q.quota_eur - coalesce(wk.won_qtd_eur, 0), 0)                                     AS remaining_quota_eur,
    round(wk.open_pipeline_eur / nullif(q.quota_eur - coalesce(wk.won_qtd_eur, 0), 0), 2)       AS coverage_ratio,
    f.actual_bookings_eur,
    round(f.actual_bookings_eur / q.quota_eur, 3)                                              AS quarter_attainment,
    wk.snapshot_week >= '2026-07-01'                                                           AS quarter_in_progress
FROM wk
LEFT JOIN quota q  USING (fiscal_quarter)
LEFT JOIN final f  USING (fiscal_quarter);

-- ---------------------------------------------------------------------
-- Stage path per deal: highest stage reached (a skipped stage counts as
-- passed). Uses deals created after the Salesforce migration, so the
-- full stage history is available, and only deals that have closed.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS marts.deal_stage_path;
CREATE TABLE marts.deal_stage_path AS
SELECT
    o.opportunity_id, o.opportunity_type, o.segment, o.lead_source, o.is_won,
    max(left(h.new_value, 1)::int) FILTER (WHERE h.new_value ~ '^[1-5] ')      AS max_open_stage_reached
FROM staging.stg_opportunities o
JOIN staging.stg_opportunity_history h ON h.opportunity_id = o.opportunity_id AND h.field = 'StageName'
WHERE o.created_date >= '2025-01-15' AND o.is_closed
GROUP BY 1, 2, 3, 4, 5;

-- ---------------------------------------------------------------------
-- Q2a. Stage conversion (New Business): of deals that reached stage N,
--      share that reached the next stage, and share eventually won.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS marts.stage_conversion;
CREATE TABLE marts.stage_conversion AS
WITH s AS (SELECT generate_series(1, 5) AS stage_number),
seg AS (
    SELECT segment FROM (VALUES ('All'), ('Commercial'), ('Mid-Market'), ('Enterprise')) v(segment)
)
SELECT
    seg.segment,
    s.stage_number,
    count(*) FILTER (WHERE p.is_won OR p.max_open_stage_reached >= s.stage_number)                          AS deals_reached,
    count(*) FILTER (WHERE p.is_won OR p.max_open_stage_reached >= s.stage_number + 1)                      AS deals_reached_next,
    round(count(*) FILTER (WHERE p.is_won OR p.max_open_stage_reached >= s.stage_number + 1)::numeric
        / nullif(count(*) FILTER (WHERE p.is_won OR p.max_open_stage_reached >= s.stage_number), 0), 3)     AS conversion_to_next,
    round(count(*) FILTER (WHERE p.is_won)::numeric
        / nullif(count(*) FILTER (WHERE p.is_won OR p.max_open_stage_reached >= s.stage_number), 0), 3)     AS win_rate_from_stage
FROM s CROSS JOIN seg
JOIN marts.deal_stage_path p
  ON p.opportunity_type = 'New Business' AND (seg.segment = 'All' OR p.segment = seg.segment)
GROUP BY 1, 2
ORDER BY 1, 2;
-- note: for stage 5, "reached next" = won

-- ---------------------------------------------------------------------
-- Q2b. Velocity: days spent in each stage (New Business, post-migration
--      deals, completed stage visits only), split won vs lost.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS marts.stage_velocity;
CREATE TABLE marts.stage_velocity AS
SELECT
    o.segment,
    left(h.new_value, 1)::int                                                    AS stage_number,
    CASE WHEN o.is_won THEN 'Won' ELSE 'Lost' END                                AS outcome,
    count(*)                                                                     AS stage_visits,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY h.valid_to::date - h.valid_from::date)  AS median_days,
    round(avg(h.valid_to::date - h.valid_from::date), 1)                         AS avg_days
FROM staging.stg_opportunities o
JOIN staging.stg_opportunity_history h ON h.opportunity_id = o.opportunity_id AND h.field = 'StageName'
WHERE o.opportunity_type = 'New Business' AND o.is_closed AND o.created_date >= '2025-01-15'
  AND h.new_value ~ '^[1-5] ' AND h.valid_to < 'infinity'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

-- ---------------------------------------------------------------------
-- Q3a. Slippage: deals open in week 1 of a quarter with a close date in
--      that quarter. What actually happened to them by quarter end?
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS marts.quarter_start_slippage;
CREATE TABLE marts.quarter_start_slippage AS
WITH start_book AS (
    SELECT s.fiscal_quarter, s.opportunity_id, s.amount_eur, s.forecast_category
    FROM marts.fct_pipeline_snapshot_weekly s
    WHERE s.week_of_quarter = 1 AND s.is_open AND s.closes_this_quarter
),
qtr AS (
    SELECT DISTINCT fiscal_quarter, quarter_end FROM marts.dim_week
),
outcome AS (
    SELECT b.*,
        CASE
            WHEN o.is_won  AND o.close_date <= q.quarter_end THEN 'Won in quarter'
            WHEN o.is_closed AND NOT o.is_won AND o.close_date <= q.quarter_end THEN 'Lost in quarter'
            ELSE 'Slipped to later quarter'
        END AS outcome
    FROM start_book b
    JOIN staging.stg_opportunities o USING (opportunity_id)
    JOIN qtr q USING (fiscal_quarter)
)
SELECT fiscal_quarter, forecast_category, outcome,
       count(*)                  AS deals,
       round(sum(amount_eur))    AS amount_eur
FROM outcome
WHERE fiscal_quarter < '2026-Q3'          -- only completed quarters
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

-- ---------------------------------------------------------------------
-- Q3b. Win rate by number of close-date pushes before the deal closed
--      (New Business, post-migration, closed deals).
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS marts.win_rate_by_pushes;
CREATE TABLE marts.win_rate_by_pushes AS
WITH closed_at AS (
    SELECT opportunity_id, min(valid_from) AS closed_at
    FROM staging.stg_opportunity_history
    WHERE field = 'StageName' AND new_value IN ('Closed Won', 'Closed Lost')
    GROUP BY 1
),
pushes AS (
    SELECT o.opportunity_id, o.is_won, o.segment,
           count(h.history_id) AS pushes
    FROM staging.stg_opportunities o
    JOIN closed_at c USING (opportunity_id)
    LEFT JOIN staging.stg_opportunity_history h
      ON h.opportunity_id = o.opportunity_id AND h.field = 'CloseDate'
     AND h.old_value IS NOT NULL AND h.new_value::date > h.old_value::date
     AND h.valid_from < c.closed_at                 -- ignore the date change made when the deal is closed
    WHERE o.opportunity_type = 'New Business' AND o.created_date >= '2025-01-15'
    GROUP BY 1, 2, 3
)
SELECT
    CASE WHEN pushes >= 3 THEN '3+' ELSE pushes::text END   AS close_date_pushes,
    count(*)                                               AS closed_deals,
    round(avg(is_won::int), 3)                             AS win_rate
FROM pushes
GROUP BY 1
ORDER BY 1;

-- =====================================================================
-- Headline results
-- =====================================================================
\echo '\n== Q1. Coverage in week 1 of each quarter vs what the quarter booked =='
SELECT fiscal_quarter, snapshot_week, coverage_ratio,
       round(open_pipeline_eur) AS open_pipeline_eur, round(quota_eur) AS quota_eur,
       round(actual_bookings_eur) AS actual_bookings_eur, quarter_attainment, quarter_in_progress
FROM marts.pipeline_coverage_weekly WHERE week_of_quarter = 1 ORDER BY fiscal_quarter;

\echo '\n== Q2a. New Business stage conversion (all segments) =='
SELECT stage_number, deals_reached, conversion_to_next, win_rate_from_stage
FROM marts.stage_conversion WHERE segment = 'All' ORDER BY stage_number;

\echo '\n== Q2b. Median days in stage, won vs lost (all segments) =='
SELECT h.stage_number,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY (h.valid_to::date - h.valid_from::date)) FILTER (WHERE o.is_won)     AS won_median_days,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY (h.valid_to::date - h.valid_from::date)) FILTER (WHERE NOT o.is_won) AS lost_median_days
FROM staging.stg_opportunities o
JOIN (SELECT *, left(new_value, 1)::int AS stage_number FROM staging.stg_opportunity_history
      WHERE field = 'StageName' AND new_value ~ '^[1-5] ' AND valid_to < 'infinity') h USING (opportunity_id)
WHERE o.opportunity_type = 'New Business' AND o.is_closed AND o.created_date >= '2025-01-15'
GROUP BY 1 ORDER BY 1;

\echo '\n== Q3a. Deals due this quarter at quarter start: what happened (all completed quarters) =='
SELECT outcome, sum(deals) AS deals, round(100.0 * sum(deals) / sum(sum(deals)) OVER (), 1) AS pct_deals,
       round(100.0 * sum(amount_eur) / sum(sum(amount_eur)) OVER (), 1) AS pct_value
FROM marts.quarter_start_slippage GROUP BY 1 ORDER BY 1;

\echo '\n== Q3b. Win rate by close-date pushes =='
SELECT * FROM marts.win_rate_by_pushes;
