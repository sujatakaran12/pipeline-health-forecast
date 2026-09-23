# Q3 outlook and how we can forecast with more confidence

**To:** VP Sales
**From:** Sujata Karan, Sales Analytics
**Date:** 21 September 2026

## The short answer

We are likely to land close to target this quarter, not above it. As of week 12 we have closed EUR 7.1M against a target of EUR 8.2M. The team's combined call is EUR 8.5M. My forecast model puts us at about EUR 8.1M, so we should plan for "on target, possibly just short" rather than "comfortably over".

## What the data shows

1. **We are starting quarters with too little pipeline.** At the start of Q2 2025 we had 5.4 times our target in open pipeline, and we beat target (125%). Since then targets have more than doubled (EUR 3.3M to 8.2M) while pipeline has stayed at EUR 14 to 19M. Coverage has fallen to 2.3 to 3.5 times, and every quarter since has finished between 79% and 97% of target.

2. **Deals that will be won move fast. Deals that will be lost get stuck.** Won deals spend about a week in each stage. Lost deals spend two to nine times longer, up to 75 days in Negotiation. Deals whose close date is pushed twice win 25% of the time; after three or more pushes, only 7%. Of the deals we expect to close at the start of a quarter, only about one in five actually closes that quarter.

3. **Our current forecasts are off in predictable ways.** Rep calls run 20 to 24% too high in weeks 8 to 12 of the quarter. The standard stage-probability method in Salesforce overstates bookings by up to 64%, because our real win rates are much lower than its defaults. A simple model trained on our own history (stage, time in stage, pushes, deal age, plus the deals we typically create mid-quarter) was within 10% of actual bookings in every week of the last four quarters.

4. **Individual reps are consistent in how they miss.** Eight reps regularly call less than 70% of what they close. Six call more than 1.5 times what they close. Because the pattern is stable, we can adjust for it.

## What I recommend

1. **Set a pipeline coverage goal of 3 times target by the first week of each quarter**, and track it weekly from mid-way through the previous quarter. Much of the fix sits upstream: more pipeline from product-led accounts, which win most often.
2. **Add two flags to the weekly pipeline review:** any deal more than three weeks in its current stage, and any deal pushed twice or more. Either qualify these deals back in with a clear next step or move them out of this quarter's forecast.
3. **Use the model forecast next to rep calls in the weekly call**, and review each rep's historical call accuracy with them. The aim is not to replace their judgement but to show where it is usually too high or too low.

## Caveats

This analysis uses synthetic data built to mirror a product-led plus enterprise SaaS business, and the forecast was tested on four quarters. On real data I would expect the model's advantage to be smaller, and I would validate it over two more quarters before relying on it.
