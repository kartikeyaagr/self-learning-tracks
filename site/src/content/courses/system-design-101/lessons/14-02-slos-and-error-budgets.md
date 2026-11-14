---
id: "14.02"
course: "system-design-101"
module: "14"
slug: "slos-and-error-budgets"
title: "SLOs and error budgets"
keyIdeas: ["An **SLO** is a target for a measured service level (SLI), like \"99.9% of requests succeed in under 300 ms over 30 days\".", "The **error budget** is what the SLO allows to fail: 0.1% of requests. It turns reliability into a shared number for product and engineering.", "Error budget policy: while budget remains, ship freely; when it's exhausted, freeze risky launches and fix reliability.", "**Burn-rate alerts**: page when the budget is being consumed fast enough to run out (e.g. 2% of the monthly budget in 1 hour), rather than on short blips.", "Pick SLIs users feel: availability, latency, freshness, correctness."]
question: "Your 99.9% SLO budget is half spent in week 1. What does the error budget policy say to do?"
answer: "Half the budget gone in week one means an average burn rate of 2: at that pace the budget runs out around the end of week two. You are not yet out, so the policy does not freeze everything, but reliability rises in priority now: do the incident's postmortem actions first, slow or postpone risky changes, and watch the slow-burn ticket alert. If the budget is exhausted, the policy freezes all changes except fixes and security patches and puts engineering time on reliability until the rolling window moves past the outage. Product, development and SRE agreed these actions in advance, so nobody has to argue about them now."
sources: [{"title": "Google SRE workbook: Implementing SLOs", "url": "https://sre.google/workbook/implementing-slos/", "note": "From SLIs to error budget policy."}, {"title": "Google SRE workbook: Alerting on SLOs", "url": "https://sre.google/workbook/alerting-on-slos/", "note": "Burn-rate alerts."}]
---

## Why reliability needs a number

"Make it more reliable" is not a goal anyone can finish. Engineers always want more stability, product teams always want more features, and without a shared number the argument is settled by whoever shouts loudest after the last outage. Service level objectives turn reliability into a measured quantity with a target, and the error budget turns the gap between 100% and that target into something both sides can spend.

The key insight is that **100% is the wrong target**. Users cannot tell the difference between 99.99% and 100% when their own Wi-Fi, phone and ISP fail far more often than that. Every extra nine costs roughly ten times more (more replicas, more regions, slower releases), and the last one is practically infinite. So you pick a level that keeps users happy, and treat everything above it as room to move fast.

Three terms stack on top of each other:

~~~dg-layers The vocabulary stack: you measure an SLI, aim for an SLO, and promise an SLA with a safety margin below it.
SLA | contract with customers; missing it costs money or credits
*SLO | internal target for an SLI over a window, e.g. 99.9% in 28 days
SLI | the measurement: good events divided by total events
Telemetry | raw logs, metrics and probes the SLI is computed from
~~~

The **SLA** (service level agreement) is an external contract: "if monthly availability is below 99.9%, you get 10% credit". The SLO is internal and should be stricter than any SLA, so you notice trouble and react before it costs money. Many internal services have SLOs and no SLA at all.

## SLIs: measuring what users feel

A **service level indicator** (SLI) is best expressed as a ratio: **good events divided by total events**, giving a value from 0% to 100%. Keeping every SLI in this one shape means the same tools can chart, alert and report on any of them. Examples:

- successful HTTP requests ÷ all HTTP requests (availability);
- gRPC calls that succeed in under 100 ms ÷ all calls (latency);
- search results served from the full index ÷ all results, including degraded ones (quality);
- pipeline runs that finished within 1 hour of their data arriving ÷ all runs (freshness).

Separate the **specification** (what users care about: "the home page loads in under 100 ms") from the **implementation** (how you measure it). The same specification can be measured in several places, each with a blind spot:

- **Server logs or metrics**: cheap and complete for requests that arrive, but blind to requests that never reached you (DNS failure, load balancer down).
- **Load balancer metrics**: catch backend crashes, but still miss failures before the load balancer.
- **Synthetic probers**: catch network and DNS failures, but only test the paths you script.
- **Client-side telemetry** (JavaScript or app instrumentation): closest to what users feel, but noisy (users' own bad networks) and the most work.

Start with the cheapest measurement that is good enough, then improve it when you find it misses real pain.

Which SLIs to pick depends on what the component does:

| Component | Typical SLIs |
| --- | --- |
| Request-driven (APIs, web) | availability, latency, quality |
| Pipeline (batch, streaming) | freshness, correctness, coverage |
| Storage | durability, availability, latency |

Keep it to about three to five SLIs per service. More than that and nobody knows which one matters.

## SLOs and the error budget

An SLO is a target for an SLI over a window: "99.9% of requests succeed, measured over 28 days". The **error budget** is what is left: 100% − 99.9% = 0.1% of requests are allowed to fail in that window. For a time-based view, 0.1% of 28 days is about 40 minutes of full outage.

| SLO | Budget per 28 days | Budget per year |
| --- | --- | --- |
| 99% | 6.7 hours | 3.65 days |
| 99.9% | 40 minutes | 8.8 hours |
| 99.95% | 20 minutes | 4.4 hours |
| 99.99% | 4 minutes | 53 minutes |

Notice what 99.99% means operationally: a 4-minute budget per month is less than the time it takes a human to wake up, log in and understand a page. A four-nines service needs automatic failover and automatic rollback; people alone cannot meet it.

### Setting the first SLO from data

A good way to start is to measure current performance and set the SLO just below it. Consider a mobile game API. Over four weeks it served 3,663,253 requests, of which 97.123% succeeded, with p90 latency of 432 ms and p99 of 891 ms. The team set SLOs just below current performance: 97% availability, 90% of requests under 450 ms, 99% under 900 ms.

<div class="ex">
<div class="lab">Worked example</div>

**pricing incidents in budget**

The availability budget is 3% of 3,663,253, about **109,897 failed requests** per four weeks.

A bad release that caused 14,066 errors cost 14,066 ÷ 109,897 ≈ **13%** of the budget.

A database failure with a 20-hour restore caused about 72,000 errors: about **65%**.

Bad releases happen two or three times a year; the database failure happened once in several years. Over a year, releases cost 3 × 13% ≈ 39% of a month's budget, the restore 65% once in many years. Per year, the releases are the bigger cost and the cheaper fix. So the team first invests in safer releases (canaries, automatic rollback), then in faster restores.

</div>

That is the real power of an error budget: it lets you rank reliability work by how much budget each class of failure actually burns.

**Windows.** A **rolling** window (say the last 28 days) matches how users remember outages: a bad Tuesday still counts for four weeks. Use a whole number of weeks so every window holds the same number of weekends (traffic patterns differ). **Calendar** windows (this month, this quarter) match business planning and SLA contracts, but reset abruptly: an outage on the 30th and one on the 2nd land in different months. A common default is a four-week rolling window for operational decisions, with weekly summaries for prioritising and quarterly reports for planning.

## The error budget policy

An SLO only matters if missing it changes behaviour. The **error budget policy** is a short written agreement, signed off in advance by product, development and SRE (site reliability engineering), saying what happens as the budget is spent. Typical rules:

- While budget remains, teams ship freely. A healthy service should spend some of its budget; never touching it means you could take more risk and move faster.
- When budget is being spent quickly, reliability work from recent incidents moves to the top of the backlog and risky launches slow down.
- When the budget is exhausted, **freeze production changes** except fixes and security patches, until enough budget returns as the window rolls forward.
- Work only on reliability until the service is back within its SLO, with leadership backing to turn down feature requests.
- In extreme cases (budget exceeded several times over), escalate to senior leadership.

~~~dg-state The error budget policy as a state machine: spending changes what the team is allowed to do.
[*] -> Healthy
Healthy -> At risk : burn rate above 1 for days
At risk -> Healthy : burn slows, postmortem fixes land
At risk -> *Exhausted : budget reaches 0
Healthy -> Exhausted : one big outage
Exhausted -> Frozen : policy freezes non-fix releases
Frozen -> Healthy : window rolls past the outage
~~~

The policy also needs an **escalation path** for disagreements and a rule for outages caused by dependencies (a cloud provider failure still burns your budget from the user's point of view; freezing anyway is friendlier to users, though some teams exclude it).

All three groups must agree to it, because each gives something up. Product agrees the threshold is good enough for users, developers agree to slow down when it is missed, and SRE agrees it can be met without heroics. If one party will not enforce the policy, change the SLO until they will. An unenforced SLO is just a dashboard.

## Alerting on the budget: burn rate

Alerting whenever the error rate crosses 0.1% is too noisy. A 0.1% error rate for 10 minutes uses only 0.02% of a 30-day budget, yet such an alert could fire up to 144 times a day while the SLO is still comfortably met. The better measure is **burn rate**: how fast you are spending the budget relative to the even pace. A burn rate of 1 uses exactly the whole budget by the end of the window. For a 99.9% SLO over 30 days:

~~~dg-bars How long a 30-day budget for a 99.9% SLO lasts at different burn rates (error rate = burn rate × 0.1%).
Burn 1 (0.1% errors) | 720 | 30 days
Burn 2 (0.2%) | 360 | 15 days
Burn 6 (0.6%) | 120 | 5 days
Burn 14.4 (1.44%) | 50 | 50 hours
*Burn 1,000 (100%) | 0.72 | 43 minutes
~~~

The formula is: budget consumed = burn rate × alert window ÷ SLO period. Recommended starting thresholds:

| Severity | Long window | Short window | Burn rate | Budget used |
| --- | --- | --- | --- | --- |
| Page | 1 hour | 5 minutes | 14.4 | 2% |
| Page | 6 hours | 30 minutes | 6 | 5% |
| Ticket | 3 days | 6 hours | 1 | 10% |

Check the first row: 14.4 × 1 h ÷ 720 h = 2%. The intuition: "if this continues, we lose the whole month's budget in about two days; wake someone up."

The **short window** (one twelfth of the long one) is the "multiwindow" part. An alert fires only when both windows exceed the threshold. The long window shows the burn is significant; the short window shows it is still happening. When errors stop, the short window drops below the threshold within minutes and the alert clears, instead of firing for another hour while the long window slowly forgets.

~~~dg-timeline A 10-minute outage at 20% errors against a 99.9% SLO: the 1-hour window detects it, the 5-minute window lets it clear quickly.
10:00 | Deploy goes out, error rate jumps to 20% (burn 200)
10:01 | 5-minute burn above 14.4
10:04 | *1-hour burn crosses 14.4: both windows high, page fires
10:06 | On-call rolls back
10:10 | Errors stop; about 4.6% of the month's budget is spent
10:15 | 5-minute burn back below 14.4, alert clears
11:10 | 1-hour window finally forgets the outage
~~~

**Low-traffic services** break this maths. At 10 requests an hour, a single failure is a 10% error rate and a burn rate of 100. Options: add synthetic traffic so there is enough signal, group small services that share failure domains into one SLO, add client retries so one failure is not a user-visible failure, or deliberately set a lower SLO.

<div class="ex">
<div class="lab">Worked example</div>

**half the budget gone in week one**

A 99.9% SLO over 4 weeks with 3 million requests has a 3,000-error budget. One outage in week 1 causes 1,500 errors: 50% spent in 25% of the window, so the average burn rate so far is 2.

The burn-rate alert paged during the outage (1-hour burn far above 14.4). Now the policy applies. You are not out of budget, so no freeze yet. But at a burn of 2 you would run out around day 14.

So the team prioritises the outage's postmortem actions, slows risky launches (more canary time, no big migrations), and watches the 3-day ticket alert. If a second incident uses the rest, the freeze starts and lasts until the rolling window moves past the first outage, roughly three weeks later.

</div>

## Common mistakes

- **SLO set at 100% or at the SLA.** You have no budget to ship with and no warning before you owe credits.
- **SLIs that users do not feel.** CPU utilisation is not an SLI; "checkout succeeded in under 1 s" is.
- **Averages as SLIs.** "Mean latency under 200 ms" can be met while 5% of users wait 5 s. Use "fraction of requests under the threshold".
- **No signed policy.** The budget runs out, product wants the launch anyway, and nothing happens. Then the SLO means nothing.
- **Alerting on raw thresholds.** Paging on every 1% spike creates fatigue; use burn rate with two windows.

## In an interview

When you state non-functional requirements, phrase availability as an SLO ("99.9% of redirects succeed in under 100 ms over 28 days") and translate it into a budget ("about 40 minutes a month"). That tells the interviewer whether you need multi-region failover (four nines) or just good deploy hygiene (three nines). If asked about operations, mention burn-rate alerting and a freeze policy.

<div class="take">
<div class="lab">Key takeaway</div>

define SLIs as good ÷ total events users feel, set an SLO slightly below what you achieve, and treat the remainder as a budget to spend. A signed error budget policy decides in advance what happens as it runs out, and burn-rate alerts with a long and a short window tell you when it is running out too fast.

</div>
