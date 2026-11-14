---
id: "08.07"
course: "system-design-101"
module: "08"
slug: "dag-workflow-orchestration"
title: "DAG workflow orchestration"
keyIdeas: ["Workflow orchestration runs multi-step processes where steps depend on each other, as a DAG (directed acyclic graph).", "The orchestrator tracks state of each step, retries failed steps, runs independent steps in parallel, and resumes after crashes.", "**Airflow**: scheduled batch DAGs, mostly data pipelines.", "**Temporal / Cadence / Step Functions**: durable execution for application workflows. Code looks sequential, but the engine persists every step so it survives restarts.", "If step 4 of 5 fails: retry with backoff; if it keeps failing, either alert or run compensating actions for steps 1-3 (saga pattern)."]
question: "A 5-step pipeline fails at step 4. What should the orchestrator do?"
answer: "Retry step 4 with exponential backoff as its retry policy sets, without rerunning steps 1 to 3, whose results are recorded durably. If the error is non-retryable or retries run out, mark the downstream step 5 as not run (upstream_failed), alert someone, and keep the run resumable so that after a fix it continues from step 4. If earlier steps had external side effects that must not stand on their own (a charge, a reservation), run compensating actions in reverse order, as in a saga. Every step should be idempotent so retries after a partial success are safe."
sources: [{"title": "Apache Airflow: Core concepts", "url": "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html", "note": "DAGs, tasks, dependencies."}, {"title": "Temporal: What is Temporal?", "url": "https://docs.temporal.io/evaluate/understanding-temporal", "note": "Durable workflow execution."}, {"title": "Hello Interview: Temporal deep dive", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/temporal", "note": "Durable workflows in interviews."}]
---

## Why one queue isn't enough

A single background job (lesson 08.01) is one step: take message, do work, ack. Real workflows have many steps with dependencies. A nightly revenue pipeline must extract orders, extract refunds, join them, load the result into a warehouse and email finance. An order fulfilment flow must charge the card, reserve stock, book a courier and send a confirmation, and if the courier booking fails after the card was charged, it must refund.

You could chain queues: step 1's worker enqueues step 2 when it finishes, and so on. That works for two or three steps. Beyond that you find yourself building the same machinery every time: remember which steps finished, pass outputs between steps, retry the failed one without rerunning the others, time out stuck steps, run independent steps in parallel, let a human see where a run is stuck, and resume after a fix. That machinery is a **workflow orchestrator**.

Two families dominate:

- **DAG schedulers** such as Apache Airflow, Dagster and Prefect, used mostly for data pipelines that run on a schedule.
- **Durable execution engines** such as Temporal, AWS Step Functions and Azure Durable Functions, used mostly for long-running business processes.

## Pipelines as graphs

A workflow's steps and dependencies form a **directed acyclic graph** (DAG): directed because each dependency points one way (refunds must be extracted before the join), acyclic because there are no loops (a step can't depend, even indirectly, on itself). A DAG lets the orchestrator compute what can run now: any step whose upstream steps have all finished.

~~~dg-tree The revenue pipeline as a dependency graph: the two extracts run in parallel, and everything downstream waits for both.
Email finance
  Load warehouse
    Join and aggregate
      *Extract orders
      *Extract refunds
~~~

### Airflow's model

Airflow names each piece:

- A **DAG** holds everything needed to run the workflow: the tasks, their dependencies, the schedule, and defaults such as retries and timeouts.
- **Tasks** are units of work run by workers (a SQL query, a Python function, a Spark job). Dependencies are declared with `>>`: `extract_orders >> join` means join runs after extract_orders.
- A DAG runs on a **schedule** (`@daily`, a cron string) or on demand. Each execution is a **DAG run** with a **data interval**, the period of data it should process. The 2024-05-01 run processes orders from 00:00 to 24:00 on 1 May, even if it actually runs at 01:00 on 2 May.
- Because each run is tied to an interval, Airflow can **backfill**: run the DAG once for each past interval, for example to rebuild a year of daily aggregates after fixing a bug.
- Each task in a run is a **task instance** with a state. Terminal states include `success`, `failed`, `upstream_failed` (a dependency failed, so this never ran) and `skipped`.
- **Retries** are configured per task (`retries=3, retry_delay=5 minutes`, optionally exponential).
- **Trigger rules** decide whether a task runs given its upstream results: `all_success` (the default), `all_failed`, `one_failed`, `all_done` (run whatever happened, for cleanup), `none_failed` and more. They let you add cleanup or alert tasks to the graph.

~~~dg-state One task instance's lifecycle in a DAG run. A failed upstream task stops downstream tasks without running them.
[*] -> Scheduled
Scheduled -> Queued : dependencies met
Queued -> Running : worker picks up
Running -> Success : exit 0
Running -> Up for retry : error, retries left
Up for retry -> Queued : retry delay passed
Running -> *Failed : retries exhausted
Scheduled -> Upstream failed : a dependency failed
~~~

### How an Airflow-style scheduler works underneath

The scheduler is a loop over a metadata database (usually Postgres). Every few seconds it parses DAG definitions, creates DAG runs whose time has come, and for each run looks for task instances whose upstream tasks satisfy the trigger rule. It marks them `queued` and sends them to an executor (Celery workers, Kubernetes pods). Workers run the task and write the final state back to the database. All the state lives in the metadata database, so a scheduler restart loses nothing: it just picks up from the recorded states. That's the key property: **the orchestrator records each step's outcome durably**, so it never needs to rerun a step that already succeeded.

## Durable execution

Temporal takes a different approach, aimed at business processes that may wait for hours or months: "charge the card, wait up to 7 days for the customer to confirm, then ship".

- **Workflows** are written as ordinary code in Go, Java, Python or TypeScript, with loops, conditionals and local variables.
- **Activities** are the steps that touch the outside world: call an API, write a database, charge a card, send an email. Each has a retry policy, timeouts and optional heartbeats, set by configuration.
- The Temporal server records every step in an **event history**: "workflow started", "activity ChargeCard scheduled", "activity ChargeCard completed with result r_77", "timer 7 days started", and so on.
- If the worker running a workflow crashes, another worker loads the history and **replays** the workflow code from the start. When the code reaches an activity that the history says already completed, the SDK returns the recorded result instead of running it again. The code reaches exactly the point where it stopped, with all local variables rebuilt, and continues.
- A `sleep(7 days)` is a durable timer stored on the server, not a sleeping thread. No worker is occupied while waiting.

Because of replay, **workflow code must be deterministic**: given the same history, it must make the same decisions in the same order. No direct clock reads, random numbers, network calls or reading files in workflow code. Those go in activities (whose results are recorded) or in SDK functions such as `workflow.now()` that record their values in the history.

~~~dg-seq Temporal: completed activities are in the event history, so a new worker replays them instead of re-running them.
Worker A > History : ChargeCard completed (r_77)
Worker A > History : ReserveStock completed
Worker A > Worker A : crash during BookCourier
Worker B > History : load event history
Worker B > Worker B : replay: reuse r_77, skip calls
Worker B > History : BookCourier scheduled again
Worker B > History : BookCourier completed
~~~

### When to reach for it

Reach for Temporal (or Step Functions) when you notice you're building a state machine by hand to persist completed steps, schedule the next step, retry failures and recover after crashes, especially when processes wait a long time between steps. Its limits: it's overkill for a single background job (use a queue), it isn't a replacement for Kafka for high-volume event streams, and its activities still need to be idempotent because they can be retried: an activity can complete its side effect, then the worker can crash before reporting completion.

~~~dg-compare Two orchestration styles. Both record step results durably so a failure resumes at the failed step.
[DAG scheduler]
Schedule -(daily)-> DAG run -(task states in DB)-> Retry failed task
+ Great for batch data pipelines
+ Backfills by data interval
- Steps fixed in the graph
- Not for waiting days on humans
[Durable execution]
Start -(code + history)-> Activities -(replay on crash)-> Continue
+ Ordinary code, loops, branches
+ Durable timers, waits of months
- Workflow code must be deterministic
- Another cluster to operate
~~~

## When step 4 fails

Now the core question. A five-step pipeline fails at step 4. What should a good orchestrator do?

1. **Retry step 4** with exponential backoff, a bounded number of times. Most failures are transient: a lock timeout, a network blip, a rate limit. Steps 1 to 3 are not rerun; their outputs are recorded (as files, table partitions or recorded activity results).
2. **Classify the error.** Some errors are not worth retrying: bad input, a missing permission, a schema mismatch. Mark these non-retryable so you fail fast instead of burning 30 minutes of retries.
3. If it still fails, **stop downstream steps**. Airflow marks step 5 `upstream_failed`; it is never run. **Alert** someone, ideally through an alert task triggered by `one_failed`, so the alert is part of the pipeline.
4. **Keep the run resumable.** After a fix, rerun from step 4. In Airflow you "clear" the failed task instance, and the scheduler runs it and then its downstream tasks. In Temporal you can reset a workflow to a point in its history.
5. **Compensate if needed.** If earlier steps changed external state that must not stand alone (the card was charged, but shipping can never happen), run **compensating actions** in reverse order: release the stock, refund the card. This is the **saga** pattern: a long transaction built from local steps, each with an undo.
6. **Make every step idempotent**, so a retry after a partial success doesn't double-apply. Writing a warehouse partition with "replace partition for 2024-05-01" is idempotent; "append rows" is not.

~~~dg-timeline One night of the revenue pipeline: step 4 fails, retries, and the downstream step runs only once it succeeds.
01:00 | Extract orders and extract refunds start in parallel
01:12 | Both extracts succeed
01:30 | Join succeeds
01:31 | *Load warehouse fails: lock timeout
01:36 | Retry 1 fails: lock still held
01:41 | Retry 2 succeeds
01:42 | Email finance runs under all_success
~~~

<div class="ex">
<div class="lab">Worked example</div>

**nightly revenue pipeline**

Steps: extract orders, extract refunds, join, load warehouse, email finance. The load fails because a long-running analyst query holds a lock on the target table. Airflow retries it twice, 5 minutes apart, and the second retry succeeds; the email task, waiting under `all_success`, then runs.

Had it kept failing, the email task would be `upstream_failed` and never send a partial report. An alert task with trigger rule `one_failed` pages on-call. The engineer finds the stuck query, kills it, and clears just the failed task instance. The scheduler reruns the load (which replaces the 1 May partition, so it's safe even if a previous attempt wrote half the rows) and then sends the email. Steps 1 to 3, which took 30 minutes, are not repeated.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**an order saga with compensation**

An order workflow in Temporal runs: ChargeCard (succeeds, payment p_33), ReserveStock (succeeds), BookCourier, SendConfirmation. BookCourier fails 5 times over 20 minutes because the courier API rejects the postcode: a non-retryable business error. The workflow's error handler runs compensations in reverse: ReleaseStock, then RefundPayment(p_33) with idempotency key `refund-order-9001`. The refund call times out once and is retried; the payment provider sees the same key and refunds only once. The customer gets a "we couldn't deliver to that address" email.

</div>

## Designing good steps

- **Small and focused.** One step does one thing, so a retry repeats as little as possible.
- **Outputs to durable storage**, not in memory, so a later step (or a rerun) can read them.
- **Idempotent writes**: overwrite a partition, upsert by key, use idempotency keys for external calls.
- **Timeouts on everything.** A step with no timeout can hang forever and block the run.
- **Pass references, not data.** Pass "s3://bucket/2024-05-01/orders.parquet", not 2 GB of rows through the orchestrator's database.

## Common mistakes

- Restarting the whole pipeline after a failure, repeating hours of work and possibly double-writing.
- Retrying non-retryable errors, delaying the alert by the whole retry budget.
- Non-idempotent steps (appending rows, sending emails without dedup), so a retry corrupts data.
- Putting I/O or `random()` in Temporal workflow code, which breaks replay.
- Using Airflow to process events one at a time with second-level latency. It's a batch scheduler; use a queue or stream.

## In an interview

Say: "The orchestrator stores each step's state, so on failure it retries only step 4 with backoff; if that keeps failing it stops downstream steps, alerts, and leaves the run resumable from step 4; steps with side effects have compensations, and every step is idempotent." Then pick a tool by workload: Airflow-style for scheduled data pipelines, Temporal-style for long-running business processes.

<div class="take">
<div class="lab">Key takeaway</div>

an orchestrator records each step's state durably, so a failure is retried or resumed at that step, not from the beginning. Retry transient errors with backoff, stop and alert on persistent ones, compensate side effects with a saga, and make every step idempotent.

</div>
