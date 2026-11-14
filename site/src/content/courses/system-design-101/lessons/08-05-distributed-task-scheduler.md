---
id: "08.05"
course: "system-design-101"
module: "08"
slug: "distributed-task-scheduler"
title: "Distributed task scheduler"
keyIdeas: ["A distributed scheduler runs jobs at set times (cron) or after delays, across many machines, without missing or double-running them.", "Store job definitions and next run times durably. A leader (or partitioned set of schedulers) finds due jobs and dispatches them to workers via a queue.", "Use leader election or row-level claiming so two schedulers don't fire the same job.", "Exactly-once is hard: aim for at-least-once dispatch plus idempotent jobs, and record each run with a unique run ID.", "Google's distributed cron uses Paxos to agree on which runs have been launched."]
question: "How do you ensure a scheduled job runs exactly once when the scheduler has 3 replicas?"
answer: "Elect one leader among the replicas (via Paxos, Raft or a renewable lease in etcd, ZooKeeper or the database, with a fencing token), or shard tasks so each has one owner, so only one replica launches a given task. Give each run a deterministic ID made of the task and its scheduled time, and insert it into an executions table with a unique constraint before launching, so a second scheduler's insert fails. A new leader checks which IDs already exist and launches only missing ones. Delivery to workers is still at least once, so the job itself is idempotent (outputs keyed by run ID, idempotency keys on external calls), which makes a rare duplicate harmless."
sources: [{"title": "Google SRE book: Distributed Periodic Scheduling with Cron", "url": "https://sre.google/sre-book/distributed-periodic-scheduling/", "note": "Reliable cron at Google using Paxos."}, {"title": "Hello Interview: Job scheduler", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler", "note": "Interview breakdown."}]
---

## Cron, but reliable

Every product has work that runs on a clock: send invoices at 02:00, expire trial accounts every hour, rebuild a sitemap nightly, charge subscriptions on the 1st of the month. The classic tool is **cron**: a daemon on one machine reads a **crontab** (a file of schedules like `0 2 * * *`, meaning minute 0, hour 2, every day) and launches commands at those times.

Plain cron has three problems once the work matters:

- **Single point of failure.** If that machine is down at 02:00, nothing runs. Nobody is told.
- **No memory.** Standard cron doesn't record what it launched or whether it succeeded. Only variants such as anacron re-run jobs that were missed while the machine was off.
- **No scale.** One machine can launch a few hundred jobs a minute. A platform where every customer schedules their own reports may need thousands a second.

The obvious fix, "run cron on three machines", creates a worse problem: now each job runs three times. A **distributed task scheduler** is the design that runs scheduled jobs on several machines for availability, while making each scheduled run happen **once**.

## Jobs fail in different ways

Before designing, ask what each job can tolerate. They fall into three groups:

- Some can safely be **skipped**. A garbage-collection run every 10 minutes: if one is missed, the next one catches up.
- Some must **not be skipped**. Monthly payroll or a daily settlement file: a missed run is an incident.
- Some must **never run twice**. Sending a newsletter to 2 million people, or charging every subscriber.

Many jobs are both "don't skip" and "don't double". When you can't guarantee both, you must choose which failure is worse. Google's internal distributed cron chose to **prefer skipping a run over launching it twice**, because a missed run is usually easy to notice and fix (run it by hand), while undoing a double launch, such as a second payroll or a second email blast, may be impossible.

~~~dg-matrix What each failure costs for typical jobs. This decides how strict the scheduler and the job must be.
Job | Skipped run OK? | Double run OK? | Late run OK?
Cache cleanup | yes | yes | yes
Sitemap rebuild | partial, next run fixes | yes | yes
Invoice generation | no | no | partial, within hours
Newsletter send | partial | no | partial
Payroll | no | no | no
~~~

## Vocabulary

- A **task** (or job definition) is a kind of work with a schedule: "generate invoices, daily at 02:00 UTC".
- A **run** (or execution, or job instance) is one occurrence: "generate invoices for 2024-05-01T02:00".
- **Scheduled time** is when the run was meant to start. **Actual time** is when it did.
- **At-least-once** means every run is guaranteed to start, maybe more than once. **At-most-once** means it starts at most once, maybe never. Real systems give at-least-once delivery and make the job idempotent to get an effect that looks like exactly once.

The key idea of this whole lesson: **identify each run by task plus scheduled time**. `invoices@2024-05-01T02:00` is a name that every replica can compute independently, that is the same no matter which replica launches it, and that can be checked for existence. Everything else builds on that.

## How Google's distributed cron works

Google's design is a good reference because it handles the hard cases explicitly.

- Cron runs as several **replicas** (typically 3 or 5) spread across failure domains, placed by the datacenter scheduler (Borg), so one machine or rack failure doesn't stop it.
- The replicas use **Paxos**, a consensus algorithm, to agree on shared state. Only the **leader** launches jobs. Followers replicate the job list and track which launches are open, so any of them can take over.
- **Before** launching, the leader records "launching job X for scheduled time T" in the Paxos log. **After** launching, it records completion. If it loses leadership, it must immediately stop talking to the scheduler, so two leaders never launch at once.
- On failover, the new leader must finish all **open launches** (started but not marked complete) quickly, within about a minute. It can't know whether an in-flight launch RPC reached Borg before the old leader died. So either launches must be idempotent, or their state must be checkable.
- The answer is to give each launch a **precomputed name that includes the scheduled time**, such as `payroll-2024-05-01T00:00`. The new leader asks Borg whether a job with that name exists and launches only missing ones. Including the time also stops it relaunching a short job that already finished and disappeared from the running list.
- State is kept as Paxos logs plus periodic **snapshots** on the replicas' local disks, with snapshots backed up. A restarted replica loads the snapshot and replays only the log after it.
- To avoid every team's job firing at exactly midnight (the **thundering herd**), crontab accepts `?` in a field, and the system hashes the job's configuration to pick a stable value, spreading launches across the hour.

~~~dg-seq Google's distributed cron: record the launch in Paxos before acting, record completion after, and check by name on failover.
Leader > Paxos log : launching payroll@...T00:00
Leader > Borg : start job named payroll@...T00:00
Leader > Leader : crash before recording done
Follower > Paxos log : elected, read open launches
Follower > Borg : does payroll@...T00:00 exist?
Borg -> Follower : yes, so skip
Follower > Paxos log : record completion
~~~

## The general recipe

Now the version you'd draw in an interview: a scheduler for many tenants, framed at about **10,000 runs a second at peak**.

### Components

1. **Schedule store**: a database table of tasks, with the cron expression, the task type, parameters, and a computed `next_run_at`. Indexed on `next_run_at`.
2. **Scheduler**: a service that repeatedly finds tasks due soon (`WHERE next_run_at <= now() + 30 s`), creates a run for each, pushes it to a queue (possibly a **delayed queue** that releases it at the exact time), and advances `next_run_at` to the following occurrence.
3. **Executions table**: one row per run, keyed by `(task_id, scheduled_time)` with a **unique constraint**. Holds status, attempts, worker, timestamps, result.
4. **Queue and workers**: workers pull runs, claim them with a lease or visibility timeout, execute and record the result.

~~~dg-flow The scheduler finds due tasks, creates uniquely keyed runs, and hands them to workers through a queue.
[Plan] Schedule store -(due in 30 s)-> *Scheduler leader -(insert run)-> Executions table
[Execute] Scheduler leader -(enqueue run id)-> Queue -(lease)-> Workers -(result)-> Executions table
~~~

### Making the scheduler highly available but single-acting

You need the scheduler on several machines, but only one deciding at a time for any given task. Two ways:

- **Leader election.** The replicas hold an election through etcd, ZooKeeper or Consul (all built on consensus, Raft or ZAB), or with a lease row in your database. The holder of the lease is the leader and renews it every few seconds; if it stops renewing, another replica takes over after the lease expires (say 10 seconds). Use a **fencing token** (a number that increases with every new lease) so a paused old leader that wakes up can be recognised and rejected.
- **Sharding by task ID.** Split tasks into, say, 64 shards; each scheduler replica owns some shards through a lease per shard. This scales beyond one leader's capacity, which you need at 10,000 runs a second.

Leader election alone is not enough. Leases expire on time, but a process can be paused (a long garbage collection, a VM migration) and not know it lost the lease. For a moment, two schedulers may both think they lead. The **unique constraint on (task, scheduled_time)** is what turns "rarely two leaders" into "never two runs": the second insert fails.

~~~dg-state One scheduled run's lifecycle. The unique key is created at Planned, so a duplicate scheduler can never create a second row.
[*] -> Planned
Planned -> Queued : enqueued at due time
Queued -> Running : worker takes lease
Running -> Succeeded : effect committed
Running -> Queued : lease expired
Running -> Failed : error
Failed -> Queued : retry with backoff
Failed -> *Dead : attempts exhausted, page
Planned -> Skipped : too late, skip policy
~~~

### Making the job safe

Delivery is still at least once. A worker can finish the work, then crash before marking the run succeeded; the lease expires, and another worker runs it again. So the job's **effect** must be idempotent:

- Write outputs under a name that includes the run key (`invoices/2024-05-01.csv`), so a re-run overwrites rather than duplicates.
- Pass the run key as an idempotency key to external APIs. Payment providers such as Stripe deduplicate requests with the same key.
- For per-item side effects (one email per customer), record each item done in a table keyed by `(run_key, customer_id)` before or atomically with sending, and skip items already recorded.

### Missed runs and catch-up

If the whole system is down from 01:30 to 03:30, what happens to the 02:00 and 03:00 runs? You need a **misfire policy** per task: run all missed ones in order (payroll), run only the latest one (sitemap rebuild), or skip them (cache cleanup). Quartz, a widely used Java scheduler, calls this the misfire instruction. Also decide **concurrency**: if the 02:00 run is still going at 03:00, does the 03:00 run start, wait or get skipped?

### Spreading load

If 50,000 customers all schedule "daily at midnight", the scheduler and every downstream system takes 50,000 runs in one second. Add **jitter** (hash the task ID into a 0 to 15 minute offset) for tasks that don't need precise timing, as Google's `?` field does.

<div class="ex">
<div class="lab">Worked example</div>

**three replicas, one run**

The 02:00 invoice task has three scheduler replicas. Replica A holds a 10-second lease in etcd with fencing token 41. At 01:59:30 it inserts `invoices@2024-05-01T02:00` into the executions table and enqueues it with a 30-second delay. Then it crashes.

Replica B's lease attempt succeeds at 01:59:42 with token 42. It scans tasks due soon, sees invoices, and tries to insert the same run key. The unique constraint rejects it, so B knows the run already exists and does nothing more. At 02:00 the queue releases the run; a worker takes it, and before sending each invoice it inserts `(run_key, customer_id)` into a sent-invoices table, skipping customers already present. Even if the queue delivers the message twice, each customer gets one invoice.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**sizing the scheduler**

A platform has 20 million customer-defined tasks. Most run daily; a few run every minute. Average load is about 20,000,000 / 86,400 ≈ 230 runs a second, but customers cluster at round hours, so peak is 40 times that, about 10,000 runs a second. Each run needs one executions insert and one enqueue. One Postgres primary handles a few thousand such inserts a second comfortably, so the design shards the executions table and schedulers by task ID into 16 shards, about 600 runs a second each at peak, and adds 0 to 5 minutes of jitter for tasks marked "time-flexible", which cuts the peak by half.

</div>

## Common mistakes

- Running cron on every replica of a service "for availability", so every job runs N times.
- Relying on leader election alone without a unique run key; pauses and network partitions eventually produce two leaders.
- Naming runs with the actual start time instead of the scheduled time, so a retry looks like a new run.
- No misfire policy, so a two-hour outage either loses runs silently or replays 120 minute-level runs at once.
- No alert when a run that must happen hasn't succeeded by its deadline. "Didn't run" is silent unless you check for it.

## In an interview

State the goal precisely: "exactly-once effect, built from at-least-once delivery plus a deterministic run ID plus an idempotent job". Then draw the four parts (schedule store, elected or sharded scheduler, executions table with a unique key, queue and workers), and mention misfire policy and jitter. Ask which jobs must never be skipped and which must never be doubled.

<div class="take">
<div class="lab">Key takeaway</div>

make one leader (or one shard owner) responsible for launching, name each run by task and scheduled time, and record that name durably with a unique constraint so a new leader can tell what already happened. Make the job idempotent for the duplicates you can't rule out.

</div>
