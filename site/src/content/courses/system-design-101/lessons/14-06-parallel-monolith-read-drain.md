---
id: "14.06"
course: "system-design-101"
module: "14"
slug: "parallel-monolith-read-drain"
title: "Parallel monolith read drain"
keyIdeas: ["When replacing an old read path (monolith query, old database) with a new one, prove they agree before switching.", "**Shadow reads**: for each request, serve from the old path, also call the new path in the background, and compare and log differences.", "GitHub's Scientist library does this: run \"control\" and \"candidate\" code, return the control result, and report mismatches and timings.", "Fix mismatches until the diff rate is near zero, then shift reads gradually (1% → 10% → 100%), keeping the old path as fallback.", "Finally drain and remove the old path."]
question: "How do you prove a new read path returns the same results before cutting over?"
answer: "Run both paths on live traffic with a tool like GitHub Scientist. The old path is the control and its result is always returned to the user. The new path runs as the candidate, its exceptions are caught and its result and timing are compared and published. First run control against control to learn the background noise, then start at a small percentage, investigate each mismatch and either fix it or add an explicit ignore or compare rule, and ramp up. Cut over only after a long period, covering rare jobs and big customers, with no unexplained mismatches. Keep dual writes so you can switch back, and never shadow code with side effects."
sources: [{"title": "GitHub Scientist", "url": "https://github.com/github/scientist", "note": "Run old and new code paths side by side and compare."}, {"title": "Stripe: Online migrations at scale", "url": "https://stripe.com/blog/online-migrations", "note": "Shadow reads before switching."}]
---

## The problem: proving a new read path is right

In a data migration (lesson 14.05) or a rewrite, there comes a moment when you switch reads from the old path to the new one: from the monolith to a new service, from the old table to a new store, from a hand-written SQL query to a search index. Tests cover the cases you thought of. Production has the cases you did not: odd rows from 2014, a customer with 40,000 subscriptions, rounding in a currency with three decimal places, a timezone that changed its rules. A new path can pass every test and still return different answers for 0.3% of real requests, and you will not know until customers complain.

The reliable way to be sure is to run both paths **in parallel on real traffic**, compare their answers, and cut over only once they agree. This is called a **shadow read**, and when you use it to gradually move read traffic off an old system (typically a monolith) you are doing a **read drain**: draining reads from the old path while proving at each step that the new one matches it.

It fits a larger pattern for replacing monoliths, the **strangler fig**: a new system grows around the old one, taking over one capability at a time, until the old one can be removed. Each capability moves in the same order: build the new path, dual write so it has the data, shadow read to prove it, switch reads, switch writes, retire the old code.

~~~dg-tree A strangler-fig migration: each capability of the monolith is drained separately, reads before writes.
Monolith
  Orders capability
    Dual write to orders service | done
    *Shadow reads at 25% | in progress
    Switch reads | waiting
    Switch writes | waiting
  Search capability | fully moved to search service
  Billing capability | not started
~~~

## GitHub Scientist

Scientist is GitHub's Ruby library for exactly this job (ports exist for many other languages). You wrap the old code as the **control** (`use`) and the new code as the **candidate** (`try`):

```ruby
def allows?(user)
  science "widget-permissions" do |e|
    e.context user: user
    e.use { model.check_user(user).valid? }  # control: old path
    e.try { user.can?(:read, model) }        # candidate: new path
  end
end
```

Each time the experiment runs, Scientist:

1. decides whether to run the candidate at all (`enabled?`, for example 1% of calls);
2. runs both blocks **in random order**, so neither always benefits from warm caches or suffers from cold ones;
3. measures the time each one takes;
4. **rescues any exception from the candidate**, so a bug in the new code cannot break the request;
5. compares the two results;
6. **always returns the control's result** to the caller;
7. passes everything (both values, both timings, any mismatch, the context) to your `publish` method, which sends counters and timings to metrics and stores mismatches for review.

~~~dg-seq One request inside an experiment: both paths run, the user gets the old answer, the comparison is published.
Caller > Scientist : allows?(user 42)
Scientist > Scientist : enabled? yes (1% sample)
Scientist > New path : candidate (runs first this time)
New path -> Scientist : false, 38 ms
Scientist > Old path : control
Old path -> Scientist : true, 112 ms
Scientist > Publisher : mismatch, timings, context user 42
Scientist -> Caller : true (control result)
~~~

The control result is always returned, so users are never exposed to the new path's answers. The cost is extra load: the candidate runs as well, adding its latency to the request (Scientist runs both blocks in the same thread by default) and its queries to the new store.

## In-process versus traffic mirroring

Scientist compares inside the application. Another approach is to mirror at the network layer: a proxy or service mesh copies a share of requests to the new service and discards its responses.

~~~dg-compare Two ways to shadow reads: compare inside the application, or copy traffic at the proxy and compare offline.
[*In-process (Scientist)]
request -> app -(both paths)-> compare -> publish
+ compares the actual returned values
+ full context: user, inputs, both results
- candidate latency added to the request
- needs code changes in the old system
[Proxy mirroring]
request -> proxy -(copy)-> new service -> discard
+ no change to the old code
+ new service can be tested at full load
- responses must be logged and diffed offline
- only safe for side-effect-free requests
~~~

In-process comparison is better for correctness work because you compare the exact values. Mirroring is better for load testing the new service, and it is the only option when you cannot change the old system.

## Running the experiment

1. **Start small.** `enabled?` turns on the candidate for a small percentage of calls, which limits extra load and the risk from a slow candidate. Scientist does **not** protect you from a candidate that hangs, so put a timeout inside the candidate block and keep the percentage low at first.
2. **Calibrate.** First run the control against itself. Random ordering plus data changing between the two reads (a row updated in the few milliseconds between them) creates false mismatches, so you learn the background noise rate before blaming the new code.
3. **Investigate mismatches.** Each one is either a bug in the new path, a bug in the old path that you must now either reproduce or consciously fix, or noise from timing and ordering.
4. **Teach it about known differences.** `compare` sets what "equal" means, such as comparing only IDs, ignoring list order or rounding floats. `ignore` marks mismatches you understand (for example, the new store returns a slightly newer timestamp). `clean` trims what gets stored for review, so you do not log sensitive data.
5. **Ramp up** until the mismatch rate is zero or fully explained, over a long enough period to cover rare paths: monthly billing jobs, end-of-year reports, the largest customers.
6. **Cut over.** Make the new path the real one. Keep dual writes in place for a while, so the old path stays current and you can switch back if a problem appears.
7. **Retire.** After a quiet period, delete the old read path. Make any remaining old accessor raise an error first, to flush out callers you missed.

~~~dg-state A read drain as a sequence of states; any mismatch sends you back to fix before ramping further.
[*] -> Calibrating
Calibrating -> Shadow 1% : control vs control noise known
Shadow 1% -> Fixing : unexplained mismatches
Fixing -> Shadow 1% : fix or ignore rule shipped
Shadow 1% -> Shadow 100% : clean for days
Shadow 100% -> *Cut over : zero unexplained for weeks
Cut over -> Shadow 100% : problem, switch back
Cut over -> Retired : old path deleted
~~~

### The big caveat: reads only

Scientist is only safe around code that **does not change data**. Running a write twice, once in each path, would apply it twice: two charges, two emails, two rows. For writes, use dual writing (14.05) and compare the stored results afterwards with a reconciliation job, or make the candidate compute what it *would* write and compare that, without writing.

Even "reads" can hide side effects: a read that updates a `last_seen` column, warms a cache, increments a rate-limit counter or triggers lazy migration. Check that the candidate path is genuinely side-effect free before wrapping it.

### Comparing at the right level

Decide what "the same" means before you start. Comparing whole JSON documents byte for byte produces endless false mismatches from field order, float formatting and timestamps. Normalise both sides first: sort lists that have no defined order, round floats, drop fields that are expected to differ. But do not normalise away real differences: if the old path returns orders sorted by date and the client depends on that order, the order is part of the contract and must match.

<div class="ex">
<div class="lab">Worked example</div>

**draining order reads out of a monolith**

A monolith answers `GET /orders?user=…` from its own tables; a new orders service should take over. A calibration run (control against control) shows 0.05% background mismatches from orders updated between the two reads.

Week 1: the candidate runs on 1% of 8,000 requests per second, about 80 per second, and 3% of those mismatch. Investigation finds two causes: the new service sorts ties on the same timestamp differently (fixed by a `compare` that sorts by ID first) and it misses cancelled orders older than 2019 (a real backfill bug: the copy job filtered on a status that did not exist then; fixed and re-run).

Weeks 2 and 3: ramp to 10%, then 100%. Unexplained mismatches stay at 0 over 14 days, including a month-end. The new path's p99 is 40 ms against 120 ms for the old.

Cut over reads, keep dual writes for a month, then make the monolith's order query raise an error in staging, find one forgotten admin report still calling it, move it, and delete the old path.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**the cost of shadowing**

At 100% shadowing, the new service handles the full 8,000 requests per second as extra load, and each request now does both queries. If the old path takes 120 ms and the new 40 ms and they run one after the other, request latency rises to about 160 ms. So you ramp: at 10%, average added latency is 4 ms and the new service sees 800 requests per second, which also doubles as a load test of the new service before it carries real traffic.

</div>

## How Stripe used it

In Stripe's subscriptions migration (14.05), Scientist experiments read subscriptions from both the old inline field and the new table on live traffic. Any mismatch raised an alert. Errors in the new path did not affect users. Reads moved only once the results matched. During the hardest step, switching writes, more experiments checked each small change. Finally, the old accessor was made to raise an error, flushing out any caller that still used it.

## Common mistakes

- **Shadowing writes.** Double charges and duplicate emails. Shadow only side-effect-free reads.
- **Comparing too strictly or too loosely.** Byte equality drowns you in noise; over-normalising hides real bugs.
- **No timeout on the candidate.** A hanging new path slows every sampled request.
- **Cutting over too early.** Two clean days miss the monthly job and the largest customer.
- **Removing dual writes at cut-over.** You lose the ability to switch back cheaply.
- **Logging sensitive data in mismatch records** without cleaning it.

## In an interview

When your design replaces a component, the interviewer may ask "how would you migrate?" Describe the strangler approach: dual write, backfill, shadow reads with result comparison and a mismatch dashboard, ramp, cut over with dual writes still on, then retire. Say explicitly that shadowing is for reads and that writes are verified by reconciliation.

<div class="take">
<div class="lab">Key takeaway</div>

before switching reads, run the new path in the shadow of the old one on real traffic. Return the old result, catch the new path's errors, compare results and timings, calibrate against background noise, and ramp up until mismatches are zero or explained. Only shadow side-effect-free reads, and keep dual writes after cutting over so you can switch back.

</div>
