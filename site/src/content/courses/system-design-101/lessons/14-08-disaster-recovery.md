---
id: "14.08"
course: "system-design-101"
module: "14"
slug: "disaster-recovery"
title: "Disaster recovery"
keyIdeas: ["**RPO** (recovery point objective): how much data you can lose, measured in time. **RTO** (recovery time objective): how long recovery may take.", "Strategies by cost and speed: backup and restore (hours, cheapest), pilot light (core data replicated, infra started on demand), warm standby (scaled-down copy running), multi-site active-active (minutes or less, most expensive).", "RTO of minutes needs warm standby or active-active.", "DR plans fail if never exercised. Run failover drills.", "Watch for hidden single-region dependencies (DNS, auth, config stores)."]
question: "Define RPO and RTO. Which DR strategy gives RTO of minutes?"
answer: "RPO (recovery point objective) is the most data the business can afford to lose, measured as time between the last copy that reached the recovery location and the disaster. RTO (recovery time objective) is the longest it can afford to be down, from the disaster (not from when someone notices) until service is restored. For an RTO of minutes, use warm standby: a scaled-down but fully working copy in another region with live data replication, which can serve traffic immediately after the database is promoted and traffic switched, and only needs scaling up. Active/active gets close to zero but costs much more and needs multi-region writes. Pilot light, which must start servers first, is typically tens of minutes, and backup and restore is hours."
sources: [{"title": "AWS: Disaster Recovery of Workloads on AWS", "url": "https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/disaster-recovery-workloads-on-aws.html", "note": "Backup/restore to multi-site active-active."}]
---

## What counts as a disaster

Lesson 14.07 dealt with losing data inside a running system. Disaster recovery (DR) deals with losing the **whole location** a workload runs in, or losing trust in it: a cloud region outage, a fire or flood at a datacentre, a fibre cut that isolates a region, a large operator mistake (deleting a production account), or ransomware that encrypts everything reachable. An event is a disaster when the workload can no longer meet its business goals in its main location, and the only way back is to run somewhere else or rebuild.

Separate DR from **high availability**. Availability is about frequent small failures (a host dies, a bug ships, a traffic spike) and is handled inside the region with redundancy, load balancing and automatic restarts. It is measured as an average over time, like 99.95% per month. DR is about rare, large events and is measured **per event** by two numbers the business must set.

## Two numbers: RPO and RTO

- **RPO (recovery point objective)**: the most data you can afford to lose, measured in time. An RPO of 5 minutes means at most the last 5 minutes of writes may be lost. It is set by how often data reaches the recovery location: nightly backups give an RPO of up to 24 hours; asynchronous replication with 1 s lag gives about 1 s; synchronous replication gives zero.
- **RTO (recovery time objective)**: the longest you can be down, from the moment of the disaster until service is restored. It is set by how much must be done before the recovery location can serve: rebuilding everything takes hours; flipping DNS to a running copy takes minutes.

~~~dg-timeline RPO looks backward from the disaster to the last recoverable copy; RTO looks forward to restored service.
13:55 | Last write that reached region B
*14:00 | Region A fails: writes 13:55 to 14:00 lost (RPO 5 min)
14:03 | Health checks confirm, failure detected
14:05 | Failover decision made
14:12 | Region B scaled up, DNS switched
14:15 | Service restored (RTO 15 min)
~~~

Two consequences are easy to miss. First, **detection is part of the RTO**. The clock starts when the region fails, not when someone notices, so a 15-minute RTO cannot include a 10-minute wait for a human to wake up. Second, **RPO is about the data that was acknowledged to users**. If you told a customer "order placed" and then lost the order, they will be far angrier than if the order had failed. With asynchronous replication you are accepting that some acknowledged writes can be lost.

How do you pick the numbers? Ask the business what an hour of downtime and an hour of lost data cost (revenue, contractual penalties, regulatory exposure, reputation). Then compare with what each strategy costs per month. A payments system might need RPO zero and RTO 5 minutes; an internal reporting tool might accept RPO 24 hours and RTO 2 days.

## The four strategies

Lower RPO and RTO cost more, because more of the system must already be running and up to date in the recovery region.

~~~dg-matrix The four DR strategies: each step up keeps more running in the recovery region, buying lower RPO and RTO for more money.
Strategy | Data in region B | Servers in region B | Typical RPO | Typical RTO | Cost
Backup and restore | backups only | none | hours | hours to a day | low
Pilot light | live replica | off or not deployed | seconds to minutes | tens of minutes | low to medium
Warm standby | live replica | small fleet running | seconds | minutes | medium
Active/active | live, all regions write | full fleet serving | near zero | near zero | high
~~~

### Backup and restore

Only backups exist in the recovery region (copied there on a schedule). After a disaster you create everything: networks, databases from backups, servers, load balancers. This depends entirely on **infrastructure as code** (Terraform, CloudFormation and similar): if the environment can only be rebuilt from someone's knowledge and a wiki page, it cannot be rebuilt in a crisis. RPO is the backup interval; RTO is the time to provision everything plus the time to restore the data, often several hours for a large database. Cheapest by far, because you pay only for storage.

### Pilot light

The core data is replicated live into the recovery region (a database replica, replicated object storage), but application servers are switched off or not deployed. Like a gas boiler's pilot flame, the smallest piece is always lit so the rest can be ignited quickly. On failover you promote the database replica, deploy or start the servers, scale them up and switch traffic. RPO is the replication lag (seconds). RTO is tens of minutes, mostly provisioning and scaling.

### Warm standby

A smaller but **fully working** copy of the whole system runs in the recovery region all the time: database replica, a few application servers, load balancers, everything wired up. It can serve traffic **immediately** at reduced capacity. On failover you promote the database, switch traffic and scale the fleet up. Because it is always running, it is also always testable: you can send a trickle of synthetic or real traffic to it every day. Warm standby is the cheapest option that reliably gives an RTO of minutes.

~~~dg-compare Pilot light versus warm standby: both have live data in region B; only warm standby can serve the moment you switch.
[Pilot light]
DB replica live -(failover)-> deploy servers -> scale up -> switch DNS
+ cheapest way to keep data current
- cannot serve until servers start
- untested paths: AMIs, quotas, config
- RTO tens of minutes
[*Warm standby]
DB replica + small fleet -(failover)-> switch DNS -> scale up
+ serves at once at reduced capacity
+ can take test traffic every day
- pay for an idle small fleet
- RTO minutes
~~~

### Multi-site active/active

The full system runs in two or more regions, and all of them serve traffic. A region failure is no longer a separate "failover" event: health-checked routing just stops sending users to the failed region, and the others absorb the load (so each must have spare capacity, typically running at no more than about 50% to 66% of what it can handle). RPO and RTO approach zero.

The cost is not just doubled infrastructure. The hard part is **writes in several regions**. Either each record has a single home region that accepts its writes (users partitioned by region, with failover of ownership), or every region accepts writes and you must resolve conflicts (last-writer-wins, CRDTs, or application merge rules; lesson 06.03 covers replication). Global strongly consistent databases (Spanner, CockroachDB) solve this with consensus across regions, at the cost of cross-region latency on every write.

### A concrete data layer

Managed databases make the lower rungs easy. Aurora Global Database, for example, replicates to secondary regions with typical lag under one second and can promote a secondary region to read/write in under a minute. That gives an RPO of about a second and a database RTO of a minute. The rest of the RTO comes from everything around the database: application servers, caches, queues, secrets, DNS.

<div class="ex">
<div class="lab">Worked example</div>

**picking a strategy**

An online store says it can lose at most 1 minute of orders (RPO 1 min) and be down at most 15 minutes (RTO 15 min). It earns about \$20,000 an hour.

Backup and restore is out: restoring a 2 TB database alone takes hours. Pilot light might meet 15 minutes if starting servers is fully automated, but first boot, cache warm-up and quota surprises make it tight.

Warm standby fits: database replicated with second-level lag, 4 application servers already running in region B (against 40 in region A), failover by promoting the replica, switching DNS (with a 60 s TTL) and autoscaling to 40. Extra cost: perhaps \$3,000 a month for the standby fleet and replica.

Add hourly snapshots and versioned object storage in case of corruption, and a quarterly game day that measures the real failover time.

</div>

## Replication does not save you from corruption

The point from 14.07 applies here with extra force. Continuous replication copies **destructive** changes too: deletes, corruption, ransomware encryption, a bad migration. Warm standby and active/active protect against losing a region, not against losing the data's integrity. So every strategy still needs **point-in-time backups or object versioning**, ideally in a separate account with separate credentials, so you can restore to a moment before the damage.

## Failover mechanics and the split-brain risk

Failing over a database means **promoting** a replica to be the new primary. Three hazards come with it:

- **Lost writes.** With asynchronous replication, writes that reached the old primary but not the replica are gone (that is your RPO). Some may surface later when the old region returns, and must be reconciled, not silently merged.
- **Split brain.** If the old primary is not dead but only cut off, both regions may accept writes. Prevent it with **fencing**: revoke the old primary's ability to write (shut it down, remove its credentials, or require a lease or epoch number that the old one no longer holds).
- **Clients still pointing at the old region.** DNS caches, connection pools and hard-coded endpoints keep traffic flowing to the failed side. Use low DNS TTLs, health-checked routing and clients that re-resolve.

~~~dg-seq A warm-standby failover: detect, fence the old primary, promote, switch traffic, scale.
Monitor > Monitor : deep health checks fail in region A for 3 min
Monitor > Operator : page: region A down, approve failover?
Operator > Region A DB : fence: revoke write access
Operator > Region B DB : promote replica to primary
Region B DB -> Operator : read/write, last 2 s of writes lost
Operator > DNS : point api.example.com to region B
Operator > Region B fleet : scale from 4 to 40 servers
Region B fleet -> Operator : serving full traffic
~~~

Should failover be automatic? For tight RTOs, yes, triggered by **deep health checks** that test real user actions (log in, load a product, place a test order) from several vantage points, not shallow pings. But automated failover on a false alarm is an outage you caused yourself, and flapping between regions is worse. Many teams automate everything except the final decision, which a human approves with one click.

## Testing: the only recovery that works is one you practise

Paths you rarely run do not work when you need them. Assumptions go stale quietly: the DR region's database instance is too small to take the full load, service quotas there were never raised, a config change or a new secret was only added in the main region, the machine images are months old, or the runbook refers to a tool that was retired. The only error recovery that works reliably is the path you test frequently.

So run **game days** that actually fail over (in production, during working hours, with everyone ready), measure the RTO and RPO you really achieve, and fix what broke. Some companies go further and run active/active precisely so that "failover" is exercised every day by normal traffic shifting.

<div class="ex">
<div class="lab">Worked example</div>

**a game day that found three problems**

The team simulates the loss of region A at 10:00. Target RTO is 15 minutes.

Failover is approved at 10:04. Replica promotion takes 50 s. Then autoscaling stalls at 25 instances: the instance quota in region B was never raised. A support request takes 40 minutes.

Meanwhile 30% of traffic still goes to region A: a mobile app version pins a regional hostname.

Payments fail in region B because the payment provider's API key was rotated in region A's secret store only.

Measured RTO: 52 minutes. Fixes: raise and monitor quotas, remove regional hostnames, replicate secrets automatically. The next game day achieves 11 minutes.

</div>

## Common mistakes

- **No RPO and RTO agreed with the business**, so nobody knows whether the plan is good enough.
- **Treating replicas as the backup**: ransomware or a bad delete reaches region B in a second.
- **Never testing failover**, then discovering quotas, secrets and stale images during the real event.
- **Forgetting dependencies**: DNS, identity provider, secrets, third-party APIs, CI/CD and monitoring must also work from the recovery region.
- **No fencing**, so a partitioned old primary keeps accepting writes and you end up with two diverged databases.

## In an interview

Ask for RPO and RTO early, or state them ("I'll assume we can lose a few seconds of data and be down for at most 15 minutes"). Then pick the cheapest strategy that meets them, name it, and say what replicates and how failover happens. Mention point-in-time backups alongside replication, and regular failover tests.

<div class="take">
<div class="lab">Key takeaway</div>

RPO is how much data you can lose; RTO is how long you can be down, including detection. Cost rises as you keep more running elsewhere: backup and restore (hours), pilot light (tens of minutes), warm standby (minutes), active/active (near zero). Replication copies corruption, so keep point-in-time backups too, fence the old primary on failover, and test failover regularly.

</div>
