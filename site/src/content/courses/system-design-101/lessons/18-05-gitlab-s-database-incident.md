---
id: "18.05"
course: "system-design-101"
module: "18"
slug: "gitlab-s-database-incident"
title: "GitLab's database incident"
keyIdeas: ["In January 2017 an engineer, working on replication problems, accidentally deleted the production database directory on the primary GitLab.com database server.", "About 6 hours of data (issues, merge requests, comments) was lost.", "Of five backup and replication mechanisms, none worked properly: pg_dump backups were failing silently because of a version mismatch, with failure emails rejected; disk snapshots weren't enabled for the DB servers; replication to the secondary was broken; S3 backups were empty.", "They restored from a staging copy made 6 hours earlier by luck.", "Lesson: regularly test restores and monitor that backups succeed. GitLab live-streamed the recovery and published a detailed postmortem."]
question: "List the backup failures GitLab found. Which single practice would have caught them?"
answer: "The failures: pg_dump backups to S3 were empty because the job ran pg_dump 9.2 against PostgreSQL 9.6; the cron failure emails were silently rejected because DMARC was not configured, so nobody knew; Azure disk snapshots were enabled for the NFS servers but not the database servers; replication copied the problem and the secondary had been wiped during the rebuild anyway, with no WAL archive to fall back on; and LVM snapshots existed only for refreshing staging, so the best restore point was a 6-hour-old manual one on throttled disks that took about 18 hours to copy, losing about 5,000 projects, 5,000 comments and 700 accounts. The single practice that would have caught all of them is regularly restoring backups, automated and verified against production, with an alert that fires when no successful restore has been reported recently rather than only when a failure is reported."
sources: [{"title": "GitLab: Postmortem of database outage of January 31", "url": "https://about.gitlab.com/blog/2017/02/10/postmortem-of-database-outage-of-january-31/", "note": "5 backup methods, none working."}]
tags: ["Case"]
tools: ["GitLab", "PostgreSQL"]
---

## Why this incident is studied

On 31 January 2017, GitLab.com, a service hosting the code of hundreds of thousands of developers, lost about six hours of production database data. The proximate cause was a tired engineer deleting a directory on the wrong server. That happens. What made it famous is what came next: the team went looking for a backup and found that **five separate backup and replication mechanisms** were either broken, missing or never meant for recovery. GitLab published an unusually honest public postmortem and even live-streamed the recovery, which is why the details are known.

This lesson walks through the timeline, each failed safety net and why it failed, and then answers the question: which single practice would have caught all of them?

## Background

GitLab.com ran on PostgreSQL 9.6 with one primary database server (`db1`) and one secondary (`db2`) kept up to date by **streaming replication**: the primary sends its write-ahead log (WAL) to the secondary, which replays it (lessons 03.08 and 06.03). The secondary existed for failover. Backups were meant to come from a nightly `pg_dump` uploaded to Amazon S3, plus various disk snapshots.

A key detail about streaming replication: the primary keeps only a limited amount of WAL on disk. If the secondary falls too far behind, the WAL it still needs is recycled, and the secondary can never catch up by streaming. It must be rebuilt from a fresh copy of the primary (`pg_basebackup`), unless WAL is also **archived** somewhere the secondary can fetch it from. GitLab was not archiving WAL.

## What happened

~~~dg-timeline The evening of 31 January 2017 (UTC) and the recovery that followed.
17:20 | LVM snapshot of production taken to refresh staging
19:00 | Load spikes: spam, plus a mass deletion job
23:00 | Secondary falls behind; needed WAL is gone
23:00 to 23:27 | Rebuild attempts with pg_basebackup hang
23:27 | *Data directory deleted on the primary, not the secondary
23:30 | Search for a usable backup begins
1 Feb, about 18:00 | Restore from the 17:20 snapshot completes
~~~

- **About 17:20.** An engineer took an **LVM snapshot** (a point-in-time copy of a logical disk volume) of the production database to refresh the staging environment for load testing. This was an ad hoc, manual action, unrelated to backups.
- **About 19:00.** Database load spiked. Part of it was spam. Part was a background job **hard-deleting** an account that had been wrongly flagged for abuse, which turned out to be a GitLab employee with a large amount of data. The deletes generated a lot of WAL.
- **About 23:00.** Replication to the secondary broke. It had fallen behind and the primary had already recycled the WAL segments it needed. With no WAL archive, the secondary had to be rebuilt by hand. Repeated `pg_basebackup` attempts appeared to **hang silently**; configuration changes (to `max_wal_senders` and `max_connections`) followed, and PostgreSQL needed restarting.
- **About 23:27.** Believing the target data directory on `db2` (the secondary) was blocking `pg_basebackup`, the engineer ran a recursive delete of the PostgreSQL data directory. The terminal was connected to **`db1`, the primary**. The command was stopped within seconds, but of roughly **300 GB** of data, only about 4.5 GB remained.

Then the team went looking for a backup.

## The backup failures

~~~dg-layers Five safety nets, and why each one failed.
*pg_dump to S3 | pg_dump 9.2 against Postgres 9.6: bucket empty
Failure emails | rejected by DMARC: nobody knew
Azure disk snapshots | enabled for NFS servers, not the database
Replication | for failover, not backup; secondary already wiped
LVM snapshots | made for staging every 24 h; a 6-hour-old one saved them
~~~

1. **`pg_dump` backups to S3 were empty.** The backup job ran on an application server that had no PostgreSQL data directory. The packaging tooling used the data directory to decide which PostgreSQL version's binaries to run, and without it fell back to **`pg_dump` 9.2**, which refuses to dump a **PostgreSQL 9.6** server. The job had been failing, producing only small or empty files, for an unknown length of time.
2. **Nobody was told.** The cron job did send failure notifications by email. But **DMARC** (an email authentication standard that lets receiving servers reject mail that fails sender checks) was not configured for those cron emails, so the receiving server **rejected** them. The alerts that would have reported the broken backups silently vanished.
3. **Azure disk snapshots** were enabled for the NFS file servers but **not for the database servers**. Even where they existed, restoring from snapshots stored in another storage account could have taken days.
4. **Replication** is for failover, not backup. It faithfully copies deletions and corruption to the secondary within milliseconds. In any case, the secondary had just been wiped as part of the rebuild.
5. **LVM snapshots** existed only to refresh staging, taken every 24 hours, not as disaster recovery. By luck, the **manual snapshot from 17:20**, about 6 hours old, was the most recent usable copy of production.

~~~dg-matrix What each mechanism needed in order to work as a backup, and whether it had it on the day.
Mechanism | Ran successfully | Failures reported | Covers the database | Survives a bad delete | Restore tested
pg_dump to S3 | no, wrong version | no, emails rejected | yes | yes | no
Azure snapshots | partial, NFS only | partial | no | yes | no
Streaming replica | yes, until it broke | yes | yes | no, copies deletes | no
LVM snapshots | yes, every 24 h | partial | yes | yes | partial, used for staging
~~~

## The cost

- Data created between **17:20 and 23:27** was lost: an estimated **5,000 projects, 5,000 comments and 700 new user accounts**, plus issues, merge requests and webhooks from that window. Git repositories and wikis were stored separately on file servers and were unaffected, so code was safe; only the database records about it were lost.
- Restoring took about **18 hours**. The 17:20 snapshot lived on staging's cheap Azure disks, throttled to about **60 Mbit/s**, in another region, and copying hundreds of gigabytes at that rate takes most of a day.
- After restoring, database sequences were bumped by 100,000 so new rows would not reuse IDs that might already exist in external systems, such as webhook payloads that had been sent during the lost window.

<div class="ex">
<div class="lab">Worked example</div>

**why the restore took 18 hours**

The snapshot to copy was on the order of a few hundred gigabytes. At 60 Mbit/s, about 7.5 MB/s, copying 300 GB takes 300,000 / 7.5 = 40,000 seconds, about 11 hours, before any restore, verification or restart work.

Add the time spent first discovering that S3 was empty and that database snapshots did not exist, plus PostgreSQL recovery and checks, and 18 hours is unsurprising.

A restore drill would have measured this number in advance. A recovery time objective (lesson 14.08) of, say, 1 hour would have shown at once that a staging snapshot on throttled disks was not a viable recovery path.

</div>

## Root causes

GitLab ran a "five whys" analysis (repeatedly asking why something happened until reaching causes that can be fixed). In summary:

- The wrong directory was deleted because the rebuild procedure was **manual and poorly documented**, and it was not obvious from the terminal which host it was connected to (both prompts looked alike).
- The rebuild was needed because replication broke under load and the needed WAL had been discarded, with **no WAL archive** to fall back on.
- The restore was slow and lossy because the real backups (`pg_dump`, Azure snapshots) **did not work** or did not exist for the database.
- That went unnoticed because failure alerts were **blocked**, and because the backup procedures had **never been tested** and had **no owner**.

Notice that only the first point is about the human mistake. The engineer's error turned an incident into a six-hour data loss because every layer meant to catch it had already failed, silently, earlier.

## What would have caught them

The question asks for the single practice. It is **regularly restoring from backups and checking the result**, automatically, with an alert if a restore does **not** succeed. A restore test would have revealed:

- the empty S3 bucket on its first attempt (nothing to restore);
- the `pg_dump` version mismatch (the dump step fails, or produces nothing usable);
- that the database servers had no disk snapshots (the test looks for the latest one and finds none);
- how long a real restore takes from each source, long before the day it mattered;
- whether the restored data is complete, by comparing row counts and recent timestamps against production.

Crucially, a restore test that alerts on **missing success**, not just on reported failure, also catches the blocked emails. When the alert is "no successful restore reported in the last 8 days", it fires whether the cause is a broken dump, a broken job, a broken mailer or a deleted cron entry. This is the general rule for monitoring anything that is supposed to happen periodically: alert on the absence of the success signal (a "dead man's switch"), because failure signals can fail too. As lesson 14.07 puts it: you only know that you can recover your recent state if you actually do so.

~~~dg-flow A weekly restore drill that alerts on the absence of success rather than the presence of failure.
[Restore] Scheduler -(weekly)-> Fetch latest dump + WAL -(restore)-> *Scratch database
[Check] Scratch database -(row counts, newest row age)-> Compare with production
[Signal] Compare with production -(pass)-> restore_success timestamp -(older than 8 days)-> Page on-call
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a weekly restore drill**

Every Sunday a job fetches the latest `pg_dump` from S3 and the WAL archive, restores them into a fresh database server, runs `SELECT count(*)` on key tables and compares against production (within a tolerance), and checks the age of the newest row (should be under 24 hours).

On success it pushes a metric `restore_last_success_timestamp`. An alert fires if that timestamp is more than 8 days old, whatever the reason.

In GitLab's situation, the first Sunday after the job started using `pg_dump` 9.2 would have produced no success metric. Eight days later, someone would have been paged, the version mismatch found and fixed, months before 31 January.

</div>

## What GitLab changed

GitLab's improvement list matches this diagnosis:

- **Automated testing of database backup restores**, run on a schedule.
- Backup monitoring through Prometheus metrics and a public dashboard, instead of email.
- Hourly LVM snapshots, and Azure disk snapshots enabled for database servers.
- **Continuous WAL archiving** for point-in-time recovery, which also lets a lagging replica catch up instead of being rebuilt.
- A named **owner** for data durability.
- Better replication runbooks, and making it obvious which host a terminal is connected to (distinct prompts for production).

## Common mistakes this incident illustrates

- Treating replication as a backup.
- Monitoring backups by "no error emails arrived".
- Backup jobs that run on a different machine from the data, with silent version fallbacks.
- Snapshots and backups that nobody has ever restored.
- Manual, undocumented recovery procedures run late at night under pressure.
- No recovery time objective, so nobody knows that a restore will take 18 hours until it does.

## In an interview

When backups come up, say "a backup you have not restored is a hope, not a backup", then give the concrete practice: scheduled automated restores into a scratch environment, data checks, and alerting on the absence of a success signal. Mention that replicas copy mistakes, and that RPO and RTO (lesson 14.08) should be measured by drills, not assumed.

<div class="take">
<div class="lab">Key takeaway</div>

GitLab had five layers of backup and none worked: pg_dump failed on a version mismatch, its failure emails were rejected by DMARC, database disk snapshots were not enabled, replication only copied the damage, and the staging snapshot that saved them was luck. Regular automated restore tests, alerting when a successful restore is not reported, would have exposed all of it long before the day it mattered.

</div>
