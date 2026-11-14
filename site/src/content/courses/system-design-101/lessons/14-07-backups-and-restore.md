---
id: "14.07"
course: "system-design-101"
module: "14"
slug: "backups-and-restore"
title: "Backups and restore"
keyIdeas: ["Backups only matter if you can restore from them. Test restores regularly and measure how long they take.", "Types: logical dumps (portable, slow for big DBs), physical snapshots, and continuous archiving (base backup + WAL) for point-in-time recovery.", "Follow 3-2-1: 3 copies, 2 different media, 1 offsite. Keep backups in a separate account so one compromised credential can't delete everything.", "Protect against logical errors (bad deploy deleting rows), not just hardware loss. Replicas copy mistakes instantly; backups with history don't.", "Monitor that backups actually run and have plausible sizes."]
question: "Why is an untested backup not a backup?"
answer: "Because a backup's only purpose is the restore, and the restore path is exactly what silently breaks when it never runs: jobs that produce empty or partial files, lost encryption keys, missing tools or credentials, unreadable tapes, schemas that no longer match, or restores that take far longer than the outage you can afford. You only find out when you need it, which is the worst possible time. You only know you can recover if you actually do, so run restores continuously and automatically, verify the restored data and the time it took against your RTO, and alert when a restore test does not report success."
sources: [{"title": "Google SRE book: Data Integrity", "url": "https://sre.google/sre-book/data-integrity/", "note": "\"No one wants backups, they want restores.\""}, {"title": "PostgreSQL docs: Backup and Restore", "url": "https://www.postgresql.org/docs/current/backup.html", "note": "Dumps, base backups, PITR."}]
---

## Nobody wants backups; they want restores

"No one really wants to make backups; what people really want are restores." A backup is only valuable if you can restore from it, within the time your users can tolerate, to the state you actually need. A green "backup job succeeded" tick tells you a process exited cleanly. It does not tell you the file is complete, readable, decryptable, or restorable in less than three weeks. So measure teams on restore success, not on backup jobs completing.

Two kinds of copy are often confused:

- **Backups** exist to load back into the running application quickly after a loss. They are taken daily, hourly or continuously, because everything since the last one is at risk.
- **Archives** are long-term copies kept for audit, legal or compliance reasons, such as financial records kept for 7 years. A week to retrieve one is fine, and they often cannot be loaded straight back into the current version of the application.

Two numbers frame every backup design (lesson 14.08 develops them for whole-site disasters). The **recovery point objective (RPO)** is how much recent data you can afford to lose, measured in time: with nightly backups, up to 24 hours of writes. The **recovery time objective (RTO)** is how long the restore may take before service is back.

## Ways data gets lost

Data loss is classified along three axes:

- **Cause**: user action, operator mistake, application bug, infrastructure defect, faulty hardware, or site disaster.
- **Scope**: widespread (everyone's data) or narrow (a few users or one table).
- **Rate**: "big bang" (a million rows replaced by ten in a minute) or **creeping** (ten rows a minute, for weeks).

That gives 24 kinds of failure (6 × 2 × 2), and no single defence covers them all. A dropped table is noticed in seconds; a bug that silently truncates one field on 0.1% of new records may go unnoticed for months. When Google analysed 19 of its own data recovery efforts, the most common user-visible losses came from **software bugs** deleting data or breaking references between records, not from hardware. The hardest ones were low-grade corruption discovered **weeks or months** later, by which time every recent backup contained the damage too.

## Replication is not a backup

"Replication and redundancy are not recoverability." Replicas, RAID and multi-AZ databases protect against hardware failure: a disk dies, a machine dies, a datacentre loses power. But if a bug or a person deletes or corrupts data, replication faithfully copies the damage to every replica within milliseconds. `DELETE FROM orders` with a missing `WHERE` clause runs on all three replicas. What you need are copies **from a point in time before** the damage.

~~~dg-compare Replication versus backups: they protect against different failures, and you need both.
[Replication]
primary -(milliseconds)-> replica -> replica
+ survives disk, host, AZ failure
+ failover in seconds, almost no data loss
- copies deletes and corruption instantly
- no way to go back in time
[*Point-in-time backups]
primary -(snapshot + log)-> separate storage
+ restore to before the bad write
+ survives bugs, operators, ransomware
- restore takes minutes to days
- loses writes since the last copy
~~~

## Defence in depth: three layers

~~~dg-layers Three layers of protection, from fastest to recover at the top to widest coverage at the bottom.
*Soft deletion | deleted data hidden but kept 15 to 60 days; undo in minutes
Tiered backups | snapshots (minutes), distributed copies (hours), offsite (days)
Early detection | validators find corruption before it spreads into every backup
~~~

**1. Soft deletion.** Deleted data is marked as deleted and hidden but kept for a while before being destroyed. Typical periods are 15, 30, 45 or 60 days; Gmail's Trash keeps messages 30 days. This covers the most common case, a user or a bug deleting something, and recovery takes minutes, without a restore at all. Most account hijackings and integrity problems are noticed within 60 days, so keeping soft-deleted data much longer adds little. (Lesson 14.09 covers how soft deletion interacts with privacy law.)

**2. Backups, tiered.**

- **Tier 1**: frequent snapshots close to the live data, kept for hours to a few days, restorable in minutes. Database snapshots on the same storage system are typical.
- **Tier 2**: copies on a separate distributed filesystem or object store, kept for a week or two, restorable in hours. If you release twice a week, keep at least two weeks, so a bug can be caught and you can restore to before the release that introduced it.
- **Further tiers**: tape or offsite object storage in another region or account, taken less often and kept longer, which survive the loss of a whole site or a compromised account.

Google often keeps 30 to 90 days of backups. Copies should sit on **different technology** and ideally in a **different trust domain** from the live data. If a filesystem bug can destroy the data, a backup on the same filesystem can be destroyed with it. If an attacker or a buggy script with production credentials can delete the database, it should not also be able to delete the backups. That is why object-lock or write-once storage, separate cloud accounts and offline copies matter, especially against ransomware.

**3. Early detection.** Bad data spreads over time: into new records derived from it and into every new backup. Once your oldest backup is newer than the corruption, it is unrecoverable. **Out-of-band validators** check invariants independently of the application: do files in Drive match their folder listings? Does every message reference in Gmail point at a real message? Does the sum of ledger entries equal the account balance? Gmail's validators detect inconsistencies within 24 hours, which is what lets developers ship more than once a week. Validate only the invariants whose failure would really hurt users; overly strict validators raise false alarms and get ignored.

## Point-in-time recovery: how it works

Most production databases combine a periodic **base backup** (a full copy) with a continuous archive of the **write-ahead log** (WAL in PostgreSQL, binlog in MySQL), the ordered record of every change. To restore to any moment, you load the latest base backup taken before that moment, then replay the log up to the exact second you want.

~~~dg-timeline Point-in-time recovery: restore Sunday's base backup, replay the log to one second before the bad DELETE.
Sun 02:00 | Base backup taken
Sun to Tue | WAL segments archived every minute or every 16 MB
Tue 14:32:00 | *DELETE without WHERE wipes the orders table
Tue 14:40 | Restore starts: copy base backup to new server
Tue 15:20 | Replay WAL from Sunday to Tue 14:31:59
Tue 16:05 | Restored copy verified, orders table exported back
~~~

PostgreSQL offers three backup methods, and the third is the one behind point-in-time recovery:

| Method | How | Good | Limits |
| --- | --- | --- | --- |
| SQL dump | `pg_dump` exports one database as SQL from a consistent snapshot; `pg_dumpall` for the whole cluster | portable across versions; restore single tables | slow for large databases; no point-in-time recovery |
| File system copy | copy the data directory with the server stopped, or an atomic snapshot of all volumes | fast for big data | whole cluster only; same major version; needs downtime or atomic snapshots |
| Continuous archiving (PITR) | a base backup (`pg_basebackup`) plus every WAL segment archived (`archive_command`) | restore to any moment, e.g. one second before a bad `DELETE` | more moving parts; a failing archive must be alerted on |

Note what RPO and RTO look like here. RPO is the gap since the last archived WAL segment, often under a minute. RTO is the time to copy the base backup plus the time to replay the log, and replay time grows with the distance from the base backup. That is why base backups are taken daily or weekly rather than monthly.

<div class="ex">
<div class="lab">Worked example</div>

**how long will the restore take?**

The database is 4 TB. Base backups are taken weekly, and the system writes 300 GB of WAL per day. Disaster strikes on day 6.

Copying 4 TB from object storage at 500 MB/s takes 4,000,000 ÷ 500 = 8,000 s, about 2.2 hours.

Replaying 6 × 300 GB = 1.8 TB of WAL at, say, 150 MB/s takes 12,000 s, about 3.3 hours.

Total RTO is about 5.5 hours plus verification. If the target is 2 hours, you need daily base backups (cutting replay to under 40 minutes), a faster copy path (local snapshots), or a delayed replica (a replica deliberately kept, say, 1 hour behind, so you can stop it before it applies a bad change).

</div>

## Untested backups fail

Recovery paths that never run quietly break. Real failures include backups that are zero bytes because a disk filled, encryption keys that were rotated and lost, a restore that needs a tool nobody has installed since the old engineer left, credentials that expired, a schema in the backup that no longer matches the application, and restores that would take three weeks because the bandwidth was never measured. You only know you can recover your recent state if you actually do so. So:

- run restores **continuously and automatically**, end to end, into a test environment;
- check that the restored data is complete and correct (row counts, checksums, application-level queries), not just that the job finished;
- measure how long a restore takes, and compare it with your RTO;
- alert when a restore test **fails to report success** (a missing heartbeat), not only when it reports failure, because a broken test system usually reports nothing;
- rehearse the rare restores too: a single table, a single user's data, a whole region from offsite copies.

Two cases show the difference:

- **Gmail, February 2011.** A bug lost user data, and Google restored it from tape for the first time at that scale. Because restores had been rehearsed, the team gave a reliable estimate, recovered over 99% of the data before the estimated finish, and restored every affected account within hours of it.
- **Google Music, March 2012.** A refactored deletion pipeline had a race condition and deleted about 600,000 audio files belonging to 21,000 users. It went unnoticed for a month.

<div class="ex">
<div class="lab">Worked example</div>

**the Google Music restore, by the numbers**

Of the roughly 600,000 deleted files, 436,223 were on tape. Restoring them took 5,475 restore jobs and recalling 5,337 tapes, about 1.5 PB.

17 tapes turned out to be bad, and the offsite vendor left out 1,862 tapes from the shipment, so redundant copies were used. It took almost 7 days.

About 161,000 files had never been backed up at all, because they were uploaded after the last backup and deleted before the next. Those users had to re-upload them.

The fixes: alert on the **global** deletion rate (for example above 10× its usual 95th percentile), so a deletion bug is caught in hours rather than a month, and keep more frequent backups for new uploads.

</div>

## Common mistakes

- **Treating replicas as backups.** They copy the mistake.
- **Backups in the same account and region as production**, deletable with the same credentials.
- **Never restoring.** The first real restore is during the incident, and it fails or takes days.
- **Retention shorter than detection time.** If corruption is noticed after 30 days and you keep 14, every backup is already bad.
- **Measuring backup success, not restore success.**

## In an interview

For any stateful component, say how it is backed up and how fast it restores. "Postgres with daily base backups and continuous WAL archiving to object storage in another account: RPO about a minute, RTO a few hours, tested weekly by an automated restore." Mention soft delete for user mistakes and validators for silent corruption.

<div class="take">
<div class="lab">Key takeaway</div>

backups only count if restores work, so test restores continuously and automatically and measure their time. Replication copies mistakes, so you also need point-in-time copies on different technology, outside production's credentials. Layer soft deletion, tiered backups and validators that catch corruption before it reaches every backup.

</div>
