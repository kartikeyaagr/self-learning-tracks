---
id: "14.03"
course: "system-design-101"
module: "14"
slug: "incident-response"
title: "Incident response"
keyIdeas: ["Incident response restores service fast, then learns from what happened.", "Roles: **incident commander** coordinates and decides; **ops lead** works on the fix; **communications lead** updates stakeholders. Separating them stops the fixer from also juggling coordination.", "Mitigate first (roll back, fail over, shed load), find the root cause later.", "Keep a live incident document with a timeline.", "Afterwards, write a **blameless postmortem**: what happened, impact, root causes, what went well, and action items with owners."]
question: "Why separate the incident commander from the person fixing the problem?"
answer: "Fixing and coordinating need different kinds of attention. Fixing needs deep focus on one technical thread; coordinating needs a wide view of the whole incident: whether the current approach is still right, whether anyone is mitigating instead of diagnosing, who is working on what, and what stakeholders need to hear. One person cannot do both. In an unmanaged incident the on-call engineer gets absorbed in debugging, nobody coordinates helpers or communications, and someone freelances an untested change that makes things worse. A separate commander keeps the big picture, makes calls like \"roll back now\", shields the fixers from interruptions, and enforces that only the operations lead changes production."
sources: [{"title": "Google SRE book: Managing Incidents", "url": "https://sre.google/sre-book/managing-incidents/", "note": "Roles: commander, ops, comms."}, {"title": "Google SRE book: Postmortem Culture", "url": "https://sre.google/sre-book/postmortem-culture/", "note": "Blameless postmortems."}]
---

## Why incidents need structure

An **incident** is an unplanned event that degrades a service enough that it needs a coordinated response: users cannot check out, a region is down, data is being corrupted. Every team has the technical skill to fix most incidents. What usually fails is the response itself: who is doing what, who is talking to whom, and who is still looking at the big picture.

Consider a typical unmanaged incident. The on-call engineer, call her Mary, is paged when one datacentre stops serving. Two more fail, the remaining ones overload, and the whole service goes down. Then the response falls apart:

- **Tunnel vision.** Mary dives into logs and rolls back a release, which does not help. Deep in the details, she loses sight of the bigger picture and of simple mitigations such as shedding load or draining traffic to a healthy region.
- **Poor communication.** Developers join informally and nobody knows who is doing what. Two people restart the same service. Executives demand updates directly from Mary and suggest fixes, pulling her out of the work every five minutes.
- **Freelancing.** Another engineer, Malcolm, on his own initiative, pushes an untested CPU affinity change to production. The servers restart with it and fail, making things worse. Nobody knew he was doing it, so for twenty minutes nobody can explain the new failure.

The fix is not smarter engineers. It is a structure that gives each person one job and makes the state of the response visible to everyone.

## The incident lifecycle

An incident moves through recognisable stages. Naming them helps everyone know what the current goal is: during mitigation the goal is to stop user pain, not to understand root cause.

~~~dg-state The lifecycle of an incident: mitigation comes before root cause, and nothing is closed until the follow-ups are tracked.
[*] -> Detected
Detected -> Declared : alert or report, IC named
Declared -> Mitigating : roles assigned, channel opened
Mitigating -> *Mitigated : user impact stopped
Mitigated -> Resolved : root cause fixed, temporary changes reverted
Resolved -> Postmortem : review written and reviewed
Postmortem -> Closed : action items owned and tracked
Mitigated -> Mitigating : impact returns
~~~

- **Detected**: an alert fires, or a customer or engineer reports a problem.
- **Declared**: someone decides this is an incident, names an incident commander and opens a channel.
- **Mitigating**: the team works to stop user impact as fast as possible.
- **Mitigated**: users are no longer hurt, though the system may be in a temporary state (a flag disabled, a region drained).
- **Resolved**: the underlying cause is fixed and temporary changes are reverted.
- **Postmortem** and **closed**: the team writes up what happened and tracks follow-up work to completion.

## Roles: separate the deciding from the doing

A good incident process is modelled on the **Incident Command System** used by fire and emergency services. Its core idea is a **recursive separation of responsibilities**: everyone knows their role and stays inside it. That also frees people to act without second-guessing each other.

~~~dg-tree Incident roles: the commander owns every role not yet handed out, and large incidents split recursively.
*Incident commander | overall picture, assigns work, holds unassigned roles
  Operations lead | the only people who change production
    Responder: database
    Responder: networking
  Communications lead | status page, stakeholder updates, incident doc
  Planning lead | bugs, handoffs, food, list of temporary changes
  Scribe | timeline of actions and decisions
~~~

- **Incident commander (IC)**: holds the overall picture, sets priorities, assigns work and removes blockers. The IC holds every role that has not been handed out yet. In a small incident the IC might also be the comms lead; in a large one each role has its own person.
- **Operations lead**: works with the IC to actually change the system. **Only the operations team changes production during the incident.** That single rule stops a Malcolm. Anyone with an idea brings it to the IC or ops lead instead of acting on it.
- **Communications lead**: the public face of the response. Sends regular updates to stakeholders and users (status page, support team, executives), shielding the responders from interruptions. Often keeps the incident document current.
- **Planning lead**: handles longer-term needs: filing bugs, arranging handoffs and food on a long incident, and recording every temporary change ("we disabled the recommendations cache flag at 14:20") so it can be reverted later.

### Why the commander must not be the fixer

Fixing needs deep focus on one technical thread: reading a stack trace, testing a hypothesis, watching a graph after a change. Coordinating needs a wide view: is the current approach still the right one, is anyone working on mitigation rather than diagnosis, who needs an update, who else should join, how long have we been at this? These two modes of attention conflict. A person deep in a debugger stops noticing that the incident is 40 minutes old, that a second team has quietly started a risky change, or that a simple failover would end the outage now. That is exactly what failed for Mary.

There is also a decision-quality reason. The fixer is invested in their current theory. A commander who is not hands-on can ask "what if it's not the deploy?" and redirect people without ego. And there is a communication reason: if the fixer is also the person executives call, every update costs ten minutes of focus.

When any role gets too big, its holder asks for more people or splits off a **sub-incident** with its own commander, which reports to the main one. The structure scales recursively.

## Running the response

- **A known command post**: one agreed place where responders coordinate, such as a dedicated chat channel (`#inc-2024-11-12-checkout`) or a video bridge. Chat gives you a timestamped log for free.
- **A live incident document**: the IC's most important artefact. Many people can edit it at once. It holds current status, user impact, a timeline, actions taken and who owns what, with the most important information at the top. It is messy but useful, and it becomes the raw material for the postmortem.
- **Regular cadence**: the comms lead posts an update at a fixed interval (every 15 or 30 minutes) even if nothing changed. Silence makes stakeholders go hunting for responders.
- **Explicit handoff**: at the end of a shift, brief the new IC by voice or video and do not leave until they have said "I have command". Then announce in the channel who is in charge.

### Severity levels

Most organisations define severity levels in advance, so the size of the response matches the size of the problem without debate.

| Level | Meaning | Response |
| --- | --- | --- |
| SEV1 | major user-facing outage or data loss | IC and full roles, executives informed, status page |
| SEV2 | significant degradation, a key feature down | IC and ops, comms to affected teams |
| SEV3 | minor impact or a contained internal problem | on-call handles, ticket, no formal roles |

### When to declare

Declare early. It is far easier to close a small incident than to bolt structure onto a big one halfway through. Declare if any of these is true: you need a second team, customers can see the outage, or the problem is still unsolved after an hour of focused work.

### Mitigate first

The priorities, in order, are: stop the bleeding, restore service, and preserve evidence for later analysis. Rolling back, failing over, draining a region, disabling a feature flag or shedding load usually beats debugging during the outage. A rollback that works in 3 minutes is better than a root-cause fix that takes 3 hours, even if you are not sure the deploy caused it. Before mitigating, take a quick snapshot of evidence if it is cheap (a heap dump, a copy of the logs, the current config) because a restart may destroy it.

<div class="ex">
<div class="lab">Worked example</div>

**a checkout incident, minute by minute**

14:02: the checkout error-rate burn alert pages Priya. Error rate is 20%. She acknowledges and declares a SEV1 at 14:05, naming herself IC, and opens `#inc-checkout`.

14:07: she asks Sam to be ops lead and Leo to handle comms. Leo posts the first status page update at 14:10 ("Some customers cannot complete payment; we are investigating") and promises another in 30 minutes.

14:09: Sam notes a deploy of the payment service at 13:58. Priya decides: roll back now, investigate later. Another engineer suggests restarting the database; Priya logs the idea and asks him to gather database metrics instead of acting.

14:16: rollback complete. Error rate falls to 0.1% by 14:19. Priya marks the incident mitigated, keeps it open, and asks the planning lead to note that the deploy pipeline for payments is paused.

14:40: Leo posts the all-clear. Next day, a postmortem finds the new version's connection pool was half the size. User impact lasted about 17 minutes at roughly 20% errors, which spends about 8% of a 99.9% monthly error budget (burn rate 200 × 17 minutes ÷ 43,200 minutes).

</div>

~~~dg-timeline The same incident as a timeline: time to mitigate is the number users feel.
14:02 | Burn-rate alert pages on-call
14:05 | SEV1 declared, IC named, channel opened
14:07 | Ops and comms leads assigned
14:09 | *Decision: roll back the 13:58 deploy
14:16 | Rollback complete
14:19 | Errors back to normal: mitigated
14:40 | All-clear posted
Next day | Blameless postmortem
~~~

## Blameless postmortems

After the incident, the team learns from it. A **postmortem** is a written record: impact (who, how many, how long), a timeline, actions taken, root causes and contributing factors, what went well, where you got lucky, and follow-up actions with owners and priorities. Teams agree in advance on what triggers one, for example:

- user-visible downtime or degradation beyond a threshold;
- **any** data loss;
- on-call intervention such as a rollback or rerouting traffic;
- resolution time beyond a threshold;
- a monitoring failure (a human found the problem before the alerts did).

**Blameless** means the postmortem looks for contributing causes in systems and processes, not for a guilty person. It assumes everyone acted reasonably with the information and tools they had at the time. The reason is practical: if people fear punishment, they hide mistakes and stay quiet about near misses, and the organisation stops learning. The approach comes from healthcare and aviation. You cannot fix people, but you can fix systems so the next person makes a better choice. "An engineer ran a command on the wrong host" becomes "production and staging hosts have identical prompts and the tool asks for no confirmation".

Wording matters. A blaming note says the backend must be rewritten because it keeps breaking and someone is tired of being paged. A blameless version proposes the rewrite as an action item, because the current runbook is long and hard to learn, and future on-call engineers would benefit.

**Review.** Senior engineers review each draft for completeness. Was the key data collected? Is the impact assessed? Is the root cause analysed deeply enough (ask "why" several times, and look for multiple contributing factors, not one)? Are the actions specific, owned and prioritised? Was it shared widely? An unreviewed postmortem, or one whose action items never get done, might as well not exist. Teams keep the culture alive with a "postmortem of the month", reading groups on old incidents, and the **Wheel of Misfortune**, where new engineers replay a past incident as a role-play with a game master.

## Common mistakes

- **The IC debugs.** The commander starts typing commands and nobody is coordinating any more.
- **Debugging before mitigating.** Forty minutes spent finding the bug while a rollback was available.
- **Changes outside the ops lead.** Well-meaning freelancers make the incident harder to understand.
- **No update cadence.** Stakeholders fill the silence by pinging responders directly.
- **Forgotten temporary changes.** A flag disabled during the incident is still off three months later.
- **Postmortems that name a person.** People learn to hide mistakes, and the next incident is worse.

## In an interview

Incident response comes up in behavioural rounds and in operations-heavy design rounds. Describe the roles, the "only ops changes production" rule and mitigate-first. When asked for a story, structure it as detection, declaration, mitigation, resolution and the postmortem's system-level fixes.

<div class="take">
<div class="lab">Key takeaway</div>

give incidents a structure. One commander coordinates and holds the big picture; only the operations lead changes production; a communications lead handles updates; everything goes into a live document. Mitigate first, then write a blameless postmortem whose action items fix the system, not the person.

</div>
