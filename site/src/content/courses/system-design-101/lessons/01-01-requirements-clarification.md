---
id: "01.01"
course: "system-design-101"
module: "01"
slug: "requirements-clarification"
title: "Requirements clarification"
keyIdeas: ["Before drawing anything, pin down **what** the system must do (functional requirements) and **how well** (non-functional requirements).", "Functional: the 3-5 core user actions. For a URL shortener: create a short link, redirect, maybe custom aliases and expiry. Push everything else out of scope out loud.", "Non-functional: scale (users, QPS, data size), latency target, availability vs consistency, durability, read/write ratio.", "Ask about the users: who calls it, from where, how often, and what happens if it's down for a minute.", "Write the agreed list down and refer back to it. Every later decision should trace to one requirement.", "Common mistake: designing for features nobody asked for, or skipping scale numbers and then not being able to justify sharding or caching."]
question: "For \"design a URL shortener\", list 3 functional requirements and 3 questions you'd ask the interviewer."
answer: "Functional: shorten a URL, redirect a short URL to the original, optionally set a custom alias or expiry. Questions: How many new links per day and how many redirects? Do links expire? Can two users shortening the same URL get the same short link? Do we need click analytics? How short must the code be? Must redirects work during a regional outage?"
sources: [{"title": "ByteByteGo: A Framework For System Design Interviews", "url": "https://bytebytego.com/courses/system-design-interview/a-framework-for-system-design-interviews", "note": "The 4-step approach; step 1 is scoping."}, {"title": "Hello Interview: Delivery Framework", "url": "https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery", "note": "Splitting functional vs non-functional requirements in about 5 minutes."}, {"title": "System Design Primer: How to approach a question", "url": "https://github.com/donnemartin/system-design-primer#how-to-approach-a-system-design-interview-question", "note": "Step 1: use cases, constraints, assumptions."}]
---

## Why this step comes first

A system design prompt is deliberately vague. "Design Twitter" could mean a weekend prototype for a hackathon or a platform for 500 million people, and those are completely different systems. The prototype is one server and one database. The platform needs precomputed feeds, sharded storage, a media pipeline and a CDN. If you start drawing before you know which one you're building, you will almost certainly build the wrong one, and every minute spent on it is wasted.

The same is true at work. A product manager who says "we need notifications" might mean a nightly email digest or sub-second push alerts on a trading app. The first is a cron job; the second is a fleet of persistent connections. Requirements clarification is the cheap step that prevents the expensive mistake.

So the first few minutes go on turning a vague prompt into a short, written, agreed list of what you are building and how well it must work. In a 45-minute interview that's roughly 3 to 8 minutes. On a real project it might be a one-page design brief. Either way, the output is the same: a list you can point back to when you make every later decision.

~~~dg-timeline How the first ten minutes of a 45-minute design session typically go.
0 min | Restate the prompt in your own words
1 min | List candidate features, ask which matter most
3 min | *Agree the top 3 functional requirements and what's out of scope
5 min | Ask about scale, latency, consistency and durability
7 min | *Write 3 to 5 measurable non-functional requirements
9 min | Note headline numbers, then start the design
~~~

## Functional requirements: what the system does

Functional requirements are the features, written as things a user (or another system) can do. A good habit is to phrase each one as "Users should be able to...". That phrasing forces you to think about who acts and what they achieve, rather than about implementation.

Narrow them to the **top three**. Real products have hundreds of features, but a design can only go deep on a few, and the core few usually decide the architecture. For a Twitter-like product:

- Users can post short text posts (tweets), optionally with an image.
- Users can follow other users.
- Users can see a home feed of recent posts from people they follow.

Everything else is explicitly declared out of scope: direct messages, search, trends, ads, notifications. Saying "I'll leave search out for now, but the design should allow adding it later" is a sign of control, not a gap. It shows you know the feature exists and that you're making a conscious choice to focus.

Why only three? Because each extra feature drags in its own storage, APIs and edge cases. Search alone needs an inverted index, a separate ingestion pipeline and ranking. Including it means spending your limited time on a second system instead of going deep on the first.

### Turning a feature into something precise

Even a simple feature hides choices. "Users can see a feed" raises questions:

- Is it ordered by time, or ranked by relevance?
- How far back does it go? The last 24 hours, or everything?
- Is it the same for everyone who follows the same people, or personalised?
- Does a new post have to appear within one second, or is a minute fine?

Each answer changes the design. A chronological feed can be precomputed per user. A ranked feed needs a scoring step at read time. A one-minute delay allows asynchronous fan-out through a queue; a one-second guarantee might not.

~~~dg-compare A vague feature versus a precise one: the precise version is something you can design and test against.
[Vague]
Users see a feed
- Ordering not defined
- Freshness not defined
- No idea how big the feed is
[*Precise]
Users see the 50 newest posts from people they follow, newest first
+ Order: reverse chronological
+ New posts visible within 60 seconds
+ Paged, 50 at a time, back 30 days
~~~

## Non-functional requirements: how well it does it

Non-functional requirements (NFRs) describe qualities: speed, scale, reliability, correctness, cost. Write them as "The system should...". The key is to make them **specific to this system** and, where possible, **measurable**.

"Low latency" is weak, because every system wants it and it can't be checked. "The home feed renders in under 200 ms at the 99th percentile" is useful. It names the operation (feed render), the measure (p99) and the threshold (200 ms). Later it will justify specific choices, like a cache in front of the database or precomputing feeds instead of building them on every request.

A checklist to pick 3 to 5 from:

- **Scale**: users, requests per second, data size, read/write ratio, traffic bursts (a live sports final, Black Friday).
- **Latency**: which operations must be fast, and how fast. Not every operation needs to be: posting can take 500 ms; scrolling the feed can't.
- **Availability vs consistency**: is it worse to show stale data or to show an error? For a feed, stale is fine. For a bank balance, it isn't.
- **Durability**: can any acknowledged data ever be lost? A chat message, a payment, a medical record: never. A typing indicator: who cares.
- **Security and compliance**: authentication, privacy laws (GDPR), data residency (data must stay in the EU).
- **Environment**: mobile clients on flaky networks, a small budget, a team that only knows Java, an existing Postgres you must reuse.

~~~dg-matrix How strict each quality needs to be depends on the product. The same checklist gives very different answers.
Product | Latency target | Stale reads OK? | Data loss OK? | Scale
Social feed | 200 ms p99 | yes, seconds | partial, likes can drift | 100M+ users
Bank transfer | 1 s | no | no | thousands per second
Chat app | 500 ms delivery | partial, typing status | no, messages | 50M users
Analytics dashboard | 5 s | yes, minutes | partial, sampled | billions of events
~~~

### Rank them, because they conflict

NFRs fight each other. Strong consistency costs latency (you must coordinate replicas) and availability (you may have to refuse requests during a network problem). Very high availability costs money (redundant everything). Low latency worldwide means data in many regions, which makes consistency harder.

So after listing them, rank them. "Correctness of payments beats latency; latency of the feed beats freshness." Agreeing on that order early means many later decisions make themselves.

## Questions worth asking

Good questions each remove a branch of the design tree. For a news feed:

- Mobile, web, or both? (Mobile means flaky networks, battery, push notifications.)
- Which features matter most? (Tells you where to go deep.)
- Chronological or ranked? (Precompute vs score at read time.)
- How many people can a user follow? (5,000 caps fan-out; "unlimited" means celebrities with 100 million followers.)
- How much traffic? (10 million daily active users is one region with caching; 1 billion is a global system.)
- Text only, or images and video too? (Video brings transcoding, object storage, CDN.)
- How fresh must the feed be? (Seconds vs minutes.)

~~~dg-tree Each answer prunes the design space. Here, two questions already decide most of the feed architecture.
Design a news feed
  Ranked by relevance? | yes
    Score candidates at read time
    *Precompute candidates, rank on read
  Chronological? | yes
    Follows capped at 5,000?
      *Fan out on write: push post IDs to followers
    Unlimited followers?
      Hybrid: push for most, pull for celebrities
~~~

If the interviewer won't answer, **state an assumption out loud and write it down**: "I'll assume 100 million daily active users and that a feed can be up to a minute stale." You'll lean on those assumptions later, and writing them down means nobody can say you changed the problem halfway through.

## Should you estimate capacity now?

Some people run detailed back-of-the-envelope numbers at the start; others only calculate when a number will change a decision. A good middle path is to note the **headline numbers** now (daily active users, rough requests per second, data growth per day) and save detailed maths for the moment it decides something, such as whether the data fits on one machine or whether one database can take the write rate.

For example, "100 million daily users, each loading the feed 10 times a day" is 1 billion feed loads a day, which is about 12,000 per second on average and maybe 30,000 at peak. You don't need more precision than that to know you'll need caching and many stateless servers.

## Common mistakes

- **Designing before agreeing.** Drawing boxes in minute one. You'll get the scale or the core feature wrong.
- **Feature sprawl.** Listing ten features and going deep on none.
- **Unmeasurable NFRs.** "Scalable, fast and reliable" says nothing.
- **Silent assumptions.** Assuming a number without saying it, then contradicting it later.
- **Forgetting to write it down.** If it isn't on the board, it'll be forgotten by minute 30.

<div class="ex">
<div class="lab">Worked example</div>

**a URL shortener in about two minutes**

Functional: create a short link from a long URL; visiting the short link redirects to the original; optional custom alias; optional expiry date.

Out of scope: analytics dashboards, user accounts, link editing.

Non-functional: redirects are fast (p99 under 100 ms) and highly available (99.99%), since a broken link is very visible; reads outnumber writes about 100 to 1; a short link must never point to the wrong URL (correctness over availability for creation); 100 million new links per month.

Headline numbers: 100M links a month is about 40 writes per second on average; at 100:1, about 4,000 redirects per second. Over 5 years that's 6 billion links, so a 7-character code from 62 symbols (62^7 ≈ 3.5 trillion) is plenty.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**what changes if the answers change**

If the interviewer says "links expire after 24 hours and we expect 10 billion a month", storage becomes a rolling window instead of growing forever, and the write rate (about 4,000 per second) now needs a distributed ID generator rather than a single counter. Same feature list, very different design: that is why you ask.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

agree on a small, written list of features and measurable qualities before designing, rank the qualities because they conflict, and state any assumption you make out loud. Every later decision should point back to one of these lines.

</div>
