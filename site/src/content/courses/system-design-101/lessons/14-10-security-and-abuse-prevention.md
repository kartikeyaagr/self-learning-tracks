---
id: "14.10"
course: "system-design-101"
module: "14"
slug: "security-and-abuse-prevention"
title: "Security and abuse prevention"
keyIdeas: ["Security basics: authenticate every request, authorize every action (check the user may access this specific object), validate input, encrypt in transit and at rest, keep secrets out of code, and patch.", "OWASP's top risks include broken access control, injection, and security misconfiguration.", "Abuse prevention: rate limits, CAPTCHAs on sensitive flows, email/phone verification, anomaly detection, blocking known bad IPs, limits on free usage.", "Public signup abuse examples: fake account farms (rate limit and verify), credential stuffing on login (rate limit per account and IP, MFA), enumeration of existing emails (return the same response either way)."]
question: "Name 3 abuse vectors for a public signup endpoint and a defence for each."
answer: "1. Automated fake signups by bots (spam, free-tier abuse): rate limit per IP and per device, show a CAPTCHA only when traffic looks automated, and require email or phone verification before the account can act. 2. Account enumeration through \"this email is already registered\": return the same response with the same timing either way, and tell existing users by email instead. 3. SMS or email pumping through verification messages, which costs money per message: rate limit per phone number, number prefix and IP, block risky prefixes you do not serve, cap and alert on daily spend, and watch the sent-to-verified ratio. Also limit payload size and request time at the gateway against resource exhaustion."
sources: [{"title": "OWASP Top 10", "url": "https://owasp.org/www-project-top-ten/", "note": "Most common web vulnerabilities."}, {"title": "OWASP API Security Top 10", "url": "https://owasp.org/API-Security/", "note": "API-specific risks."}]
---

## Security as a design property

Security in system design is less about clever cryptography and more about boring, consistent checks in the right places. Most real breaches come from a missing authorisation check, a misconfigured storage bucket, an unlimited endpoint or a leaked credential, not from someone breaking TLS. And a growing share of harm is **abuse** rather than intrusion: attackers using your legitimate features (signup, login, password reset, checkout, messaging) at a scale or in a way you never intended.

Two words get mixed up constantly:

- **Authentication** (authn) answers "who are you?": passwords, sessions, tokens, MFA.
- **Authorisation** (authz) answers "are you allowed to do this, to this object?": permission checks on every action.

A system can have perfect authentication and still leak every customer's data if it forgets authorisation on one endpoint.

A useful habit is **threat modelling**: for each component, ask what an attacker could do and what it would cost you. A simple checklist is STRIDE: Spoofing identity, Tampering with data, Repudiation (denying an action because nothing was logged), Information disclosure, Denial of service, and Elevation of privilege. You do not need to be exhaustive; you need to find the cheap, likely attacks before attackers do.

## The common vulnerability lists

OWASP, an open security community, maintains two widely used lists of the most critical risks, built from data on real vulnerabilities.

**OWASP Top 10 (2025), web applications:**

1. **Broken access control**: users can act outside their permissions, such as viewing another user's data or reaching admin functions.
2. **Security misconfiguration**: default passwords, open storage buckets, verbose error pages, unneeded features left on.
3. **Software supply chain failures**: vulnerable or malicious dependencies, build systems and update channels.
4. **Cryptographic failures**: sensitive data sent or stored without strong encryption; weak password hashing.
5. **Injection**: untrusted input interpreted as code or queries (SQL, shell commands, cross-site scripting).
6. **Insecure design**: missing security controls in the design itself, such as no rate limit on a password reset.
7. **Authentication failures**: weak passwords allowed, credential stuffing not stopped, broken sessions.
8. **Software or data integrity failures**: trusting updates, plugins or serialised data without verifying them.
9. **Security logging and alerting failures**: attacks not logged, logs not watched, no alert.
10. **Mishandling of exceptional conditions**: errors that fail open, leak information, or leave the system in a bad state.

**OWASP API Security Top 10 (2023):**

1. **Broken object level authorisation (BOLA)**: `GET /orders/1234` returns order 1234 without checking it belongs to the caller. It is the top API risk because every endpoint that takes an ID can have it.
2. **Broken authentication**.
3. **Broken object property level authorisation**: returning or accepting fields the user should not see or set (`is_admin: true` in a profile update, called **mass assignment**).
4. **Unrestricted resource consumption**: no limits on request rate, payload size, page size or expensive operations.
5. **Broken function level authorisation**: normal users can call admin endpoints.
6. **Unrestricted access to sensitive business flows**: automated abuse of legitimate features, such as bots buying all the tickets or mass account creation.
7. **Server-side request forgery (SSRF)**: making the server fetch a URL the attacker chooses, including internal addresses like the cloud metadata service at `169.254.169.254`.
8. **Security misconfiguration**.
9. **Improper inventory management**: forgotten old API versions and undocumented endpoints that still work.
10. **Unsafe consumption of APIs**: trusting data from third-party APIs too much.

## Broken object level authorisation, step by step

BOLA deserves a closer look because it is so common and so simple. The developer checks that the caller is logged in, then loads the object by the ID in the URL. Nobody checks that the object belongs to the caller.

~~~dg-seq A BOLA attack: authentication succeeds every time, authorisation is never checked.
Attacker > API : log in as a normal user (own account 7)
API -> Attacker : session token
Attacker > API : GET /invoices/88213 (own invoice)
API -> Attacker : 200, invoice data
Attacker > API : GET /invoices/88214 (someone else's)
API > DB : SELECT * FROM invoices WHERE id = 88214
DB -> API : row for account 912
API -> Attacker : 200, another customer's invoice
Attacker > Attacker : loop IDs 1 to 100,000
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a BOLA bug, found and fixed**

`GET /api/invoices/88213` returns the invoice to any logged-in user. An attacker loops over IDs 1 to 100,000 at 20 requests per second and downloads every customer's invoices in under 90 minutes. Login worked perfectly; authorisation was missing.

Fix: the query becomes `SELECT … WHERE id = $1 AND account_id = $caller_account` (taken from the session, never from the request). It returns 404, not 403, for other people's IDs, so the attacker cannot even learn which IDs exist.

A regression test checks that a second account gets 404 on the first account's invoices. A per-user rate limit and an alert on "more than 200 distinct invoice IDs per user per hour" catch the next one. Random, unguessable IDs (UUIDs) slow enumeration down, but they are a second line of defence, never the only one: IDs leak through URLs, emails and logs.

</div>

## Defence in layers

No single control stops everything, so the recurring defences sit at different layers. Each layer stops a different class of attack, and each assumes the one above might fail.

~~~dg-layers Defence in depth for a public API, from the network edge down to the data.
Edge | TLS, DDoS scrubbing, web application firewall, per-IP rate limits
Gateway | authentication, per-user and per-key quotas, size limits, timeouts
*Service | authorisation on every object, input validation, field allow-lists
Data | row-level security, least-privilege credentials, encryption at rest
Detection | logs of auth failures and denials, anomaly alerts, audit trail
~~~

- **Authorisation on every object access.** Check `order.owner_id == caller` in the data layer, not just at login. Centralise it (a policy library or service) so it is not reimplemented, slightly differently, in 40 handlers.
- **Validate and limit input.** Use parameterised queries (never string-concatenate SQL), allow-lists of fields that can be updated, and limits on body size, page size and query complexity (GraphQL depth limits).
- **Rate limits and quotas** (lesson 14.11) per user, per API key and per IP, plus stricter ones on sensitive flows: login, signup, password reset, checkout, sending messages.
- **Bot and abuse defences** for business flows: challenges (CAPTCHA or proof-of-work) only when signals look suspicious, device fingerprinting, email or phone verification, and **velocity checks** ("5 accounts from this device today").
- **Least privilege.** Each service's database user can touch only its own tables; each machine role can read only its own secrets. A compromised service then exposes little.
- **Secrets and dependencies**: secrets in a vault with rotation, never in code or images; dependency scanning; signed builds and pinned versions.
- **SSRF protection**: when the server fetches user-supplied URLs (webhooks, link previews), resolve the host and block private and metadata address ranges, and fetch from an isolated network.
- **Logging and alerts** on authentication failures, permission denials and unusual volumes, so an attack is noticed while it is happening.

## Case: a public signup endpoint

A signup form is open to anyone on the internet, so it attracts abuse. Think about who benefits from abusing it: spammers want many accounts, fraudsters want free-tier credits, attackers want to know who your users are, and some want to make you pay for SMS messages.

~~~dg-matrix Abuse vectors on a public signup endpoint and which defences help against each.
Abuse | Rate limit by IP | Rate limit by target | CAPTCHA on suspicion | Verify email/phone | Uniform response
Fake account bots | yes | partial | yes | yes | no
Account enumeration | partial | yes | partial | no | yes
SMS or email pumping | yes | yes, per number | yes | no, it is the attack | no
Credential stuffing | yes | yes, per account | yes | no | partial
Resource exhaustion | yes | no | partial | no | no
~~~

| Abuse | What it costs you | Defence |
| --- | --- | --- |
| Fake account creation by bots | spam, fake reviews, free-tier abuse, polluted metrics | per-IP and per-device rate limits, CAPTCHA when signals look automated, email or phone verification before the account can act |
| Account enumeration ("email already registered") | attackers learn who has accounts, then target them with phishing or stuffing | the same response and timing whether or not the email exists; send the "you already have an account" message by email instead |
| SMS or email pumping | attackers trigger verification messages to premium-rate numbers or spam lists; you pay per message | rate limit per phone number, per number prefix and per IP; block high-risk country prefixes you do not serve; cap daily spend and alert on it |
| Credential stuffing via the linked login | reused leaked passwords take over accounts | rate limits per account and per IP, breached-password checks, MFA |
| Resource exhaustion | huge payloads or slow requests tie up servers | size limits and timeouts at the gateway |

<div class="ex">
<div class="lab">Worked example</div>

**an SMS pumping attack, priced**

Signup sends a 6-digit code by SMS. An attacker scripts 40,000 signups overnight using phone numbers in a range they share revenue on, at premium-rate destinations costing about \$0.08 per message.

Cost by morning: 40,000 × \$0.08 = \$3,200, and the attacker keeps a share. None of these accounts ever verifies.

Defences, in order of payoff: a cap of 3 codes per phone number per day and 20 per number prefix per hour; a global spend alert at 3× normal hourly SMS volume; blocking country prefixes with no customers; a CAPTCHA before sending a code when the IP or device is new; and monitoring the ratio of codes sent to codes verified (normal is 80%; under attack it falls to 2%).

</div>

### Uniform responses and timing

Account enumeration works because the system behaves differently for existing and new emails. Fixing the message is not enough if the response time also differs: checking a password hash for an existing user might take 200 ms (bcrypt is deliberately slow) while a missing user returns in 5 ms. Attackers can measure that. Make both paths do similar work, or do the slow work asynchronously and return immediately in both cases.

## Common mistakes

- **Authorisation only at the gateway or UI.** Hidden buttons do not stop direct API calls.
- **Trusting IDs, roles or tenant IDs from the request body** instead of from the authenticated session.
- **Mass assignment**: binding the whole JSON body to the database model, so a user sets `role=admin`.
- **No limits on cost-generating flows** such as SMS, email, file processing or AI calls.
- **Secrets in code, images or logs**, discovered later in a public repository.
- **Detailed error messages** (stack traces, SQL) returned to clients.
- **No logging of denials**, so a slow enumeration attack runs for weeks unnoticed.

## In an interview

After the main design, spend a minute on security and abuse: where authentication happens (gateway, JWT), where authorisation is enforced (every object access, in the service or the database), which flows need rate limits and bot defences, and what you log. If the prompt involves a public form, name two or three abuse vectors and a defence for each. It shows you have run a system on the open internet.

<div class="take">
<div class="lab">Key takeaway</div>

most real attacks exploit missing authorisation checks, weak limits and misconfiguration, not clever cryptography. Check permission on every object, validate and limit every input, rate limit sensitive and cost-generating business flows, give uniform responses that reveal nothing, and log and alert on suspicious patterns.

</div>
