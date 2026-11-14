---
id: "24.08"
course: "inference-eng-course"
module: "24"
slug: "api-network-tenant-model-security"
title: "API, Network, Tenant & Model Security"
keyIdeas: ["Authenticate and rate-limit every request; set per-tenant token quotas.", "Isolate tenants' caches (cache salts or separate pools) to prevent leakage and timing side channels.", "Keep prompts and outputs out of logs by default; encrypt at rest and in transit.", "Protect weights: access control, integrity checks, no public engine endpoints.", "Validate inputs: max lengths, allowed parameters, safe structured-output grammars."]
question: "List four security concerns specific to multi-tenant LLM inference."
answer: "Cross-tenant data leakage through shared prefix caches (timing attacks revealing whether another tenant's prompt is cached, or bugs reusing KV across tenants); prompt and output data exposure in logs or traces; abuse and resource exhaustion (huge prompts, max tokens) as denial of service; and model weight theft or tampering. Also prompt injection affecting tool-using agents."
sources: [{"title": "OWASP Top 10 for LLM Applications", "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/", "note": "Common LLM risks."}, {"title": "vLLM: Security", "url": "https://docs.vllm.ai/en/latest/usage/security.html", "note": "Deployment security guidance."}]
tags: ["security"]
---

## Threats and controls

| Threat | Control |
| --- | --- |
| Unauthenticated access | API keys or OAuth at the gateway; engines on a private network |
| Resource abuse | rate limits, maximum prompt and output tokens, quotas |
| Cross-tenant cache leakage | per-tenant cache salt; separate pools for sensitive tenants |
| Sensitive data in logs | log ids and counts, not content; limit retention |
| Weight theft | storage access control, encrypted volumes |
| Weight tampering | checksums, signed artifacts |
| Prompt injection through tools | sandbox tools, least privilege |
| Engine vulnerabilities | patch regularly; never expose engine ports publicly |

## Prefix cache timing

If two tenants share a prefix cache, one tenant can send a guessed prompt and measure TTFT. A fast TTFT means the prefix was already cached, so someone else sent it recently.

<div class="ex">
<div class="lab">Worked example</div>

**what the timing leaks**

An attacker suspects another tenant uses a system prompt that starts with a certain company's name and a confidential product code.

They send that guess as a prompt. Cached prefill of 2,000 tokens returns the first token in 40 ms. Uncached, it takes 150 ms.

The difference is easy to measure. Salting the cache hash with the tenant id means different tenants never share cache entries, so the timing reveals nothing.

</div>

## Network

Engine endpoints should never face the internet directly. Put them behind a gateway that handles authentication, limits and input validation. Many engines' HTTP servers are not built to resist hostile traffic.

## Deep dive: how much the prefix cache timing leaks

A cached prefix makes TTFT shorter. An attacker who can send prompts and time responses compares the two distributions. With hit and miss TTFT means differing by ⟦m1085⟧ and noise ⟦m1086⟧, the separation is

⟦M1084⟧

<div class="ex">
<div class="lab">Worked example</div>

**guessing whether someone else sent a document**

A 4,000-token document: miss TTFT about 140 ms, hit about 30 ms, jitter ⟦m1087⟧ ms. ⟦m1088⟧, so a single probe guesses correctly about 99.7% of the time.

Salting the cache key per tenant (or per user for sensitive data) removes the signal entirely: another tenant's prefix can never be a hit.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

secure the gateway, isolate tenants (including their caches), keep content out of logs, and protect weights. Engines belong on private networks.

</div>
