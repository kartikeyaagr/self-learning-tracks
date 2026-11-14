---
id: "02.01"
course: "system-design-101"
module: "02"
slug: "api-design-contracts"
title: "API design contracts"
keyIdeas: ["An API is a contract. Once clients depend on it, changing it is expensive, so design it deliberately.", "REST style: nouns as resources (`/urls`, `/urls/{id}`), HTTP verbs for actions, status codes for outcomes (201 created, 404 not found, 409 conflict, 429 rate limited).", "Paginate lists. Cursor-based pagination (`?cursor=abc&limit=50`) is stable under inserts; offset pagination gets slow and skips items.", "Version from day one (`/v1/` or a header). Add fields freely; never remove or repurpose them without a new version.", "Return consistent error bodies with a machine-readable code and a human message.", "Make unsafe operations idempotent where possible (idempotency keys on POST)."]
question: "Design the endpoints for creating, reading and listing short URLs. How do you paginate the list?"
answer: "POST /v1/urls {long_url, custom_alias?, expires_at?} creates a link and returns 201 with the code (409 if the alias is taken). GET /{code} redirects with 302 to the long URL, or 404. GET /v1/urls?page_size=20&page_token=... lists the caller's own links (owner taken from the auth token), newest first, and returns next_page_token. Use cursor pagination keyed on (created_at, code), encoded as an opaque token, because each page is an index seek that stays fast at any depth and does not skip or repeat rows when links are created or deleted while paging."
sources: [{"title": "Hello Interview: API Design", "url": "https://www.hellointerview.com/learn/system-design/core-concepts/api-design", "note": "Resources, verbs, pagination, versioning for interviews."}, {"title": "Google API Improvement Proposals", "url": "https://google.aip.dev/general", "note": "Resource-oriented design rules Google uses for its APIs."}, {"title": "Microsoft REST API Guidelines", "url": "https://github.com/microsoft/api-guidelines", "note": "Errors, versioning, long-running operations."}]
---

## An API is a contract

An **API** (application programming interface) is the promise a service makes to its callers: these are the resources, these are the operations, this is what you send, this is what comes back, and this is how errors look. Once other teams, mobile apps or paying customers depend on it, changing it is expensive. You cannot force every phone in the world to update its app on the day you rename a field.

That is why interviewers ask you to write the API early. It fixes the scope (every functional requirement should map to an endpoint), it shows you think about callers and not just servers, and it becomes the contract every component behind it must fulfil.

Most interview APIs, and most public APIs, are **REST over HTTP**. Lesson 02.03 covers HTTP and alternatives like gRPC; this lesson is about designing the contract well.

## Model resources, not actions

Model the API around **resources** (nouns) and use HTTP methods for the actions. You write `POST /urls`, not `POST /createShortUrl`. Resources are usually plural, and they nest when one belongs to another: `/users/{id}/posts`.

Every resource should have one **canonical name**: a path that identifies exactly one thing, such as `publishers/123/books/456`. Callers can store it, pass it around and use it in other calls. Keep nesting shallow (two levels is usually enough), because deep paths make URLs brittle when relationships change.

~~~dg-tree A resource hierarchy. Each node is a collection or an item with one canonical path, and the methods attach to these paths.
/v1
  /users | collection: GET lists, POST creates
    /users/{user_id} | item: GET, PATCH, DELETE
      /users/{user_id}/urls | that user's links
  /urls | collection of short links
    */urls/{code} | one link
      /urls/{code}:disable | custom action
  /{code} | public redirect, outside /v1
~~~

Some operations are not natural creates, reads, updates or deletes: "cancel an order", "publish a draft", "resend a verification email". You have two reasonable options. Model them as a change of state (`PATCH /orders/123 {status: "cancelled"}`), or use a **custom method** written with a colon, `POST /orders/123:cancel`, which is the convention in Google's public API standards. Either is fine; what matters is consistency across the API.

## Methods and what they promise

Each HTTP method carries a meaning that clients, proxies and caches rely on. Two properties matter most:

- **Safe**: the request does not change anything on the server. Caches and crawlers may call safe methods freely.
- **Idempotent**: calling it twice has the same effect on the server as calling it once.

~~~dg-matrix What each HTTP method promises. Idempotent methods can be retried after a timeout without fear; POST cannot, without extra help.
Method | Safe | Idempotent | Cacheable | Typical use
GET | yes | yes | yes | read a resource or a list
POST | no | no | partial, rarely | create, or trigger an action
PUT | no | yes | no | replace a resource at a known ID
PATCH | no | partial, not guaranteed | no | change some fields
DELETE | no | yes | no | remove a resource
~~~

Idempotency matters because networks fail and clients retry. A retried `PUT /users/42 {name: "Ana"}` is harmless: the result is the same. A retried `DELETE` is harmless: the second call finds nothing to delete (and may return 404, but the server state is the same). A retried `POST /orders` can create two orders. Lesson 02.09 shows how **idempotency keys** make POST safe to retry: the client sends a unique key in a header, and the server returns the stored result if it sees the same key again.

PATCH is idempotent if it sets fields to values (`{status: "shipped"}`), but not if it applies a relative change (`{balance: "+10"}`). Prefer absolute values.

## Where inputs go

- **Path** identifies the resource: `/urls/abc123`.
- **Query string** filters, sorts and paginates: `?limit=20&cursor=...&sort=-created_at`.
- **Body** carries the data you are creating or changing, usually JSON.
- **Headers** carry metadata: the auth token (`Authorization: Bearer ...`), content type, idempotency keys, conditional-request tags.

**Never take the caller's identity from the request body.** The server should derive "who is calling" from the verified auth token. If `POST /v1/urls` accepted `{owner_id: 42}`, any client could create links as any user. The same applies to `GET /v1/urls?user_id=42`: the server must check the caller is allowed to see user 42's links, or better, offer `GET /v1/me/urls` and read the user from the token.

## Status codes and errors

Use status codes as intended, so clients can react without parsing messages:

| Code | Meaning | Client should |
| --- | --- | --- |
| 200 OK | success with a body | use the body |
| 201 Created | resource created | read the Location header or returned ID |
| 204 No Content | success, nothing to return | carry on |
| 400 Bad Request | malformed or invalid input | fix the request; retrying won't help |
| 401 Unauthorized | not authenticated | log in or refresh the token |
| 403 Forbidden | authenticated but not allowed | stop |
| 404 Not Found | no such resource (or hidden from you) | stop |
| 409 Conflict | clashes with current state, e.g. alias taken | change the input |
| 412 Precondition Failed | the resource changed since you read it | re-read and retry |
| 429 Too Many Requests | rate limited | wait for Retry-After, then retry |
| 500 Internal Server Error | server bug | maybe retry, report |
| 503 Service Unavailable | temporarily overloaded or down | retry with backoff |

The split between 4xx (the caller's fault, don't retry unchanged) and 5xx (the server's fault, retrying may help) is what lets client libraries decide automatically whether to retry.

Error bodies should have a stable, machine-readable code plus a human message, and details where useful:

```
{"error": {"code": "ALIAS_TAKEN", "message": "The alias 'sale' is already in use.", "field": "custom_alias"}}
```

Clients branch on `code`, never on the text of `message`, which you are free to reword.

### Concurrent updates

Two people edit the same resource; the second save silently overwrites the first. HTTP has a standard fix, **optimistic concurrency**: every GET returns an `ETag` header (a version tag). The client sends it back on update as `If-Match: "v7"`. If the resource has changed since, the server returns 412 and the client re-reads and retries instead of overwriting.

## Pagination: offset vs cursor

Every list endpoint must be paginated, because lists grow without bound. A user with 50,000 links must not receive them in one response.

**Offset pagination** (`?offset=40&limit=20`) is simple and lets users jump to page 5. It has two problems:

- **It gets slow on deep pages.** `OFFSET 20000 LIMIT 20` makes the database find and discard 20,000 rows before returning 20. Cost grows linearly with depth.
- **It is unstable under inserts and deletes.** If a new item is inserted at the top while someone is paging, everything shifts down by one, and they see the last item of page 1 again at the top of page 2. A deletion makes them skip an item.

**Cursor pagination** (`?page_token=<opaque>&page_size=20`) returns a token that encodes where the last page ended, usually the sort key of the last item. The next query is "items after this key": `WHERE (created_at, code) < (:last_created_at, :last_code) ORDER BY created_at DESC, code DESC LIMIT 20`. With an index on `(created_at, code)`, the database seeks straight to the right place, so every page costs the same at any depth, and inserts at the top don't shift later pages. You lose random page jumps, which most feeds and lists never need.

~~~dg-compare Offset versus cursor pagination for page 1,001 of a list sorted by creation time.
[Offset]
page 1001 -(OFFSET 20000 LIMIT 20)-> DB walks and discards 20,000 rows -> 20 rows
+ Simple, supports jump to page N
- Cost grows with depth
- Duplicates or gaps when rows change
[*Cursor]
next page -(WHERE key < last_key LIMIT 20)-> DB seeks via index -> 20 rows
+ Same cost at any depth
+ Stable while rows are inserted
- No jumping to page N
~~~

Rules for good cursors:

- **Sort by a unique key**, or add a tie-breaker. Sorting only by `created_at` breaks when two items share a timestamp; `(created_at, code)` is unique.
- **Make the token opaque**: base64-encode it (and ideally sign it), so clients can't construct or depend on it and you can change the encoding later.
- **Return `next_page_token`**, and an empty one when there are no more pages. Clients loop until it is empty.
- **Cap the page size** on the server (for example a default of 20, a maximum of 100), whatever the client asks for.
- **Add pagination from day one.** Adding it later breaks clients that assumed one response contained everything.

<div class="ex">
<div class="lab">Worked example</div>

**why offset breaks while paging**

A user lists their links, newest first, 20 per page, with offset pagination. They load page 1 (links 1 to 20).

While they read, they create a new link in another tab. Every existing link moves down one position.

They load page 2 with `offset=20`. Position 21 is now the old link 20, which they already saw. They see a duplicate, and if a link had been deleted instead, they would silently skip one.

With a cursor, page 2 is "links created before link 20's (created_at, code)". The new link is newer, so it does not affect the result: page 2 starts exactly at the old link 21.

</div>

## Consistency across the API

Google's public API standards (the AIPs) are a detailed, published style guide for resource APIs, and their main value is consistency. The ideas that come up most:

- **Standard methods**: Get, List, Create, Update, Delete, with predictable request and response shapes.
- **Custom methods** for actions that don't fit, with the colon syntax.
- **Resource names**: every resource has one canonical path.
- **Field names**: consistent `lower_snake_case`, with units in the name (`size_bytes`, `duration_seconds`, `expire_time`), so nobody guesses whether `timeout` is seconds or milliseconds.
- **List methods** take `page_size` and `page_token` and return `next_page_token`.

A caller who learns one endpoint can guess the rest. Inconsistency (some lists use `cursor`, others `page`; some timestamps are seconds, others ISO strings) is a tax on every client.

## Versioning, auth and rate limits

- **Versioning**: put a major version in the path (`/v1/`) or a header. **Additive changes** (new optional fields, new endpoints, new optional query parameters) need no new version, as long as clients ignore fields they don't recognise. **Breaking changes** (removing or renaming fields, changing types or meanings, making an optional field required) need a new major version, run alongside the old one until clients migrate.
- **Authentication**: **API keys** identify a calling application and suit server-to-server calls. User sessions usually use **tokens** such as JWTs (JSON Web Tokens), which carry signed claims (user ID, expiry, scopes) that any service can verify with a public key, without a database lookup. Short expiry times (minutes to an hour) limit the damage from a leaked token.
- **Rate limiting** protects the service from one noisy caller. Return 429 with a `Retry-After` header, and publish the limits so client developers can plan.

~~~dg-seq Creating a link, then reading the list. The server takes the owner from the token, and the list returns a cursor for the next page.
Client > API : POST /v1/urls {long_url} + Bearer token
API > API : verify token, owner = user 42
API -> Client : 201 {code: abc123}, Location: /v1/urls/abc123
Client > API : GET /v1/urls?page_size=20
API -> Client : 200 {urls: [20 items], next_page_token: "eyJj..."}
Client > API : GET /v1/urls?page_size=20&page_token=eyJj...
API -> Client : 200 {urls: [7 items], next_page_token: ""}
~~~

## Common mistakes

- **Verbs in URLs** and inconsistent naming across endpoints.
- **Returning unbounded lists.**
- **Trusting user IDs sent in the body or query** without checking them against the token.
- **Breaking changes without a version bump**, such as renaming a field.
- **Leaking internals**: auto-increment database IDs (which reveal volumes and invite enumeration) or table structures copied straight into responses.
- **Using 200 for errors** with `{"success": false}` in the body, which defeats every HTTP-aware client, proxy and monitor.

<div class="ex">
<div class="lab">Worked example</div>

**URL shortener API**

`POST /v1/urls` with body `{long_url, custom_alias?, expires_at?}` returns `201 {code, short_url}`. A duplicate alias returns `409 ALIAS_TAKEN`. The client may send an `Idempotency-Key` header so a retried create doesn't make two links.

`GET /{code}` returns `302` with a `Location` header pointing to the long URL, or `404` if unknown or expired. (302 rather than 301, so browsers don't cache the redirect forever and every click can be counted.)

`GET /v1/urls?page_size=20&page_token=...` returns `{urls: [...], next_page_token}` for the caller's own links, newest first, using a cursor on `(created_at, code)`.

`DELETE /v1/urls/{code}` returns `204`. The caller's identity always comes from the auth token, and deleting someone else's link returns 404 so codes can't be probed.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

model nouns with canonical names, use methods for what they promise (and make POST retry-safe with idempotency keys), derive identity from the auth token, return honest status codes with machine-readable errors, and paginate every list with an opaque cursor from day one.

</div>
