---
id: "01.10"
course: "system-design-101"
module: "01"
slug: "repository-pattern"
title: "Repository pattern"
keyIdeas: ["The repository pattern puts data access behind an interface that looks like an in-memory collection: `add(order)`, `get(id)`, `list_by_customer(id)`.", "Business logic depends on the interface, not on SQL or an ORM. Storage details stay in one place.", "In tests you swap the real repository for a fake in-memory one, so domain tests run fast without a database.", "It also makes storage migrations easier: change the implementation, keep the interface.", "Don't over-abstract. If the repository just mirrors every ORM method, it adds code without isolating anything."]
question: "What does a repository let you swap out in tests?"
answer: "You can swap the real database-backed repository for an in-memory fake (for example one wrapping a set or a dictionary) that implements the same small interface. Business logic tests then run in microseconds without a database, network, fixtures or cleanup, and a few separate integration tests check the real SQL adapter."
sources: [{"title": "Cosmic Python: Repository Pattern", "url": "https://www.cosmicpython.com/book/chapter_02_repository.html", "note": "Free book chapter with code."}, {"title": "Martin Fowler: Repository", "url": "https://martinfowler.com/eaaCatalog/repository.html", "note": "The short catalog definition."}]
---

## The problem it solves

In many applications, business logic is tangled with database code. SQL strings or ORM queries are scattered through request handlers, services and even templates. A function that decides whether an order can be allocated to a warehouse batch also builds a query, opens a session and commits a transaction.

That causes three problems:

- **Testing is slow and fragile.** Every test of a business rule needs a real database with the right tables and rows. A suite of 2,000 such tests can take many minutes, and it breaks when someone changes a fixture.
- **The domain is tied to the storage technology.** Model classes inherit from ORM base classes, so the business rules depend on how tables are laid out, and changing storage means touching business code.
- **Query logic is duplicated.** The same "find active orders for this customer" query appears in five places, each slightly different.

The **repository pattern** is a small, focused fix for all three. It is not specific to any language or framework; it shows up in Java (Spring Data repositories), C# (.NET), Python, Go and elsewhere.

## The pattern

A **repository** sits between the domain logic and the data-mapping layer, and behaves "like an in-memory collection of domain objects". Code adds objects, removes them and looks them up by ID as if they were in a list or a dictionary. The repository translates those operations into database work behind the scenes.

The benefits:

- **Query logic lives in one place.** All the SQL for orders lives in the order repository.
- **The domain stays free of persistence code.** Business classes are plain objects with behaviour, with no knowledge of tables, sessions or connections.
- **Dependencies point one way.** The business logic depends on an abstract repository interface; the database code depends on the business objects, not the other way round.

### The vocabulary: ports and adapters

- A **port** is an interface your core code defines and depends on: "something I can add orders to and get orders from".
- An **adapter** is a concrete implementation of a port for one technology: a Postgres adapter, a DynamoDB adapter, an in-memory adapter for tests.

This is the idea behind **onion**, **hexagonal** and **ports-and-adapters** architectures. They differ in diagrams and naming, but share one rule: **dependencies point inward**, toward the domain. The domain knows nothing about the outer layers.

~~~dg-layers Dependencies point inward. The domain at the centre knows nothing about the database; adapters on the outside depend on the interfaces the inner layers define.
Adapters | SQL repository, HTTP handlers, message consumers
Service layer | use cases: allocate order, cancel order
*Repository interface (port) | add, get, list: defined by the core
Domain model | plain classes and business rules, no imports from outer layers
~~~

## Building one, step by step

The example domain: a warehouse allocates order lines to batches of stock.

### 1. Invert the dependency

In a typical ORM setup, model classes inherit from the ORM's base class, so the domain imports the database library. Instead, keep the domain model as plain classes and define the table mapping separately, so that the ORM layer imports the model and not the reverse. In SQLAlchemy this is "imperative" (classical) mapping: you declare tables, then tell the ORM to map them onto your plain classes.

### 2. Define a tiny interface (the port)

The interface should be as small as the domain needs. Often two methods are enough:

- `add(batch)`: store a new object.
- `get(reference)`: fetch one by its ID.

```
class AbstractRepository(abc.ABC):
    @abc.abstractmethod
    def add(self, batch: Batch) -> None: ...

    @abc.abstractmethod
    def get(self, reference: str) -> Batch: ...
```

Add methods only when a use case needs them (`list_for_sku(sku)`), and name them in the language of the domain, not of SQL.

### 3. Implement adapters

The production adapter wraps a database session:

```
class SqlAlchemyRepository(AbstractRepository):
    def __init__(self, session):
        self.session = session

    def add(self, batch):
        self.session.add(batch)

    def get(self, reference):
        return self.session.query(Batch).filter_by(reference=reference).one()
```

Committing is left to the caller, because one use case may touch several repositories and should commit them together. That job is usually given to a **Unit of Work** object, which owns the transaction and exposes the repositories.

The test adapter wraps an ordinary set:

```
class FakeRepository(AbstractRepository):
    def __init__(self, batches):
        self._batches = set(batches)

    def add(self, batch):
        self._batches.add(batch)

    def get(self, reference):
        return next(b for b in self._batches if b.reference == reference)
```

Each method is one line. A useful heuristic: **if a fake is hard to write, your abstraction is too complicated.**

### 4. Use the fake in tests

Service-layer tests build a `FakeRepository` holding a few objects and run real business logic against it, with no database, no network, no fixtures and no cleanup. They are fast, focused on behaviour, and don't break when the schema changes. A smaller set of integration tests then checks that the SQL adapter really does store and load objects correctly.

~~~dg-compare Swap the adapter, keep the domain. The service code is identical in both columns; only the object passed in changes.
[Production]
Service -> Repository port -> SqlAlchemyRepository -> Postgres
+ Real persistence and transactions
- Each test needs a database
[*Tests]
Service -> Repository port -> FakeRepository -> Python set
+ Microseconds per test
+ No fixtures or cleanup
- Doesn't test SQL: keep a few integration tests
~~~

~~~dg-seq One use case end to end. The service sees only the repository and the unit of work; it never touches SQL.
Handler > Service : allocate(order line)
Service > Unit of work : begin
Service > Repository : get batches for SKU
Repository -> Service : 3 Batch objects
Service > Service : domain picks earliest batch
Service > Unit of work : commit
Unit of work > Database : one transaction
Service -> Handler : batch ref
~~~

<div class="ex">
<div class="lab">Worked example</div>

**what the fake buys you in test time**

A service has 1,500 tests of business rules. Against a real database, each test sets up rows, runs, and rolls back, about 40 ms each: 1,500 × 40 ms = 60 s, and more on a busy CI machine.

With a fake repository, each test runs in about 0.2 ms: 1,500 × 0.2 ms = 0.3 s. Developers can run the whole suite on every save.

The team keeps about 30 integration tests against a real Postgres in Docker that check the adapter's queries and mapping: 30 × 100 ms = 3 s. Total feedback time drops from a minute to a few seconds, and failures point straight at the business rule that broke.

</div>

## What else a repository lets you swap

Tests are the most common reason, but the same seam helps elsewhere:

- **A different database.** Moving orders from Postgres to DynamoDB means writing one new adapter and changing one line of wiring. The business rules don't change. (In practice, a move like this also changes which queries are efficient, so the interface may need adjusting, but the change is contained.)
- **A cache layer.** A caching repository can wrap the real one: check Redis, fall back to the inner repository on a miss.
- **A remote service.** A repository can fetch data from another service's API rather than a table; the domain code doesn't care.
- **Read models.** You can have a separate, query-optimised repository (or a plain SQL view) for screens and reports, while writes go through the domain repository.

<div class="ex">
<div class="lab">Worked example</div>

**wrapping a repository with a cache**

Product lookups hit Postgres 8,000 times a second, mostly for the same 2,000 products. The team writes `CachedProductRepository(inner, redis)`. Its `get(id)` checks Redis first and, on a miss, calls `inner.get(id)` and stores the result with a 60-second time to live.

Nothing in the service layer changes; only the wiring at startup does. If the cache serves 95% of reads, Postgres sees about 400 lookups a second instead of 8,000. Tests still use the plain fake and don't need Redis.

</div>

## Trade-offs

**Pros:**

- A simple boundary between storage and domain, which keeps both easier to change.
- Easy fakes and fast unit tests.
- You can model the business problem before thinking about tables.
- Full control over the schema, since the ORM maps onto your classes rather than shaping them.
- Swapping or wrapping storage later is easier.

**Cons:**

- An ORM already gives some decoupling, so a repository on top of an ORM can feel like an abstraction over an abstraction.
- Hand-maintained mappings are extra code that has to stay in sync with the tables.
- Every extra layer adds maintenance cost and a "why is this here?" moment for new developers.
- It can hide expensive queries. `for id in ids: repo.get(id)` looks innocent and is the **N+1 query problem**: 1 query becomes 100. Add batch methods like `get_many(ids)` when you need them.

~~~dg-tree Where the pieces live in a small service using the pattern. Only the adapters folder imports the database library.
allocation service
  domain | model.py: Batch, OrderLine, allocate()
  service_layer | services.py, unit_of_work.py
  adapters
    *repository.py | AbstractRepository, SqlAlchemyRepository
    orm.py | table definitions and mapping
  entrypoints | flask_app.py, message handlers
  tests
    unit | uses FakeRepository
    integration | real Postgres
~~~

## When not to use it

A simple CRUD app (forms that save rows and pages that show them) doesn't need a domain model or a repository. If the business logic is "validate input and save it", the extra layer is pure cost; use the ORM or plain SQL directly and test against a real database. The pattern pays off as the business logic grows complex: pricing rules, allocation, workflows with many states, anything you want to test thoroughly and quickly.

## Common mistakes

- **Leaking the ORM through the interface**, for example returning query objects that callers keep filtering. Then the domain depends on the ORM again.
- **Generic repositories with dozens of methods** (`find_by_x_and_y_order_by_z`). Keep the interface small and domain-shaped.
- **Committing inside the repository**, which stops a use case from combining several changes in one transaction.
- **Never testing the real adapter.** Fakes test logic; a few integration tests must still test SQL.

## In an interview

This comes up more in design discussions of code structure than in large-scale system design. If asked, explain the dependency direction, the small interface, and the two adapters (real and fake), and give the testing benefit as the main reason. Mention that it also isolates storage choices, so changing databases later is contained to one layer.

<div class="take">
<div class="lab">Key takeaway</div>

put storage behind a small, collection-like interface that the domain defines, so business logic depends on the interface rather than the database. That lets you swap in an in-memory fake for fast tests, or a different or cached store in production. Use it when domain logic is substantial; skip it for thin CRUD apps.

</div>
