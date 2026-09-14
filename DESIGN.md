# Design

How Dead Reckoning works, in the order the pieces depend on each other. The wire
formats are in `schemas/`, and they are the authority: nodes validate each other's
records against them, so anything here that disagrees with a schema is this
document being out of date.

## Records

Everything a node does ends in a record. Records are append only, and the store
has no update path: the type exposes none, and the database aborts on `UPDATE` or
`DELETE`, so reaching past the class does not help. Each record carries the hash
of the previous record from the same node, giving one chain per node from a
genesis record to the present.

A record's hash covers its canonical JSON with the hash and signature removed.
Canonical means object keys sorted by code point, no insignificant whitespace,
UTF-8, and floats rendered by Python's `repr`, which is the shortest string that
round trips. Unset fields are omitted rather than written as null, because several
envelope fields have no null variant in the schema and because omission is the
only rule two implementations can follow identically without sharing a list of
which fields are nullable.

Secrets are redacted before hashing, so the digest covers what is stored rather
than what was collected. Records travel between nodes; a secret that reached the
store would leave with them.

## Time

Every record carries a hybrid logical clock stamp: a physical component tracking
real time, and a logical counter that breaks ties and preserves causality when the
physical component stalls or moves backwards. The clock is persisted, so it never
regresses across a restart, and a node whose operator sets the system clock back
to 1970 still issues strictly increasing stamps.

Separately, the node tracks how far it trusts its own clock. This exists for one
specific failure: a laptop that has been off the grid for hours comes back with a
drifted clock, the first TLS handshake fails with "certificate not yet valid", and
a system that reports that as a security failure sends someone chasing a
compromise that never happened. When the clock is not trusted, a certificate
validity error is classified as clock skew and the remediation text says so.

## Health

Each dependency is probed and observed, and every interaction is classified into
one of twelve failure classes. The granularity is the point: a name that will not
resolve, a port that refuses, a handshake that fails on dates, and a token that
has expired look alike to a retry loop and mean entirely different things. A
rejected token in particular is not an unreachable dependency: the service
answered and refused, so waiting will not help and identity has to refresh.

Each dependency has a circuit breaker. Observations from real calls feed it
exactly like active probes, so a call that has already failed does not wait for
the next probe to be believed. A dependency nobody has probed is `UNKNOWN`, which
is not the same as healthy, and the router will not select an unprobed tier.

## Modes

The operating mode is a pure function of the health vector, the previous mode, and
which dependencies have reconciliation work waiting. Pure means it reads no clock
and no database, so every combination of inputs can be enumerated in a test
instead of sampled.

Reconnecting is absorbing: a dependency returning mid-sequence does not knock the
node out of work it has already started. Islanded means no model off this machine
is reachable, which is the condition the runtime is named for. Connected requires
every dependency that gates capability to be usable, excluding peers, because a
node is not degraded merely for having a peer that is switched off.

Falling and rising are treated differently. Losing a dependency takes effect at
once, because an agent that believes it still has a capability it has lost will
fabricate. Regaining one waits out a dwell, because a link that returns for three
seconds has not really returned.

## The manifest

Before every model call the runtime builds a manifest: the mode, the tier
answering, identity state, time trust, budget, and every registered tool with its
current availability. It is rendered into the system prompt and its hash is
recorded on every decision made under it, so a decision can be tied to exactly
what the model was told rather than to a description of it.

Unavailable tools are listed, with reasons. Omitting them would be the obvious
economy and it is wrong: a model that cannot see a capability exists routes around
its absence silently, while one told the capability exists and is unreachable can
abstain and say why.

The builder takes the health vector and the evaluation instant as parameters
rather than reading them, so the same code renders the live manifest and answers
what the node could still do under a hypothetical.

## Tool contracts and dispatch

A contract declares what a tool does when its backend is gone: answer from a local
copy, record the intent for later, or refuse. It also declares its side effect
class, its consequence level, the preconditions a deferred call must satisfy, how
stale its local data may be, and how its idempotency key is derived.

Dispatch is designed as two stages. The first reads backend health and offline
policy alone. The second applies identity and approval, and can only tighten: it
may move a call from live to queued, or from queued to refused, never the reverse.
Approval is meant to be held by the outbox even against a healthy backend, so that
there is one hold mechanism and the hold is reviewable.

The second stage is not yet applied on the live dispatch path.
`ToolEnforcer.dispatch` reads only backend health and offline policy and never
calls `authority_for`, so a call whose backend is healthy runs live whatever the
identity state or the tool's approval setting. Identity and approval take effect
only for calls that were queued anyway: `authority_for` decides whether an outbox
entry needs approval when the entry is created, and is consulted again when the
outbox drains, where it can cancel the entry, leave it waiting, or hold it for
approval.

The enforcer decides the path, not the model. And if the model produces a final
answer that depends on a tool which returned unavailable, the runtime converts it
into an abstention, preserves what the model actually said, and records that it
had to. Asking a model nicely to abstain works until the moment it matters.

## The outbox

A side effect that cannot happen yet becomes an entry carrying the conditions that
justified it: the crew was free, the ticket was unassigned, and where each of
those was observed. Hours later, when the link returns, they are checked again
against the live backend. If one no longer holds, the action does not fire; it
becomes a question for the agent, carrying both the value observed then and the
value observed now.

That is the difference between deferring work and deferring a decision. Most
systems replay the action. This one re-examines whether the action is still
right.

Exactly once is guarded twice. An idempotency key rejects a duplicate intent
before anything is stored, and an entry found mid-execution after a crash is never
retried: the tool is asked whether its effect landed if it can answer, and the
entry fails loudly for a person if it cannot.

## Routing and re-review

Tiers are ranked, rank zero highest fidelity. The router takes the best available
tier that meets the task class's floor, and if nothing meets the floor it
escalates rather than quietly answering worse. Under a low battery it takes the
cheapest local tier that still meets the floor, which is the right trade on a
truck, and records that the choice was power constrained when that is the only
reason a review was needed.

Decisions taken below a task class's review threshold are flagged. When a better
tier becomes reachable they are re-examined, with evidence merged from every
node, not only what the deciding node had. The reviewer must be strictly better
than the tier under review: without that, a tier that survived the outage
qualifies as its own reviewer and the queue fills with reviews that agree by
construction.

The original record is never modified. A disagreement produces a conflict.

## Sync and conflicts

Nodes exchange records with a hub or directly with each other over four endpoints.
Merging is union by id, which is lossless and order independent because records
are immutable. Before accepting a batch the receiver recomputes each record's hash
and checks the chain joins up with what it already holds, and refuses the whole
batch if it does not: accepting half would leave a hole that every later
verification reports and nobody can explain.

Two decisions conflict when they concern the same subject and the same outcome
key, assert different values, and were made without either node having seen the
other. That last condition is what separates a conflict from a supersession: a
node that had already merged the other's decision knew, and chose otherwise.

Detection is a pure function over a set of records. Emitting a conflict record
changes the record set, so a detector that read its own output would not be
idempotent, and two nodes merging in different orders could disagree about what
happened. Detection reads decisions only; a separate filter skips what this node
has already recorded.

A conflict is not resolved automatically. A person chooses, and the resolution
states the outcome that now stands. Release and cancel are expressed as a
predicate over a single entry, so that the resolving node and every node that
later merges the record reach the same verdict without coordinating.

## The hub

The hub stores everything it is given and decides nothing. It never drains an
outbox, never resolves a conflict, and never reasons about a record it holds.
Compromising it yields a copy of the logs, which is a privacy problem, and no
ability to make a node do anything, which would be a safety one.

## Fault injection

Faults are injected at the transport boundary, so an injected failure travels the
same path as a real one: the client raises, the classifier names it, the breaker
counts it. Nothing downstream has a special case for it.

Every injection is recorded as an injection. A demonstration of a system that
survives outages is worth nothing if the audience cannot tell whether the outage
was real.
