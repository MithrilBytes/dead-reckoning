# Dead Reckoning

Dead Reckoning is an agent runtime for places where the things an agent depends on
stop answering. Model endpoints, tool backends, the identity provider, the clock
source and other nodes can all become unreachable, slow or untrustworthy without
warning, and the agent still has to produce decisions someone can act on and
account for afterwards. It keeps working by changing what it does rather than by
retrying what it was doing: the model is told each turn what it can no longer
reach, tools degrade according to contracts they declare in advance, side effects
that cannot happen yet are queued together with the conditions that justified
them, and every decision records the model tier and operating mode that produced
it. When the connection returns, queued actions are re-checked against the world
as it now is, decisions made on smaller models are re-examined by larger ones, and
two nodes that reached opposite conclusions while isolated produce a conflict for
a person rather than a race for the last write.

The name is the navigation technique. Without external fixes you steer on known
heading, speed and elapsed time, and correct when a fix returns.

## The scenario

A storm takes out a substation. Repair crews drive out with a node each running on
a laptop. The cell towers they reach headquarters through are on grid power, so as
the outage spreads the trucks lose connectivity because of the very outage they
were sent to fix.

```
make demo
```

Runs end to end in about a second. No API key, no model download, no network:
the models are scripted, the utility's systems are in process, and the hub runs on
loopback.

```
00:00  Two trucks, connected, triaging at full fidelity.
00:15  The SCADA link drops. A sensor ticket becomes an abstention with a
       conditional answer; a water pumping station is still called P1, because
       that answer never needed the sensor.
00:30  Everything drops. A local model takes over, decisions are stamped rank 2,
       priority changes are queued, and a crew dispatch waits for a person.
00:45  The other truck, also dark, has a field report the first one does not:
       same ticket, opposite call.
01:00  The link returns. The dispatch does not fire, because headquarters got
       there first and the precondition no longer holds. The disagreement
       surfaces as a conflict. A supervisor decides, and every chain verifies.
```

`make record` produces an asciinema recording of that run, if asciinema is
installed. The repository does not carry one yet.

## Why the network layer is not enough

Durable execution frameworks checkpoint a workflow and resume it after a crash or
a timeout. They are good at that, and they resume the same plan. If the plan
called for a value that is no longer obtainable, resuming means asking for it
again and eventually failing, or worse, proceeding without it. Nothing in a
checkpoint says that a missing input should change what the agent concludes.

Model routers fall back to another endpoint when one returns an error. That keeps
a model answering, which is not the same as keeping the answers honest. A router
does not know that the tool the model wanted is also gone, does not stamp which
model actually answered onto the decision, and has no notion of going back to
re-examine an answer once a better model is reachable again.

Local-first sync engines converge data across disconnected replicas, usually with
last write wins or a field merge. That is correct for data and wrong for
decisions. When two nodes disagree it is normally because they knew different
things, and silently keeping the later one discards the evidence along with the
answer. A disagreement between two decisions is a question, and the useful thing
a runtime can do with a question is put it in front of a person.

## How it compares

Based on public documentation as of September 2026. Where a capability is marked
"not observed", that means it was not visible in the project's public materials,
not that the project lacks it. Corrections are welcome as issues.

Legend: present, partial, not observed, or not applicable to that project.

| Capability | Offline agent runtimes | Durable execution | Local-first sync | Model routers | Identity continuity | Dead Reckoning |
|---|---|---|---|---|---|---|
| Offline-first operation | yes | no | yes | no | identity only | yes |
| Tiered models with local fallback | yes | no | n/a | error driven | n/a | yes |
| Connectivity state machine with hysteresis | partial | no | no | no | partial | yes |
| Per-dependency failure classification | not observed | no | no | partial | no | yes |
| Capability manifest injected into model context | not observed | no | no | no | no | yes |
| Runtime-enforced abstention when inputs are gone | safety rules only | no | no | no | no | yes |
| Typed offline tool contracts | not observed | no | no | no | no | yes |
| Durable outbox for deferred side effects | not observed | activities | yes | no | no | yes |
| Preconditions captured at deferral, rechecked at execution | not observed | no | no | no | no | yes |
| Consequence-based human approval | yes | interrupts | no | no | yes | yes |
| Identity continuity with authority reduction | not observed | no | no | no | yes | partial |
| Time-trust tracking, TLS errors read as clock skew | not observed | no | no | no | no | yes |
| Model tier and mode stamped on every decision | not observed | no | no | no | no | yes |
| Re-review of low-fidelity decisions on reconnect | not observed | no | no | no | no | yes |
| Multi-node log merge | not observed | no | data only | no | no | yes |
| Decision conflicts go to adjudication, never last write wins | not observed | no | no | no | no | yes |
| Append-only hash-chained records | not observed | no | no | no | audit trail | yes |
| Deterministic replay | partial | partial | no | no | no | scripted |
| Fault injection recorded as injected | not observed | no | no | no | sandbox | yes |
| Physical actuation, hardware I/O | yes | no | no | no | no | not a goal |
| Production hardening, multi-tenancy | partial | yes | partial | yes | yes | not a goal |

Dead Reckoning sits above the network layer. Circuit breakers and retries keep a
call alive. Sync engines keep data converging. Routers keep a model answering.
Each assumes the plan was still the right plan. This is for the case where it was
not: where a missing input should produce an abstention rather than an answer,
where an action should wait for its justification to be rechecked rather than
fire, where a decision made on a small model in the dark should be looked at again
by a large one in the light, and where two isolated nodes reaching opposite
conclusions should be a conflict for a human rather than a race for the last
write.

## The concepts

**Mode** is the node's assessment of itself, derived from the health of everything
it depends on: connected, degraded, islanded, or reconnecting. Falling is
immediate; rising waits, so a flapping link does not rewrite the mode every few
seconds.

**Manifest** is what the model is told it can do, rebuilt before every call and
hashed onto every decision made under it. It lists unavailable tools too, with the
reason, because a model that cannot see a capability exists will route around its
absence silently.

**Contract** is what a tool declares about itself when its backend is gone: answer
from a local copy, record the intent for later, or refuse. The runtime dispatches
by contract, so the model cannot talk its way past it.

**Outbox** holds side effects that could not happen yet, each with the conditions
that justified it. Before one executes, those conditions are checked again.

**Tier stamp** records which model answered and under what conditions, so a
decision can be found later and weighed.

**Re-review** re-examines flagged decisions once a better model is reachable, with
evidence merged from every node rather than only what the deciding node had.

**Conflict** is two decisions about the same thing that cannot both stand, made
without either node having seen the other. It is surfaced, never resolved
automatically.

## Quickstart

```
make install
make demo
make check
```

Then, for a single node:

```
cp dr.example.toml dr.toml
./.venv/bin/dr init
./.venv/bin/dr status
./.venv/bin/dr chaos frontier --fail CONNECT_TIMEOUT
./.venv/bin/dr status
./.venv/bin/dr verify
```

Python 3.11 or newer. Runtime dependencies are httpx, pydantic, typer and rich.

## Design

`DESIGN.md` covers the mechanisms in more detail: the failure taxonomy, mode
derivation, the dispatch matrix, the outbox lifecycle, and how conflicts are
detected and adjudicated. `schemas/` holds the wire formats, which are what nodes
validate each other's records against. `docs/prompts/planner_v1.md` is the planner
system prompt, versioned because its hash is recorded on every decision it
produces.

## Status and non-goals

This is version 0.1 and a reference implementation. It is meant to be read.

Not in scope: a general-purpose agent framework, physical actuation, hardware
power sensing, cryptographic signatures (records are hash chained and a signer
hook exists, unused), multi-tenancy, production hardening of the hub, or adapters
for other agent frameworks. Conflict handling is deliberately narrow: set union
for immutable records, and explicit conflict records for anything a person should
decide.

Known gaps are tracked as issues rather than hidden. The live path against a real
model endpoint has not been exercised end to end; the scenario and the test suite
run against scripted models, which is what makes them deterministic and what means
they do not tell you how a particular model behaves.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
