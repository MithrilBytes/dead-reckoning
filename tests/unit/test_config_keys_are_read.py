# SPDX-License-Identifier: Apache-2.0
"""Every configuration key is read by the runtime, or names the reader it waits for.

The loader refuses unknown keys, so a typo cannot pass for a setting. The opposite
defect is quieter: a key the loader accepts and nothing acts on. An operator who
sets `read_timeout_s = 180` believes the node will wait three minutes for a slow
model, and nothing tells them it will not.

A key counts as read when code in src, demo or hub takes that attribute from a
value known to be the model that declares it. The check learns what a value is
from annotations, assignments, loops, comprehensions and return annotations, which
strict type checking already makes this code spell out. A receiver it cannot
follow is reported as unread, and the fix is an annotation, not a pending entry.

Inside config.py a validator does not count, because checking a value is not
acting on it. A method or property on a model counts for what it reads once code
elsewhere uses it, which is how `redaction.keys` reaches the record store through
`Config.redact_keys()`.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import typing
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from deadreckoning.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]

SETTINGS_FILE = REPO_ROOT / "src" / "deadreckoning" / "config.py"

READER_DIRS = ("src", "demo", "hub")

PENDING: dict[str, str] = {
    "node.sync_listen": "the node's own sync endpoint, which serves peers on this address",
    "tiers[].base_url": "the HTTP model client, as the address it sends requests to",
    "tiers[].api_key_env": "the HTTP model client, which reads the key when it makes a call",
    "tiers[].supports_tools": (
        "the HTTP model client, which asks for decisions as JSON text when this is false"
    ),
    "tiers[].connect_timeout_s": "the HTTP model client, as its connect timeout",
    "tiers[].read_timeout_s": "the HTTP model client, as its read timeout",
    "tiers[].max_tokens": "the HTTP model client, as the output limit on each request",
    "tiers[].seed": "the HTTP model client, on requests to providers that accept a seed",
    "tiers[].canary": "the canary scheduler, to decide whether it probes a tier",
    "dependencies[].token_env": (
        "the sync and identity provider clients, which send the token as a bearer credential"
    ),
    "dependencies[].canary": "the canary scheduler and preflight probes, as the request to send",
    "tools": "the tool loader, which layers these overrides over the contracts declared in code",
    "task_classes": "the task runner, which routes and budgets each task by its class",
    "health.canary_interval_s": "the canary scheduler, as the probe period for a closed breaker",
    "health.rate_limit_window_s": (
        "the HTTP failure classifier, as how long a 429 without Retry-After holds"
    ),
    "identity": "the identity tracker, once a node builds one from these settings",
    "sync.interval_s": "the periodic sync loop, as the time between rounds",
    "sync.page_size": "the sync client, as the number of records it asks for in one pull",
    "sync.reconcile_deadline_s": (
        "reconciliation, as the longest it waits on a peer that is still reconnecting"
    ),
    "provisioning.auto": "the provisioning schedule, which runs passes on a timer when this is set",
    "provisioning.interval_s": "the provisioning schedule, as the time between passes",
    "provisioning.horizon_s": "the forecast command, as its look ahead when none is given",
    "provisioning.max_rows_per_tool": "the provisioning command, when it builds a pass",
    "provisioning.max_wall_s": "the provisioning pass, which stops once it has run this long",
    "provisioning.max_calls_per_pass": "the provisioning command, when it builds a pass",
}
"""Keys whose reader is not written yet, each with that reader in plain words.

An entry covers the key it names and everything under it. It has to leave once
anything reads one of those keys, so no entry outlives its reader.
"""

# A reference is a class name, or a class name behind one prefix: `*T` is an
# iterable of T, `{T` a mapping to T, `~T` what a mapping's items() yields, and
# `^T` one of those pairs. Containers of containers are not followed.
SEQUENCES = frozenset(
    {
        "list",
        "tuple",
        "set",
        "frozenset",
        "Sequence",
        "Iterable",
        "Iterator",
        "Collection",
        "AsyncIterable",
        "AsyncIterator",
    }
)
MAPPINGS = frozenset({"dict", "Mapping", "MutableMapping"})
WRAPPERS = frozenset({"Optional", "Annotated", "Final", "ClassVar"})
KEEPS_CONTAINER = frozenset({"sorted", "list", "tuple", "reversed", "set", "frozenset", "iter"})
TAKES_ELEMENT = frozenset({"next", "min", "max"})
VALIDATORS = frozenset({"model_validator", "field_validator"})

# A name in the scope that binds it, or an attribute that a class's methods set on self.
Key = tuple["Scope | str", str]


def _wrap(prefix: str, ref: str | None) -> str | None:
    return prefix + ref if ref and ref[0].isidentifier() else None


def _element(ref: str | None) -> str | None:
    if ref and ref[0] == "*":
        return ref[1:]
    if ref and ref[0] == "~":
        return "^" + ref[1:]
    return None


def _decorators(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for decorator in node.decorator_list:
        match decorator:
            case ast.Name(id=name) | ast.Call(func=ast.Name(id=name)):
                names.add(name)
            case _:
                pass
    return names


def _parameters(node: ast.AST) -> list[ast.arg]:
    match node:
        case (
            ast.FunctionDef(args=arguments)
            | ast.AsyncFunctionDef(args=arguments)
            | ast.Lambda(args=arguments)
        ):
            rest = [arg for arg in (arguments.vararg, arguments.kwarg) if arg]
            return [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, *rest]
        case _:
            return []


def _binds(node: ast.AST) -> list[str]:
    """The names a node binds in the scope it belongs to, as Python decides locality."""
    match node:
        case (
            ast.Name(id=name, ctx=ast.Store())
            | ast.FunctionDef(name=name)
            | ast.AsyncFunctionDef(name=name)
            | ast.ClassDef(name=name)
            | ast.ExceptHandler(name=str() as name)
            | ast.MatchAs(name=str() as name)
            | ast.MatchStar(name=str() as name)
        ):
            return [name]
        case ast.alias(name=name, asname=asname):
            return [asname or name.partition(".")[0]]
        case _:
            return []


def _nested(annotation: object) -> tuple[str, type[BaseModel]] | None:
    """The model a field holds, and whether directly, in a sequence, or as mapping values."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return "", annotation
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is types.UnionType or origin is typing.Union:
        return next((found for arg in args if (found := _nested(arg))), None)
    mapping = origin in (dict, Mapping)
    if args and (mapping or origin in (list, tuple, set, frozenset, Sequence)):
        found = _nested(args[-1] if mapping else args[0])
        if found and not found[0]:
            return ("{" if mapping else "*"), found[1]
    return None


class Settings:
    """The keys a settings model accepts, as dotted paths down to the leaves."""

    def __init__(self, root: type[BaseModel]) -> None:
        self.fields: dict[str, dict[str, str | None]] = {}
        self.keys: list[tuple[str, str, str]] = []
        self._walk(root, "")

    def _walk(self, model: type[BaseModel], prefix: str) -> None:
        own = self.fields.setdefault(model.__name__, {})
        for name, info in model.model_fields.items():
            found = _nested(info.annotation)
            if found is None:
                own[name] = None
                self.keys.append((prefix + name, model.__name__, name))
                continue
            kind, child = found
            own[name] = kind + child.__name__
            self._walk(child, prefix + name + {"": ".", "*": "[].", "{": ".*."}[kind])


class Scan:
    """What each class holds and each function returns, across a set of source files.

    Annotations on classes and functions are taken as declared. A local name, or an
    attribute a method sets on self, takes the one type all of its bindings agree
    on, and the scan finds it in two rounds. The first skips a binding it cannot type
    yet, because that binding may only be waiting on a name typed later in the
    round. The second repeats until nothing changes and treats a binding it still
    cannot type as unknown, which leaves the name untyped. Both rounds leave a name
    bound to two types untyped. An untyped name never regains a type and a typed one
    never changes type, so each round stops.
    """

    def __init__(self, settings: Settings, settings_file: Path, sources: Iterable[Path]) -> None:
        self.settings = settings
        self.settings_file = settings_file
        trees = {path: ast.parse(path.read_text(encoding="utf-8")) for path in sources}
        self.classes = set(settings.fields)
        for tree in trees.values():
            self.classes.update(n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        self.members: dict[str, dict[str, str]] = {
            name: {field: ref for field, ref in fields.items() if ref}
            for name, fields in settings.fields.items()
        }
        self.methods: dict[str, dict[str, str]] = {}
        self.functions: dict[str, str] = {}
        self.scopes: list[Scope] = []
        self.known: dict[Key, str] = {}
        self.untyped: set[Key] = set()
        self.settling = False
        self.changed = False
        for path, tree in trees.items():
            self._declare(path, tree)

    def _declare(self, path: Path, tree: ast.Module) -> None:
        classes_and_functions = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        top = [n for n in tree.body if not isinstance(n, classes_and_functions)]
        Scope(self, path, None, tree, top)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                Scope(self, path, None, node, node.body)
                if ref := self.annotation(node.returns, None):
                    self.functions[node.name] = ref
            if not isinstance(node, ast.ClassDef):
                continue
            methods = self.methods.setdefault(node.name, {})
            members = self.members.setdefault(node.name, {})
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    ref = self.annotation(item.annotation, node.name)
                    if ref and item.target.id not in self.settings.fields.get(node.name, {}):
                        members[item.target.id] = ref
                if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                Scope(self, path, node.name, item, item.body)
                if ref := self.annotation(item.returns, node.name):
                    (members if "property" in _decorators(item) else methods)[item.name] = ref

    def annotation(self, node: ast.expr | None, owner: str | None) -> str | None:
        match node:
            case ast.Constant(value=str() as text):
                return self.annotation(ast.parse(text, mode="eval").body, owner)
            case ast.Name(id="Self"):
                return owner
            case ast.Name(id=name) | ast.Attribute(attr=name) if name in self.classes:
                return name
            case ast.BinOp(op=ast.BitOr(), left=left, right=right):
                return self.annotation(left, owner) or self.annotation(right, owner)
            case ast.Subscript(value=ast.Name(id=base) | ast.Attribute(attr=base), slice=inner):
                parts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
                if base in WRAPPERS:
                    return self.annotation(parts[0], owner)
                if base in SEQUENCES:
                    return _wrap("*", self.annotation(parts[0], owner))
                if base in MAPPINGS:
                    return _wrap("{", self.annotation(parts[-1], owner))
            case _:
                pass
        return None

    def member(self, owner: str, attr: str) -> str | None:
        return self.members.get(owner, {}).get(attr) or self.known.get((owner, attr))

    def learn(self, key: Key, ref: str | None) -> None:
        """Take one binding of a name, or of an attribute on self, into account."""
        if key in self.untyped or (ref is None and not self.settling):
            return
        held = self.known.get(key)
        if ref is not None and held in (None, ref):
            self.changed = self.changed or held is None
            self.known[key] = ref
        else:
            self.known.pop(key, None)
            self.untyped.add(key)
            self.changed = True

    def read(self) -> set[tuple[str, str]]:
        """Every (class, attribute) pair read outside the settings file, or on its behalf."""
        for settling in (False, True):
            self.settling = settling
            self.changed = True
            while self.changed:
                self.changed = False
                for scope in self.scopes:
                    scope.infer()
        found: set[tuple[str, str]] = set()
        on_behalf: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for scope in self.scopes:
            reads = set(scope.reads())
            if scope.path != self.settings_file:
                found |= reads
            elif scope.method and scope.owner and not scope.validator:
                on_behalf.setdefault((scope.owner, scope.method), set()).update(reads)
        used = [key for key in on_behalf if key in found]
        while used:
            for item in on_behalf.pop(used.pop(), set()):
                if item not in found:
                    found.add(item)
                    if item in on_behalf:
                        used.append(item)
        return found


class Scope:
    """One function, lambda or module top level, and the names bound in it.

    A name belongs to the innermost function that binds it, so a nested function or
    lambda sees the names around it unless it binds them itself, as Python resolves
    them. Module names are not followed into functions. Bindings are not ordered: a
    name bound to two types in one scope, or once to a value the scan cannot type,
    has no type anywhere in that scope, because the scan cannot tell which binding
    reaches a read.
    """

    def __init__(
        self,
        scan: Scan,
        path: Path,
        owner: str | None,
        node: ast.AST,
        body: Sequence[ast.AST],
        parent: Scope | None = None,
    ) -> None:
        self.scan = scan
        self.path = path
        self.owner = owner
        self.node = node
        self.parent = parent
        self.method: str | None = None
        self.validator = False
        self.receiver = False
        if parent:
            self.method, self.validator = parent.method, parent.validator
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            decorators = _decorators(node)
            self.method = node.name
            self.validator = bool(decorators & VALIDATORS)
            self.receiver = owner is not None and "staticmethod" not in decorators
        self.handled: set[ast.AST] = set()
        self.children: dict[ast.AST, Scope] = {}
        self.own: list[ast.AST] = []
        scan.scopes.append(self)
        pending = list(reversed(body))
        while pending:
            item = pending.pop()
            self.own.append(item)
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                inner = [item.body] if isinstance(item, ast.Lambda) else item.body
                self.children[item] = Scope(scan, path, owner, item, inner, self)
            else:
                pending.extend(reversed(list(ast.iter_child_nodes(item))))
        shared = {
            name
            for item in self.own
            if isinstance(item, ast.Global | ast.Nonlocal)
            for name in item.names
        }
        bound = {arg.arg for arg in _parameters(node)}
        bound.update(name for item in self.own for name in _binds(item))
        self.bound = bound - shared

    def scope_of(self, name: str) -> Scope | None:
        scope: Scope | None = self
        while scope is not None and name not in scope.bound:
            scope = scope.parent
        return scope

    def learn(self, name: str, ref: str | None) -> None:
        if scope := self.scope_of(name):
            self.scan.learn((scope, name), ref)

    def type_of(self, node: ast.expr | None) -> str | None:
        scan = self.scan
        match node:
            case ast.Name(id=name):
                scope = self.scope_of(name)
                return scan.known.get((scope, name)) if scope else None
            case ast.Attribute(value=value, attr=attr):
                receiver = self.type_of(value)
                return scan.member(receiver, attr) if receiver else None
            case ast.Call(func=ast.Attribute(value=ast.Name(id=name), attr=attr)) if (
                name in scan.methods and self.scope_of(name) is None
            ):
                return scan.methods[name].get(attr)
            case ast.Call(func=ast.Attribute(value=value, attr=attr)):
                receiver = self.type_of(value)
                if receiver and receiver[0] == "{":
                    prefix = {"values": "*", "items": "~", "get": ""}.get(attr)
                    return None if prefix is None else prefix + receiver[1:]
                if receiver and attr == "model_copy":
                    return receiver
                return scan.methods.get(receiver, {}).get(attr) if receiver else None
            case ast.Call(func=ast.Name(id=name), args=[first, *_]) if name in KEEPS_CONTAINER:
                found = self.type_of(first)
                return found if found and found[0] in "*~" else None
            case ast.Call(func=ast.Name(id=name), args=[first, *_]) if name in TAKES_ELEMENT:
                return _element(self.type_of(first))
            case ast.Call(func=ast.Name(id="enumerate"), args=[first, *_]):
                return _wrap("~", _element(self.type_of(first)))
            case ast.Call(func=ast.Name(id="cls")):
                return self.owner
            case ast.Call(func=ast.Name(id=name)):
                return name if name in scan.classes else scan.functions.get(name)
            case ast.Await(value=value):
                return self.type_of(value)
            case ast.Subscript(value=value):
                receiver = self.type_of(value)
                return receiver[1:] if receiver and receiver[0] in "*{" else None
            case ast.IfExp(body=body, orelse=orelse):
                return self.type_of(body) or self.type_of(orelse)
            case ast.BoolOp(values=values):
                return next((found for value in values if (found := self.type_of(value))), None)
            case ast.NamedExpr(value=value):
                return self.type_of(value)
            case ast.ListComp(elt=elt) | ast.GeneratorExp(elt=elt) | ast.SetComp(elt=elt):
                return _wrap("*", self.type_of(elt))
            case ast.DictComp(value=value):
                return _wrap("{", self.type_of(value))
            case _:
                pass
        return None

    def bind(self, target: ast.expr, ref: str | None) -> None:
        match target:
            case ast.Name(id=name):
                self.handled.add(target)
                self.learn(name, ref)
            case ast.Tuple(elts=[_, second]) if ref and ref[0] == "^":
                self.bind(second, ref[1:])
            case ast.Attribute(value=ast.Name(id="self"), attr=attr) if self.owner:
                declared = self.scan.settings.fields.get(self.owner, {})
                if attr not in declared and attr not in self.scan.members.get(self.owner, {}):
                    self.scan.learn((self.owner, attr), ref)
            case _:
                pass

    def infer(self) -> None:
        """Offer every binding in this scope to the scan, typed from what it knows so far."""
        self.handled.clear()
        if isinstance(self.node, ast.FunctionDef | ast.AsyncFunctionDef):
            arguments = self.node.args
            named = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
            for position, arg in enumerate(named):
                if position == 0 and self.receiver:
                    self.learn(arg.arg, self.owner)
                else:
                    self.learn(arg.arg, self.scan.annotation(arg.annotation, self.owner))
            for arg in (arguments.vararg, arguments.kwarg):
                if arg:
                    self.learn(arg.arg, None)
        for node in self.own:
            match node:
                case ast.Assign(targets=targets, value=value):
                    for target in targets:
                        self.bind(target, self.type_of(value))
                case ast.AnnAssign(target=target, annotation=annotation, value=value):
                    ref = self.scan.annotation(annotation, self.owner) or self.type_of(value)
                    self.bind(target, ref)
                case (
                    ast.For(target=target, iter=source)
                    | ast.AsyncFor(target=target, iter=source)
                    | ast.comprehension(target=target, iter=source)
                ):
                    self.bind(target, _element(self.type_of(source)))
                case ast.withitem(context_expr=context, optional_vars=ast.expr() as target):
                    ref = self.type_of(context)
                    self.bind(target, _element(ref) or ref)
                case ast.NamedExpr(target=target, value=value):
                    self.bind(target, self.type_of(value))
                case ast.Call(func=ast.Name(id="sorted" | "min" | "max"), args=[source, *_]):
                    element = _element(self.type_of(source))
                    for keyword in node.keywords:
                        key = keyword.value
                        if keyword.arg == "key" and isinstance(key, ast.Lambda) and key.args.args:
                            self.scan.learn((self.children[key], key.args.args[0].arg), element)
                case ast.Name() if node in self.handled:
                    pass
                case _:
                    for name in _binds(node):
                        self.learn(name, None)

    def reads(self) -> Iterator[tuple[str, str]]:
        for node in self.own:
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                receiver = self.type_of(node.value)
                if receiver:
                    yield receiver, node.attr


def unread_keys(root: type[BaseModel], settings_file: Path, sources: Iterable[Path]) -> list[str]:
    settings = Settings(root)
    read = Scan(settings, settings_file, sources).read()
    return [key for key, model, field in settings.keys if (model, field) not in read]


def _covers(entry: str, key: str) -> bool:
    return key == entry or key.startswith((entry + ".", entry + "["))


@pytest.fixture(scope="module")
def unread() -> list[str]:
    sources = sorted(path for name in READER_DIRS for path in (REPO_ROOT / name).rglob("*.py"))
    return unread_keys(Config, SETTINGS_FILE, sources)


def test_every_key_is_read_or_names_its_reader(unread: list[str]) -> None:
    orphaned = [key for key in unread if not any(_covers(entry, key) for entry in PENDING)]
    assert orphaned == [], (
        f"nothing reads {orphaned}. Read each one, delete it from Config, or add it to"
        " PENDING with the reader that will use it."
    )


def test_a_pending_entry_leaves_once_its_key_is_read(unread: list[str]) -> None:
    keys = [key for key, _, _ in Settings(Config).keys]
    unknown = [entry for entry in PENDING if not any(_covers(entry, key) for key in keys)]
    assert unknown == [], f"PENDING names keys Config does not have: {unknown}"
    now_read = [
        entry for entry in PENDING if any(_covers(entry, key) and key not in unread for key in keys)
    ]
    assert now_read == [], f"these PENDING entries cover keys that are now read: {now_read}"


SCRATCH_FIELD = """\
    never_read_s: int = 0

    @model_validator(mode="after")
    def _never_read_is_not_negative(self) -> Self:
        if self.never_read_s < 0:
            raise ValueError("never_read_s must not be negative")
        return self

"""

SCRATCH_IMPORTS = """\
import json
from collections.abc import AsyncIterator

from scratch_config import Config, HealthConfig


"""

READERS = {
    "local": """\
def window(config: Config) -> int:
    health = config.health
    return health.never_read_s
""",
    "closure": """\
def window(config: Config) -> int:
    def inner() -> int:
        return config.health.never_read_s

    return inner()
""",
    "sort-key": """\
def widest(configs: list[HealthConfig]) -> HealthConfig:
    return max(configs, key=lambda health: health.never_read_s)
""",
    "async-method": """\
class Poller:
    def __init__(self, config: Config) -> None:
        self.config = config

    async def health(self) -> HealthConfig:
        return self.config.health

    async def window(self) -> int:
        health = await self.health()
        return health.never_read_s
""",
    "async-for": """\
async def window(stream: AsyncIterator[HealthConfig]) -> int:
    async for health in stream:
        return health.never_read_s
    return 0
""",
    "staticmethod": """\
class Poller:
    @staticmethod
    def window(health: HealthConfig) -> int:
        return health.never_read_s
""",
}

NOT_READERS = {
    "parameter-shadows": """\
def window(config: Config) -> int:
    health = config.health

    def inner(health: object) -> int:
        return health.never_read_s

    return inner(health)
""",
    "lambda-shadows": """\
def window(config: Config) -> int:
    health = config.health
    pick = lambda health: health.never_read_s
    return pick(health)
""",
    "rebound-to-unknown": """\
def window(config: Config, raw: str) -> int:
    health = config.health
    health = json.loads(raw)
    return health.never_read_s
""",
    "bound-to-two-models": """\
def window(config: Config) -> int:
    health = config.redaction
    health = config.health
    return health.never_read_s
""",
    "attribute-rebound-to-unknown": """\
class Poller:
    def __init__(self, config: Config, raw: str) -> None:
        self.health = config.health
        if raw:
            self.health = json.loads(raw)

    def window(self) -> int:
        return self.health.never_read_s
""",
}


@pytest.fixture
def scratch_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[type[BaseModel], Path]:
    """A copy of config.py with one more health key, which only a validator reads."""
    source = SETTINGS_FILE.read_text(encoding="utf-8")
    anchor = "class HealthConfig(Strict):\n"
    assert anchor in source
    path = tmp_path / "scratch_config.py"
    path.write_text(source.replace(anchor, anchor + SCRATCH_FIELD, 1), encoding="utf-8")
    location = importlib.util.spec_from_file_location("scratch_config", path)
    assert location is not None
    assert location.loader is not None
    module = importlib.util.module_from_spec(location)
    monkeypatch.setitem(sys.modules, "scratch_config", module)
    location.loader.exec_module(module)
    return module.Config, path


def _unread_with(scratch_settings: tuple[type[BaseModel], Path], reader: str) -> list[str]:
    root, path = scratch_settings
    planted = path.with_name("scratch_reader.py")
    planted.write_text(SCRATCH_IMPORTS + reader, encoding="utf-8")
    return unread_keys(root, path, [path, planted])


def test_a_key_only_a_validator_reads_is_reported_by_name(
    scratch_settings: tuple[type[BaseModel], Path],
) -> None:
    root, path = scratch_settings
    assert "health.never_read_s" in [key for key, _, _ in Settings(root).keys]
    assert "health.never_read_s" in unread_keys(root, path, [path])


@pytest.mark.parametrize("reader", list(READERS.values()), ids=list(READERS))
def test_a_reader_outside_the_settings_file_clears_the_key(
    scratch_settings: tuple[type[BaseModel], Path], reader: str
) -> None:
    assert "health.never_read_s" not in _unread_with(scratch_settings, reader)


@pytest.mark.parametrize("reader", list(NOT_READERS.values()), ids=list(NOT_READERS))
def test_a_name_that_may_hold_another_value_is_not_a_reader(
    scratch_settings: tuple[type[BaseModel], Path], reader: str
) -> None:
    assert "health.never_read_s" in _unread_with(scratch_settings, reader)
