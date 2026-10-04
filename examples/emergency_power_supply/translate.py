"""Translate the emergency-power-supply benchmark into RAICHU models.

The benchmark is the BDMP of the emergency power supply of a nuclear
power plant published by M. Bouissou (2017), "A Benchmark on Reliability
of Complex Discrete Systems: Emergency Power Supply of a Nuclear Power
Plant", MARS 2017, EPTCS 244, pp. 200-216, doi:10.4204/EPTCS.244.8,
arXiv:1703.06575. The model is not re-drawn from the paper's figures: it
is translated from the paper's ancillary file `export_fig0.fi`, the
Figaro 0 text that was the input of FIGSEQ and YAMS (arXiv ancillary
file, README: "only instantaneous probabilistic transitions and timed
transitions associated to exponential distributions"; the one-hour
battery depletion is already the Erlang(2, 2/h) of section 3.3). The file
is fetched from arXiv and pinned by its SHA-256, and never copied into
the repository.

# Figaro 0 semantics, and how each part is carried over

A Figaro 0 state is the value of every attribute declared with an
initial value (`= v`). After every transition an *interaction phase*
runs: each attribute declared `REINITIALISATION v` is reset to `v`, then
the interaction rules run step by step, in `STEPS_ORDER`, each step until
nothing changes. The occurrence rules are evaluated on the state the
phase ends in (Bouissou and Houdebine 2002, the semantics the file's
README points to).

- **Reset attributes** (`S`, `required`, `relevant_evt`, `to_be_fired`,
  the repairman `priority`) are functions of the state. They are computed
  here symbolically, step by step, in static single assignment: a rule
  reading an attribute a later step writes reads its reset value, one
  reading an attribute an earlier (or the same) step writes reads that
  step's value. Each becomes a RAICHU attribute written by a sensitive
  function, whose fixpoint is the step's fixpoint because the dependency
  graph is acyclic (checked).
- **Memory attributes**, declared with an initial value but written by
  interaction rules (`already_S`, read by the priority-AND gates; the
  demand latches `already_standby` of the on-demand leaves; the repair
  queue `waiting_for_rep` and the repairman counter `nb_avail_repairmen`)
  keep a value from one phase to the next. RAICHU stores the value each
  one had when the phase started, and every transition that starts a
  phase writes the value the previous phase ended with, before its own
  `INDUCING` assignments (edge effects).
- **State attributes**, written only by `INDUCING` (`failF`, `failI`,
  `failAG`), are RAICHU attributes written by edge effects.
- **Timed occurrences** (`DIST EXP`) become exponential self-loop
  transitions, one automaton each, guarded by the occurrence condition.
- **Instantaneous occurrences** (`DIST INS`) are fired *simultaneously*
  by Figaro 0 when several are valid in the same state (the 2017 paper,
  section 4, reads FIGSEQ's output that way). RAICHU fires instantaneous
  transitions one at a time, so each one first moves to a pending state
  invisible to the rest of the model; once no instantaneous occurrence is
  valid any more, a phase controller commits all pending choices, and
  only then do the next ones become valid. The draws therefore see the
  state their Figaro 0 counterparts see.
- **The feared event** is the `S` attribute of the undesirable event
  `UE_1`, the loss of both bus bars LHA and LHB. A watcher reaches its
  `reached` state as soon as it holds between two phases, and every
  instantaneous transition is closed once it has: a Figaro sequence ends
  there.

The generator checks its own translation: every reset and memory
attribute it derives is evaluated on random states against a direct
interpreter of the Figaro 0 steps (fixpoint iteration of the rules), and
the two must agree on every sample.

# Variants

Each variant is one keyword of `build` and one file of `main`:

- `eps_benchmark.json`: the published, repairable model.
- `eps_benchmark_nonrepairable.json`: the non-repairable variant compared
  in Bouissou et al. (2020), arXiv:2004.13283, Table 3, obtained with the
  model's own switch, `repairable_system OF OPTIONS = FALSE`.
- `eps_benchmark_fast_line_repair.json`: the sensitivity case of the same
  paper, Table 2: the mean repair time of the common-cause loss of both
  lines (`mu OF CCF_GEV_LGR`, 1/200 per hour) divided by 10.
- `eps_benchmark_battery_1h.json`: the battery depletion as the fixed one
  hour the system description states (2017, section 3.3), the setting of
  the YAMS run of the 2020 paper, Table 1. Each of the two exponential
  stages (rate 2 per hour) of the file's Erlang(2, 2/h) becomes a fixed
  delay of half an hour, so the depletion takes exactly one hour; a stage
  whose guard drops restarts from zero (`on_interruption: reset`), the
  "optimistic" battery of the 2020 paper, section 5.2.

The translated models are adaptations of the Figaro 0 file, which is not
ours to redistribute: they are written to `model/`, which is not
versioned, and every reader regenerates them.

Run: python translate.py [--out DIR]  (fetches the Figaro 0 file into a
cache, or reads EPS_FIGARO_SOURCE when it names a local copy).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"

# Variant name -> keywords of `build`.
VARIANTS = {
    "eps_benchmark": {},
    "eps_benchmark_nonrepairable": {"repairable": False},
    "eps_benchmark_fast_line_repair": {"line_ccf_repair_factor": 10.0},
    "eps_benchmark_battery_1h": {"battery": "fixed"},
}
# The battery stages of the file's Erlang(2, 2/h): two leaves per train.
BATTERY_STAGES = ("BATT_A1", "BATT_A2", "BATT_B1", "BATT_B2")
# Fixed duration of one stage, in hours, so that two stages last one hour.
BATTERY_STAGE_TIME = 0.5

FIGARO_URL = "https://arxiv.org/src/1703.06575v1/anc/export_fig0.fi"
FIGARO_SHA256 = "417755be9dd0d6c5bc1205347a964576791d213d9714455dfb66643b99355345"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "raichu"

# The feared event of the benchmark: S of the undesirable event UE_1.
UNDESIRABLE_EVENT = "UE_1"
TARGET_NAME = "LHA_and_LHB_lost"
# Mission time of the benchmark (2017 paper, section 2.1), in hours.
MISSION_TIME = 10000.0

FEARED = "feared_event"
PHASE = "phase_ctl"

# ---------------------------------------------------------------- source


def figaro_source() -> str:
    """The Figaro 0 text of the benchmark, checked against its pinned
    SHA-256 (CRLF line ends normalised after the check)."""
    local = os.environ.get("EPS_FIGARO_SOURCE")
    if local:
        raw = Path(local).read_bytes()
    else:
        cached = CACHE / "eps_export_fig0.fi"
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(FIGARO_URL, timeout=60) as response:
                cached.write_bytes(response.read())
        raw = cached.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FIGARO_SHA256:
        raise ValueError(f"Figaro source digest {digest} != pinned {FIGARO_SHA256}")
    return raw.decode("ascii").replace("\r\n", "\n")


# ---------------------------------------------------------------- lexer

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>\(\*.*?\*\))
  | (?P<label>"[^"]*")
  | (?P<enum>'[^']*')
  | (?P<num>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)
  | (?P<op><--|<=|>=|<>|[=<>+\-*/(),;])
  | (?P<id>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE | re.DOTALL,
)


def tokenize(text: str) -> list[tuple[str, str]]:
    tokens = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:
            raise SyntaxError(
                f"unexpected character at {pos}: {text[pos : pos + 20]!r}"
            )
        pos = match.end()
        kind = match.lastgroup
        if kind in ("ws", "comment"):
            continue
        tokens.append((kind, match.group()))
    return tokens


# ---------------------------------------------------------------- AST
# Expressions are tuples:
#   ("const", value)                    bool | float | str
#   ("ref", obj, attr)
#   ("not", e) ("and", (e, ...)) ("or", (e, ...))
#   ("cmp", op, a, b)                   op in = <> < > <= >=
#   ("arith", op, a, b)                 op in + - * /
#   ("if", c, a, b)


@dataclass
class Rule:
    """An interaction rule `IF cond THEN target <-- value, ...`."""

    obj: str
    name: str
    step: str
    cond: tuple
    assigns: list[tuple[tuple[str, str], tuple]]


@dataclass
class Alternative:
    """One alternative of an occurrence rule."""

    kind: str  # FAULT | REPAIR | TRANSITION
    name: str
    law: str  # EXP | INS
    param: tuple
    inducing: list[tuple[tuple[str, str], tuple]]


@dataclass
class Occurrence:
    obj: str
    name: str
    cond: tuple
    alternatives: list[Alternative]


@dataclass
class FObject:
    name: str
    type: str
    interface: dict[str, list[str]] = field(default_factory=dict)
    # attr -> (domain, "reinit" | "init", expr)
    attributes: dict[str, tuple] = field(default_factory=dict)
    constants: dict[str, tuple] = field(default_factory=dict)


@dataclass
class FModel:
    steps: list[str]
    objects: dict[str, FObject]
    rules: list[Rule]
    occurrences: list[Occurrence]


_SECTIONS = {"INTERFACE", "ATTRIBUTE", "CONSTANT", "INTERACTION", "OCCURRENCE"}


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def peek(self, offset=0):
        j = self.i + offset
        return self.tokens[j][1] if j < len(self.tokens) else None

    def next(self):
        token = self.tokens[self.i][1]
        self.i += 1
        return token

    def expect(self, value):
        token = self.next()
        if token != value:
            raise SyntaxError(f"expected {value!r}, got {token!r} at token {self.i}")
        return token

    def at_end(self):
        return self.i >= len(self.tokens)

    # -- file
    def parse(self) -> FModel:
        self.expect("STEPS_ORDER")
        steps = []
        while self.peek() != "GROUP_NAMES":
            steps.append(self.next())
            self.expect(";")
        self.expect("GROUP_NAMES")
        while self.peek() != "OBJECT":
            self.next()
            self.expect(";")
        objects, rules, occurrences = {}, [], []
        while not self.at_end():
            obj = self.parse_object(rules, occurrences)
            objects[obj.name] = obj
        return FModel(steps, objects, rules, occurrences)

    def parse_object(self, rules, occurrences) -> FObject:
        self.expect("OBJECT")
        name = self.next()
        self.expect("IS_A")
        obj = FObject(name, self.next())
        self.expect(";")
        while not self.at_end() and self.peek() != "OBJECT":
            section = self.next()
            if section not in _SECTIONS:
                raise SyntaxError(f"unknown section {section!r} in {name}")
            while not self.at_end() and self.peek() not in _SECTIONS | {"OBJECT"}:
                if section == "INTERFACE":
                    key = self.next()
                    self.expect("=")
                    items = []
                    while self.peek() != ";":
                        items.append(self.next())
                    self.expect(";")
                    obj.interface[key] = items
                elif section in ("ATTRIBUTE", "CONSTANT"):
                    key = self.next()
                    if self.peek() == "LABEL":
                        self.next()
                        self.next()
                    self.expect("DOMAIN")
                    domain = self.parse_domain()
                    if self.peek() == "REINITIALISATION":
                        self.next()
                        mode = "reinit"
                    else:
                        self.expect("=")
                        mode = "init"
                    value = self.parse_expr()
                    self.expect(";")
                    if section == "ATTRIBUTE":
                        obj.attributes[key] = (domain, mode, value)
                    else:
                        obj.constants[key] = (domain, value)
                elif section == "INTERACTION":
                    rules.append(self.parse_rule(name))
                else:
                    occurrences.append(self.parse_occurrence(name))
        return obj

    def parse_domain(self):
        token = self.peek()
        if token in ("BOOLEAN", "INTEGER", "REAL"):
            return self.next()
        values = []
        while self.peek() and self.peek().startswith("'"):
            values.append(self.next().strip("'"))
        return tuple(values)

    def parse_rule(self, obj) -> Rule:
        name = self.next()
        self.expect("GROUP")
        self.next()
        self.expect("STEP")
        step = self.next()
        cond = ("const", True)
        if self.peek() == "IF":
            self.next()
            cond = self.parse_expr()
        self.expect("THEN")
        assigns = self.parse_assigns()
        self.expect(";")
        return Rule(obj, name, step, cond, assigns)

    def parse_assigns(self):
        assigns = [self.parse_assign()]
        while self.peek() == ",":
            self.next()
            assigns.append(self.parse_assign())
        return assigns

    def parse_assign(self):
        attr = self.next()
        self.expect("OF")
        obj = self.next()
        self.expect("<--")
        return ((obj, attr), self.parse_expr())

    def parse_occurrence(self, obj) -> Occurrence:
        name = self.next()
        self.expect("GROUP")
        self.next()
        self.expect("IF")
        cond = self.parse_expr()
        self.expect("MAY_OCCUR")
        alternatives = [self.parse_alternative()]
        while self.peek() == "OR_ELSE":
            self.next()
            alternatives.append(self.parse_alternative())
        self.expect(";")
        return Occurrence(obj, name, cond, alternatives)

    def parse_alternative(self) -> Alternative:
        kind = self.next()
        if kind not in ("FAULT", "REPAIR", "TRANSITION"):
            raise SyntaxError(f"unknown alternative {kind!r}")
        name = self.next()
        if self.peek() == "LABEL":
            self.next()
            self.next()
        self.expect("DIST")
        law = self.next()
        param = self.parse_expr()
        inducing = []
        if self.peek() == "INDUCING":
            self.next()
            inducing = self.parse_assigns()
        return Alternative(kind, name, law, param, inducing)

    # -- expressions
    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        args = [self.parse_and()]
        while self.peek() == "OR":
            self.next()
            args.append(self.parse_and())
        return args[0] if len(args) == 1 else ("or", tuple(args))

    def parse_and(self):
        args = [self.parse_not()]
        while self.peek() == "AND":
            self.next()
            args.append(self.parse_not())
        return args[0] if len(args) == 1 else ("and", tuple(args))

    def parse_not(self):
        if self.peek() == "NOT":
            self.next()
            return ("not", self.parse_not())
        return self.parse_cmp()

    def parse_cmp(self):
        left = self.parse_add()
        if self.peek() in ("=", "<>", "<", ">", "<=", ">="):
            op = self.next()
            return ("cmp", op, left, self.parse_add())
        return left

    def parse_add(self):
        left = self.parse_mul()
        while self.peek() in ("+", "-"):
            op = self.next()
            left = ("arith", op, left, self.parse_mul())
        return left

    def parse_mul(self):
        left = self.parse_unary()
        while self.peek() in ("*", "/"):
            op = self.next()
            left = ("arith", op, left, self.parse_unary())
        return left

    def parse_unary(self):
        if self.peek() == "-":
            self.next()
            return ("arith", "-", ("const", 0.0), self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        token = self.next()
        if token == "(":
            inner = self.parse_expr()
            self.expect(")")
            return inner
        if token == "TRUE":
            return ("const", True)
        if token == "FALSE":
            return ("const", False)
        if token.startswith("'"):
            return ("const", token.strip("'"))
        if token[0].isdigit():
            return ("const", float(token))
        self.expect("OF")
        return ("ref", self.next(), token)


# ---------------------------------------------------------------- expressions


def refs(expr, out=None):
    """The `(obj, attr)` pairs an expression reads."""
    if out is None:
        out = set()
    tag = expr[0]
    if tag == "ref":
        out.add((expr[1], expr[2]))
    elif tag == "not":
        refs(expr[1], out)
    elif tag in ("and", "or"):
        for arg in expr[1]:
            refs(arg, out)
    elif tag in ("cmp", "arith"):
        refs(expr[2], out)
        refs(expr[3], out)
    elif tag == "if":
        for arg in expr[1:]:
            refs(arg, out)
    return out


def substitute(expr, mapping):
    """Replace each `ref` found in `mapping` by its expression."""
    tag = expr[0]
    if tag == "ref":
        return mapping.get((expr[1], expr[2]), expr)
    if tag == "const":
        return expr
    if tag == "not":
        return ("not", substitute(expr[1], mapping))
    if tag in ("and", "or"):
        return (tag, tuple(substitute(a, mapping) for a in expr[1]))
    if tag in ("cmp", "arith"):
        return (
            tag,
            expr[1],
            substitute(expr[2], mapping),
            substitute(expr[3], mapping),
        )
    if tag == "if":
        return ("if",) + tuple(substitute(a, mapping) for a in expr[1:])
    raise ValueError(tag)


def _arith(op, a, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a / b


def _compare(op, a, b):
    return {
        "=": a == b,
        "<>": a != b,
        "<": a < b,
        ">": a > b,
        "<=": a <= b,
        ">=": a >= b,
    }[op]


def evaluate(expr, lookup):
    """Concrete value of `expr`, `lookup((obj, attr))` giving each read."""
    tag = expr[0]
    if tag == "const":
        return expr[1]
    if tag == "ref":
        return lookup((expr[1], expr[2]))
    if tag == "not":
        return not evaluate(expr[1], lookup)
    if tag == "and":
        return all(evaluate(a, lookup) for a in expr[1])
    if tag == "or":
        return any(evaluate(a, lookup) for a in expr[1])
    if tag == "cmp":
        return _compare(expr[1], evaluate(expr[2], lookup), evaluate(expr[3], lookup))
    if tag == "arith":
        return _arith(expr[1], evaluate(expr[2], lookup), evaluate(expr[3], lookup))
    if tag == "if":
        return evaluate(expr[2] if evaluate(expr[1], lookup) else expr[3], lookup)
    raise ValueError(tag)


TRUE = ("const", True)
FALSE = ("const", False)


def simplify(expr, is_bool):
    """Constant folding and boolean simplification. `is_bool(ref)` says
    whether a referenced attribute is boolean."""
    tag = expr[0]
    if tag in ("const", "ref"):
        return expr
    if tag == "not":
        inner = simplify(expr[1], is_bool)
        if inner[0] == "const":
            return ("const", not inner[1])
        if inner[0] == "not":
            return inner[1]
        return ("not", inner)
    if tag in ("and", "or"):
        neutral = tag == "and"
        args = []
        for arg in expr[1]:
            arg = simplify(arg, is_bool)
            if arg[0] == tag:
                args.extend(arg[1])
            elif arg[0] == "const":
                if arg[1] != neutral:
                    return ("const", not neutral)
            else:
                args.append(arg)
        unique = []
        for arg in args:
            if arg not in unique:
                unique.append(arg)
        for arg in unique:
            if ("not", arg) in unique:
                return ("const", not neutral)
        if not unique:
            return ("const", neutral)
        if len(unique) == 1:
            return unique[0]
        return _absorb((tag, tuple(unique)))
    if tag == "cmp":
        a = simplify(expr[2], is_bool)
        b = simplify(expr[3], is_bool)
        if a[0] == "const" and b[0] == "const":
            return ("const", _compare(expr[1], a[1], b[1]))
        if expr[1] in ("=", "<>") and b[0] == "const" and isinstance(b[1], bool):
            positive = (expr[1] == "=") == b[1]
            return a if positive else simplify(("not", a), is_bool)
        return ("cmp", expr[1], a, b)
    if tag == "arith":
        a = simplify(expr[2], is_bool)
        b = simplify(expr[3], is_bool)
        if a[0] == "const" and b[0] == "const":
            return ("const", _arith(expr[1], a[1], b[1]))
        return ("arith", expr[1], a, b)
    if tag == "if":
        c = simplify(expr[1], is_bool)
        a = simplify(expr[2], is_bool)
        b = simplify(expr[3], is_bool)
        if c[0] == "const":
            return a if c[1] else b
        if a == b:
            return a
        if a == TRUE and b == FALSE:
            return c
        if a == FALSE and b == TRUE:
            return simplify(("not", c), is_bool)
        if _is_boolean(a, is_bool) and _is_boolean(b, is_bool):
            return simplify(
                ("or", (("and", (c, a)), ("and", (("not", c), b)))), is_bool
            )
        return ("if", c, a, b)
    raise ValueError(tag)


def _is_boolean(expr, is_bool):
    tag = expr[0]
    if tag == "const":
        return isinstance(expr[1], bool)
    if tag == "ref":
        return is_bool((expr[1], expr[2]))
    if tag in ("not", "and", "or", "cmp"):
        return True
    if tag == "if":
        return _is_boolean(expr[2], is_bool) and _is_boolean(expr[3], is_bool)
    return False


def _absorb(expr):
    """x AND (x OR y) = x, x OR (x AND y) = x."""
    tag, args = expr
    other = "or" if tag == "and" else "and"
    plain = {a for a in args if a[0] != other}
    kept = [
        a for a in args if not (a[0] == other and any(sub in plain for sub in a[1]))
    ]
    return kept[0] if len(kept) == 1 else (tag, tuple(kept))


def _atoms(expr, is_bool, out):
    """Boolean atoms of a boolean expression (refs and comparisons)."""
    tag = expr[0]
    if tag == "not":
        _atoms(expr[1], is_bool, out)
    elif tag in ("and", "or"):
        for arg in expr[1]:
            _atoms(arg, is_bool, out)
    elif tag == "const":
        pass
    else:
        if expr not in out:
            out.append(expr)
    return out


def _truth(expr, valuation):
    tag = expr[0]
    if tag == "const":
        return expr[1]
    if tag == "not":
        return not _truth(expr[1], valuation)
    if tag == "and":
        return all(_truth(a, valuation) for a in expr[1])
    if tag == "or":
        return any(_truth(a, valuation) for a in expr[1])
    return valuation[expr]


def minimise(expr, is_bool, max_atoms=10):
    """Exact two-level minimisation of a small boolean expression: the
    truth table over its atoms, then a greedy prime-implicant cover. An
    expression with more atoms, or that is not boolean, is returned as
    is. The result is equivalent on every valuation."""
    expr = simplify(expr, is_bool)
    if expr[0] in ("const", "ref") or not _is_boolean(expr, is_bool):
        return expr
    if expr[0] == "if":
        return expr
    atoms = _atoms(expr, is_bool, [])
    if len(atoms) > max_atoms:
        return expr
    n = len(atoms)
    ones = []
    for bits in range(1 << n):
        valuation = {a: bool(bits >> i & 1) for i, a in enumerate(atoms)}
        if _truth(expr, valuation):
            ones.append(bits)
    if not ones:
        return FALSE
    if len(ones) == 1 << n:
        return TRUE
    # Quine-McCluskey prime implicants: (value, mask) with mask = don't-care bits.
    terms = {(b, 0) for b in ones}
    primes = set()
    while terms:
        merged = set()
        used = set()
        term_list = sorted(terms)
        for i, (v1, m1) in enumerate(term_list):
            for v2, m2 in term_list[i + 1 :]:
                if m1 == m2:
                    diff = v1 ^ v2
                    if diff and diff & (diff - 1) == 0:
                        merged.add((v1 & ~diff, m1 | diff))
                        used.add((v1, m1))
                        used.add((v2, m2))
        primes |= terms - used
        terms = merged
    uncovered = set(ones)
    cover = []
    for value, mask in sorted(primes, key=lambda p: -p[1].bit_count()):
        covered = {b for b in uncovered if b & ~mask == value}
        if covered:
            cover.append((value, mask))
            uncovered -= covered
    products = []
    for value, mask in cover:
        literals = []
        for i, atom in enumerate(atoms):
            if mask >> i & 1:
                continue
            literals.append(atom if value >> i & 1 else ("not", atom))
        products.append(literals[0] if len(literals) == 1 else ("and", tuple(literals)))
    result = products[0] if len(products) == 1 else ("or", tuple(products))
    result = simplify(result, is_bool)
    return result if size(result) <= size(expr) else expr


def size(expr):
    tag = expr[0]
    if tag in ("const", "ref"):
        return 1
    if tag == "not":
        return 1 + size(expr[1])
    if tag in ("and", "or"):
        return 1 + sum(size(a) for a in expr[1])
    if tag in ("cmp", "arith"):
        return 1 + size(expr[2]) + size(expr[3])
    return 1 + sum(size(a) for a in expr[1:])


# ---------------------------------------------------------------- analysis


class Translation:
    """The Figaro 0 model folded, classified and put in static single
    assignment over its interaction steps."""

    def __init__(self, fmodel: FModel, overrides: dict | None = None):
        self.f = fmodel
        self.step_index = {s: i for i, s in enumerate(fmodel.steps)}
        self.domain = {}
        self.kind = {}  # (obj, attr) -> const | derived | memory | state
        self.initial = {}
        self.const = {}
        for obj in fmodel.objects.values():
            for attr, (domain, mode, value) in obj.attributes.items():
                self.domain[(obj.name, attr)] = domain
                self.initial[(obj.name, attr)] = (mode, value)
            for attr, (domain, value) in obj.constants.items():
                self.domain[(obj.name, attr)] = domain
        self._fold_constants(overrides or {})
        self.rules = list(fmodel.rules)
        self.occurrences = list(fmodel.occurrences)
        self._classify()
        self._ssa()

    # -- constants
    def is_bool(self, ref):
        return self.domain.get(ref) == "BOOLEAN"

    def _fold_constants(self, overrides):
        pending = {}
        for obj in self.f.objects.values():
            for attr, (_domain, value) in obj.constants.items():
                pending[(obj.name, attr)] = value
        for ref, value in overrides.items():
            if ref not in pending:
                raise KeyError(f"override of unknown constant {ref}")
            pending[ref] = ("const", value)
        while pending:
            progress = False
            for ref, value in list(pending.items()):
                value = self._fold(value)
                if value[0] == "const":
                    self.const[ref] = self._typed(ref, value[1])
                    del pending[ref]
                    progress = True
                else:
                    pending[ref] = value
            if not progress:
                raise ValueError(f"non-constant constants: {sorted(pending)}")

    def _typed(self, ref, value):
        domain = self.domain.get(ref)
        if domain == "INTEGER" or domain == "REAL":
            return float(value)
        return value

    def _fold(self, expr):
        mapping = {ref: ("const", v) for ref, v in self.const.items()}
        return simplify(substitute(expr, mapping), self.is_bool)

    # -- classification
    def _classify(self):
        while True:
            self.rules = [
                Rule(
                    r.obj,
                    r.name,
                    r.step,
                    self._fold(r.cond),
                    [(t, self._fold(v)) for t, v in r.assigns],
                )
                for r in self.rules
            ]
            self.rules = [r for r in self.rules if r.cond != FALSE]
            occurrences = []
            for occ in self.occurrences:
                cond = self._fold(occ.cond)
                if cond == FALSE:
                    continue
                alternatives = [
                    Alternative(
                        a.kind,
                        a.name,
                        a.law,
                        self._fold(a.param),
                        [(t, self._fold(v)) for t, v in a.inducing],
                    )
                    for a in occ.alternatives
                ]
                occurrences.append(Occurrence(occ.obj, occ.name, cond, alternatives))
            self.occurrences = occurrences
            written = {}
            induced = {}
            for rule in self.rules:
                for target, value in rule.assigns:
                    written.setdefault(target, []).append(value)
            for occ in self.occurrences:
                for alt in occ.alternatives:
                    for target, value in alt.inducing:
                        induced.setdefault(target, []).append(value)
            new_consts = {}
            for ref, (mode, value) in self.initial.items():
                if ref in self.const:
                    continue
                init = self._fold(value)
                if init[0] != "const":
                    raise ValueError(f"non-constant initial value of {ref}")
                if mode == "reinit":
                    if ref in induced:
                        raise ValueError(f"INDUCING writes reset attribute {ref}")
                    if ref not in written:
                        new_consts[ref] = init[1]
                elif ref not in written and all(
                    v == init for v in induced.get(ref, [])
                ):
                    new_consts[ref] = init[1]
            if not new_consts:
                break
            for ref, value in new_consts.items():
                self.const[ref] = self._typed(ref, value)
        for ref, (mode, value) in self.initial.items():
            if ref in self.const:
                continue
            if mode == "reinit":
                self.kind[ref] = "derived"
            elif ref in written:
                self.kind[ref] = "memory"
            else:
                self.kind[ref] = "state"
            if isinstance(self.domain[ref], tuple):
                raise TypeError(f"enumerated attribute {ref} is not constant")
        self.init_value = {
            ref: self._typed(ref, self._fold(value)[1])
            for ref, (mode, value) in self.initial.items()
            if ref not in self.const
        }

    # -- static single assignment over the steps
    def _ssa(self):
        """Compute, for every non-constant attribute, the expression of
        its value at the end of the phase, over the *primary* inputs
        (state attributes and memory attributes as the phase started) and
        named intermediate nodes."""
        self.nodes = {}  # name -> (expr, is_bool)
        self.node_home = {}  # name -> (obj, attr) it is named after
        env = {}
        for ref, kind in self.kind.items():
            if kind == "derived":
                env[ref] = ("const", self.init_value[ref])
            else:
                env[ref] = ("ref",) + ref  # primary input
        by_step = {}
        for rule in self.rules:
            by_step.setdefault(rule.step, []).append(rule)
        self.level_vars = {}
        for step in self.f.steps:
            rules = by_step.get(step, [])
            if not rules:
                continue
            targets = {}
            for rule in rules:
                for target, value in rule.assigns:
                    targets.setdefault(target, []).append((rule, value))
            level = set()
            for target, writes in targets.items():
                if self.kind.get(target) != "derived" or not self.is_bool(target):
                    continue
                if env[target] != ("const", self.init_value[target]):
                    continue
                if len({v for _, v in writes}) != 1:
                    continue
                value = writes[0][1]
                if value != ("const", not self.init_value[target]):
                    continue
                if any(len(r.assigns) != 1 for r, _ in writes):
                    continue
                if any(target in refs(r.cond) for r, _ in writes):
                    continue
                level.add(target)
            passes = set(targets) - level
            # Level attributes: one node each, read live within the step.
            level_node = {
                t: self._node_name(t, None if self._single_step(t) else step)
                for t in level
            }
            local = dict(env)
            for target in level:
                local[target] = ("ref", "#node", level_node[target])
            # Pass attributes: sequential passes of their rules until the
            # group has had one pass per rule plus one (checked against the
            # interpreter afterwards).
            pass_rules = [r for r in rules if any(t in passes for t, _ in r.assigns)]
            groups = self._groups(pass_rules, passes)
            for group_rules, group_vars in groups:
                current = {v: env[v] for v in group_vars}
                for iteration in range(len(group_rules) + 1):
                    for k, rule in enumerate(group_rules):
                        tag = f"{step}_{iteration}_{k}"
                        read = dict(local)
                        read.update(current)
                        cond = self._simplify(substitute(rule.cond, read))
                        cond = self._materialise(
                            cond, (rule.obj, f"cond_{rule.name}"), tag, True
                        )
                        updates = {}
                        for target, value in rule.assigns:
                            new = self._simplify(
                                ("if", cond, substitute(value, read), current[target])
                            )
                            updates[target] = self._materialise(new, target, tag)
                        current.update(updates)
                for var in group_vars:
                    local[var] = current[var]
            # Level nodes read the pass values of the same step as final.
            for target in level:
                conds = [
                    self._simplify(substitute(r.cond, local))
                    for r, _ in targets[target]
                ]
                expr = self._simplify(("or", tuple(conds)))
                if self.init_value[target]:
                    expr = self._simplify(("not", expr))
                self.nodes[level_node[target]] = (expr, True)
                self.node_home[level_node[target]] = target
            env = local
            self.level_vars[step] = level
        self.final = env

    def _node_name(self, ref, tag):
        return (ref[0], ref[1] if tag is None else f"{ref[1]}__{tag}")

    def _single_step(self, ref):
        return len({r.step for r in self.rules for t, _ in r.assigns if t == ref}) == 1

    def _simplify(self, expr):
        return minimise(expr, self._is_bool_any)

    def _is_bool_any(self, ref):
        if ref[0] == "#node":
            return self.nodes[ref[1]][1] if ref[1] in self.nodes else True
        return self.is_bool(ref)

    def _materialise(self, expr, var, tag, is_bool=None):
        if expr[0] in ("const", "ref") or size(expr) <= 12:
            return expr
        name = self._node_name(var, tag)
        self.nodes[name] = (expr, self.is_bool(var) if is_bool is None else is_bool)
        self.node_home[name] = var
        return ("ref", "#node", name)

    def _groups(self, rules, variables):
        parent = {v: v for v in variables}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        for rule in rules:
            touched = [t for t, _ in rule.assigns if t in variables]
            touched += [r for r in refs(rule.cond) if r in variables]
            for other in touched[1:]:
                parent[find(other)] = find(touched[0])
        groups = {}
        for rule in rules:
            root = find(next(t for t, _ in rule.assigns if t in variables))
            groups.setdefault(root, ([], set()))[0].append(rule)
        for var in variables:
            groups.setdefault(find(var), ([], set()))[1].add(var)
        return [(r, sorted(v)) for r, v in groups.values() if r]

    # -- concrete reference interpreter
    def interpret_phase(self, primary):
        """Run the Figaro 0 interaction phase on `primary` (a value for
        each state and memory attribute) by fixpoint iteration of the
        folded rules, step by step; return every attribute's final value."""
        values = dict(primary)
        for ref, kind in self.kind.items():
            if kind == "derived":
                values[ref] = self.init_value[ref]

        def lookup(ref):
            if ref in self.const:
                return self.const[ref]
            return values[ref]

        by_step = {}
        for rule in self.rules:
            by_step.setdefault(rule.step, []).append(rule)
        for step in self.f.steps:
            rules = by_step.get(step, [])
            for _ in range(10000):
                changed = False
                for rule in rules:
                    if evaluate(rule.cond, lookup):
                        new = [(t, evaluate(v, lookup)) for t, v in rule.assigns]
                        for target, value in new:
                            if values[target] != value:
                                values[target] = value
                                changed = True
                if not changed:
                    break
            else:
                raise RuntimeError(f"step {step} does not converge")
        return values

    def evaluate_symbolic(self, expr, primary, cache):
        def lookup(ref):
            if ref[0] == "#node":
                if ref[1] not in cache:
                    cache[ref[1]] = self.evaluate_symbolic(
                        self.nodes[ref[1]][0], primary, cache
                    )
                return cache[ref[1]]
            if ref in self.const:
                return self.const[ref]
            return primary[ref]

        return evaluate(expr, lookup)

    def self_check(self, samples=400, seed=17):
        """Compare the symbolic phase with the interpreter on random
        primary states; raise on the first disagreement."""
        rng = random.Random(seed)
        primaries = [r for r, k in self.kind.items() if k in ("state", "memory")]
        for _ in range(samples):
            primary = {}
            for ref in primaries:
                if self.is_bool(ref):
                    # Sparse failures: most leaves working, as in the chain.
                    primary[ref] = rng.random() < 0.15
                else:
                    primary[ref] = float(rng.choice([0, 1, 1, 2]))
            expected = self.interpret_phase(primary)
            cache = {}
            for ref in self.kind:
                got = self.evaluate_symbolic(self.final[ref], primary, cache)
                if got != expected[ref]:
                    raise AssertionError(
                        f"phase mismatch on {ref}: symbolic {got}, Figaro {expected[ref]}"
                    )


# ---------------------------------------------------------------- RAICHU emission


def _bool(value):
    return {"kind": "bool", "value": bool(value)}


def _value(value):
    if isinstance(value, bool):
        return _bool(value)
    return {"kind": "float", "value": float(value)}


_CMP = {"=": "eq", "<>": "ne", "<": "lt", ">": "gt", "<=": "le", ">=": "ge"}


class Emitter:
    """Write the RAICHU model of a translation."""

    def __init__(self, tr: Translation, name: str, fixed_delays: dict | None = None):
        self.tr = tr
        self.name = name
        # Figaro object -> fixed duration replacing its failure in function.
        self.fixed_delays = fixed_delays or {}
        self.attr_home = {}  # (obj, attr) primary -> component, attribute
        self.node_attr = {}  # node name -> (component, attribute)

    def expr(self, e):
        tag = e[0]
        if tag == "const":
            return {"op": "const", "value": _value(e[1])}
        if tag == "ref":
            if e[1] == "#node":
                component, attribute = self.node_attr[e[2]]
            elif (e[1], e[2]) in self.tr.const:
                return {"op": "const", "value": _value(self.tr.const[(e[1], e[2])])}
            else:
                component, attribute = e[1], e[2]
            return {
                "op": "attr",
                "attr": {"component": component, "attribute": attribute},
            }
        if tag == "not":
            return {"op": "bool", "bool_op": "not", "args": [self.expr(e[1])]}
        if tag in ("and", "or"):
            return {"op": "bool", "bool_op": tag, "args": [self.expr(a) for a in e[1]]}
        if tag == "cmp":
            return {
                "op": "cmp",
                "cmp": _CMP[e[1]],
                "lhs": self.expr(e[2]),
                "rhs": self.expr(e[3]),
            }
        if tag == "arith":
            if e[1] == "+":
                return {"op": "add", "args": [self.expr(e[2]), self.expr(e[3])]}
            if e[1] == "*":
                return {"op": "mul", "args": [self.expr(e[2]), self.expr(e[3])]}
            op = "sub" if e[1] == "-" else "div"
            return {"op": op, "lhs": self.expr(e[2]), "rhs": self.expr(e[3])}
        if tag == "if":
            return {
                "op": "if",
                "cond": self.expr(e[1]),
                "then": self.expr(e[2]),
                "otherwise": self.expr(e[3]),
            }
        raise ValueError(tag)

    def build(self) -> dict:
        tr = self.tr
        phase_end = tr.final
        target_expr = phase_end[(UNDESIRABLE_EVENT, "S")]

        # Occurrences: which are timed, which are instantaneous.
        timed, instantaneous = [], []
        for occ in tr.occurrences:
            laws = {a.law for a in occ.alternatives}
            if laws == {"EXP"}:
                if len(occ.alternatives) != 1:
                    raise ValueError(
                        f"multi-alternative timed occurrence {occ.obj}.{occ.name}"
                    )
                timed.append(occ)
            elif laws == {"INS"}:
                instantaneous.append(occ)
            else:
                raise ValueError(f"mixed laws in {occ.obj}.{occ.name}")

        def phase_expr(e):
            return tr._simplify(substitute(e, phase_end))

        def effect_expr(e):
            return tr._simplify(substitute(e, effect_view))

        # Live analysis: what the guards, the target, the INDUCING values
        # and (transitively) the memory updates read.
        roots = [target_expr]
        occ_guard = {}
        for occ in timed + instantaneous:
            occ_guard[id(occ)] = phase_expr(occ.cond)
            roots.append(occ_guard[id(occ)])
            for alt in occ.alternatives:
                roots.append(phase_expr(alt.param))
                for _, value in alt.inducing:
                    roots.append(phase_expr(value))
        live_nodes, live_primary = set(), set()
        stack = list(roots)
        live_memory = set()
        while stack:
            e = stack.pop()
            for ref in refs(e):
                if ref[0] == "#node":
                    if ref[1] not in live_nodes:
                        live_nodes.add(ref[1])
                        stack.append(tr.nodes[ref[1]][0])
                elif ref in tr.const:
                    continue
                else:
                    if ref not in live_primary:
                        live_primary.add(ref)
                        if tr.kind[ref] == "memory":
                            live_memory.add(ref)
                            stack.append(phase_end[ref])
        # Memory updates: M_start := value of M at the end of the phase.
        # Each end value an edge effect reads is a named node, so that the
        # effects of one firing all read the phase that ended, never an
        # attribute an earlier effect of the same firing already wrote.
        updates = []
        effect_view = dict(phase_end)
        for ref in sorted(live_memory):
            end = phase_end[ref]
            if end == ("ref",) + ref:
                continue
            if end[0] != "const" and not (end[0] == "ref" and end[1] == "#node"):
                node = (ref[0], f"{ref[1]}__end")
                tr.nodes[node] = (end, tr.is_bool(ref))
                live_nodes.add(node)
                end = ("ref", "#node", node)
            effect_view[ref] = end
            updates.append((ref, end))
        _check_acyclic(tr, live_nodes)

        # Components: one per Figaro object that owns something live.
        components = {}

        def component(name):
            if name not in components:
                components[name] = {
                    "name": name,
                    "attributes": [],
                    "automata": [],
                    "sensitive_functions": [],
                }
            return components[name]

        # Watcher first, so its instantaneous transition has the lowest index.
        component(FEARED)
        for ref in sorted(live_primary):
            component(ref[0])["attributes"].append(
                {
                    "name": ref[1],
                    "kind": "bool" if tr.is_bool(ref) else "float",
                    "init": _value(tr.init_value[ref]),
                }
            )
        # Named nodes that stay: the ones read by more than one live
        # expression or large; the others are inlined.
        for node in sorted(live_nodes):
            obj, attr = node
            self.node_attr[node] = (obj, attr)
        for node in sorted(live_nodes):
            obj, attr = node
            expr, is_bool = tr.nodes[node]
            home = component(obj)
            home["attributes"].append(
                {
                    "name": attr,
                    "kind": "bool" if is_bool else "float",
                    "init": _value(False if is_bool else 0.0),
                }
            )
            home["sensitive_functions"].append(
                {
                    "name": f"set_{attr}",
                    "effects": [
                        {
                            "target": {"component": obj, "attribute": attr},
                            "value": self.expr(expr),
                        }
                    ],
                }
            )
        # Order: the Figaro objects in file order after the watcher.
        order = [FEARED] + [o for o in tr.f.objects if o in components] + [PHASE]

        phase_idle = {
            "op": "state_active",
            "state": {"component": PHASE, "automaton": "phase", "state": "idle"},
        }
        phase_commit = {
            "op": "state_active",
            "state": {"component": PHASE, "automaton": "phase", "state": "commit"},
        }
        not_reached = {
            "op": "state_active",
            "state": {"component": FEARED, "automaton": "feared", "state": "ok"},
        }

        def effect(ref, value_expr):
            return {
                "target": {"component": ref[0], "attribute": ref[1]},
                "value": value_expr,
            }

        memory_effects = [effect(ref, self.expr(end)) for ref, end in updates]

        def inducing_effects(alt):
            out = []
            for target, value in alt.inducing:
                if target not in live_primary:
                    continue  # written but never read
                out.append(effect(target, self.expr(effect_expr(value))))
            return out

        kind_of = {"FAULT": "failure", "REPAIR": "repair"}
        n_timed = n_inst = 0
        for occ in timed:
            alt = occ.alternatives[0]
            rate = alt.param
            if rate[0] != "const":
                raise ValueError(f"state-dependent rate in {occ.obj}.{occ.name}")
            if rate[1] <= 0:
                continue
            law = {"distrib": "exp", "rate": rate[1]}
            if occ.obj in self.fixed_delays and alt.kind == "FAULT":
                law = {
                    "distrib": "delay",
                    "time": self.fixed_delays[occ.obj],
                    "on_interruption": "reset",
                }
            transition = {
                "name": alt.name,
                "source": alt.name,
                "targets": [alt.name],
                **law,
                "monitored": True,
                "guard": self.expr(occ_guard[id(occ)]),
                "effects": memory_effects + inducing_effects(alt),
            }
            if alt.kind in kind_of:
                transition["kind"] = kind_of[alt.kind]
            component(occ.obj)["automata"].append(
                {
                    "name": alt.name,
                    "states": [alt.name],
                    "init": alt.name,
                    "transitions": [transition],
                }
            )
            n_timed += 1
        pending_states = []
        for occ in instantaneous:
            probs = [a.param for a in occ.alternatives]
            if any(p[0] != "const" for p in probs):
                raise ValueError(f"state-dependent probability in {occ.obj}.{occ.name}")
            values = [p[1] for p in probs]
            if abs(sum(values) - 1.0) > 1e-12:
                raise ValueError(
                    f"probabilities of {occ.obj}.{occ.name} sum to {sum(values)}"
                )
            automaton = f"{occ.alternatives[0].name}_draw"
            states = ["idle"] + [a.name for a in occ.alternatives]
            draw = {
                "name": "draw",
                "source": "idle",
                "targets": [a.name for a in occ.alternatives],
                "distrib": "inst",
                "probs": values[:-1],
                "monitored": True,
                "guard": {
                    "op": "bool",
                    "bool_op": "and",
                    "args": [self.expr(occ_guard[id(occ)]), phase_idle, not_reached],
                },
            }
            if occ.alternatives[0].kind == "FAULT":
                draw["kind"] = "failure"
            transitions = [draw]
            for alt in occ.alternatives:
                if any(phase_expr(v)[0] != "const" for _, v in alt.inducing):
                    raise ValueError(
                        f"state-dependent INDUCING in {occ.obj}.{occ.name}"
                    )
                transitions.append(
                    {
                        "name": f"commit_{alt.name}",
                        "source": alt.name,
                        "targets": ["idle"],
                        "distrib": "inst",
                        "probs": [],
                        "guard": phase_commit,
                        "effects": inducing_effects(alt),
                    }
                )
                pending_states.append(
                    {
                        "op": "state_active",
                        "state": {
                            "component": occ.obj,
                            "automaton": automaton,
                            "state": alt.name,
                        },
                    }
                )
            component(occ.obj)["automata"].append(
                {
                    "name": automaton,
                    "states": states,
                    "init": "idle",
                    "transitions": transitions,
                }
            )
            n_inst += 1
        any_pending = {"op": "bool", "bool_op": "or", "args": pending_states}
        components[FEARED]["automata"].append(
            {
                "name": "feared",
                "states": ["ok", "reached"],
                "init": "ok",
                "transitions": [
                    {
                        "name": "reach",
                        "source": "ok",
                        "targets": ["reached"],
                        "distrib": "inst",
                        "probs": [],
                        "guard": {
                            "op": "bool",
                            "bool_op": "and",
                            "args": [self.expr(target_expr), phase_idle],
                        },
                    }
                ],
            }
        )
        components[PHASE] = {
            "name": PHASE,
            "automata": [
                {
                    "name": "phase",
                    "states": ["idle", "commit"],
                    "init": "idle",
                    "transitions": [
                        {
                            "name": "start_commit",
                            "source": "idle",
                            "targets": ["commit"],
                            "distrib": "inst",
                            "probs": [],
                            "guard": any_pending,
                            "effects": memory_effects,
                        },
                        {
                            "name": "end_commit",
                            "source": "commit",
                            "targets": ["idle"],
                            "distrib": "inst",
                            "probs": [],
                            "guard": {
                                "op": "bool",
                                "bool_op": "not",
                                "args": [any_pending],
                            },
                        },
                    ],
                }
            ],
        }
        body = {
            "name": self.name,
            "components": [
                {k: v for k, v in components[c].items() if v} for c in order
            ],
            "targets": [
                {
                    "name": TARGET_NAME,
                    "component": FEARED,
                    "automaton": "feared",
                    "state": "reached",
                }
            ],
        }
        self.stats = {
            "timed_transitions": n_timed,
            "instantaneous_draws": n_inst,
            "state_attributes": sum(1 for r in live_primary if tr.kind[r] == "state"),
            "memory_attributes": len(live_memory),
            "derived_attributes": len(live_nodes),
            "memory_updates_per_phase": len(updates),
        }
        return {
            "raichu_model": {"format": 1, "requires": ["transition_effects"]},
            "model": body,
        }


def _check_acyclic(tr, nodes):
    """The sensitive functions must form a DAG: their fixpoint is then the
    stepwise value, reached whatever the propagation order."""
    state = {}

    def visit(node, path):
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            raise ValueError(f"cyclic derived attributes: {path + [node]}")
        state[node] = 1
        for ref in refs(tr.nodes[node][0]):
            if ref[0] == "#node":
                visit(ref[1], path + [node])
        state[node] = 2

    for node in sorted(nodes):
        visit(node, [])


def build(
    repairable: bool = True,
    line_ccf_repair_factor: float = 1.0,
    battery: str = "erlang",
    name: str | None = None,
) -> tuple[dict, dict]:
    """Translate the benchmark into a RAICHU model document; return the
    document and a summary of the translation.

    ``repairable`` is the file's own switch; ``line_ccf_repair_factor``
    divides the mean repair time of the common-cause loss of both lines;
    ``battery`` is ``"erlang"`` (the file's two exponential stages) or
    ``"fixed"`` (two fixed half-hour stages, one hour in all)."""
    if battery not in ("erlang", "fixed"):
        raise ValueError(f"battery must be 'erlang' or 'fixed', not {battery!r}")
    fmodel = Parser(tokenize(figaro_source())).parse()
    overrides = {}
    if not repairable:
        overrides[("OPTIONS", "repairable_system")] = False
    if line_ccf_repair_factor != 1.0:
        mu = fmodel.objects["CCF_GEV_LGR"].constants["mu"][1]
        overrides[("CCF_GEV_LGR", "mu")] = evaluate(mu, {}.get) * line_ccf_repair_factor
    tr = Translation(fmodel, overrides)
    tr.self_check()
    if name is None:
        name = "eps_benchmark" + ("" if repairable else "_nonrepairable")
    fixed = {}
    if battery == "fixed":
        fixed = {stage: BATTERY_STAGE_TIME for stage in BATTERY_STAGES}
    emitter = Emitter(tr, name, fixed_delays=fixed)
    document = emitter.build()
    return document, emitter.stats


def write(name: str, out_dir: Path = MODEL_DIR) -> Path:
    """Build the variant ``name`` of :data:`VARIANTS` into ``out_dir``."""
    document, stats = build(name=name, **VARIANTS[name])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    print(path.name, stats, f"{path.stat().st_size / 1e3:.0f} kB")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=MODEL_DIR)
    parser.add_argument("variants", nargs="*", choices=list(VARIANTS))
    args = parser.parse_args()
    for name in args.variants or VARIANTS:
        write(name, args.out)


if __name__ == "__main__":
    main()
