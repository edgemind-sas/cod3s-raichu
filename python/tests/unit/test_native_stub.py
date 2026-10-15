"""The type stub of the native module matches what the module accepts.

`_pyraichu.pyi` is written by hand beside the PyO3 signatures, so it can
drift: `monte_carlo_json` gained `allow_fmu_import`, `fmu_base_dir` and
`require_parallel` without its stub entry, and a type checker then flagged
the valid calls `pyraichu.monte_carlo` makes. Every function the stub
declares is compared here, parameter by parameter, with the signature the
compiled module reports.
"""

import ast
import inspect
from pathlib import Path

import pytest

from pyraichu import _pyraichu

STUB = Path(_pyraichu.__file__).with_name("_pyraichu.pyi")
if not STUB.exists():  # an installed wheel lays the stub beside the package
    STUB = Path(__file__).resolve().parents[3] / "bindings/pyraichu/python/pyraichu/_pyraichu.pyi"

_MISSING = object()


def stub_functions() -> dict[str, list[tuple[str, object]]]:
    tree = ast.parse(STUB.read_text())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        positional = node.args.posonlyargs + node.args.args
        defaults = [_MISSING] * (len(positional) - len(node.args.defaults)) + [
            ast.literal_eval(d) for d in node.args.defaults
        ]
        defaults += [
            _MISSING if d is None else ast.literal_eval(d) for d in node.args.kw_defaults
        ]
        out[node.name] = [(a.arg, d) for a, d in zip(args, defaults)]
    return out


STUBBED = stub_functions()


def test_the_stub_declares_functions():
    assert "monte_carlo_json" in STUBBED and "simulate_json" in STUBBED


@pytest.mark.parametrize("name", sorted(STUBBED))
def test_a_stubbed_function_matches_the_native_signature(name):
    native = inspect.signature(getattr(_pyraichu, name))
    got = [
        (p.name, _MISSING if p.default is inspect.Parameter.empty else p.default)
        for p in native.parameters.values()
    ]
    assert STUBBED[name] == got
