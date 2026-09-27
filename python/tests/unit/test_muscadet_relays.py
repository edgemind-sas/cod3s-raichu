"""The muscadet layer is one package, and its old import paths still work.

`pyraichu.muscadet` holds the authoring layer (its ``__init__``), the engine
seam (`engine`), the declaration reader (`declare`) and the plugin
(`plugin`). The three modules that used to live elsewhere keep a relay at
their old path for one release:

======================================  ==============================
old path                                new path
======================================  ==============================
``pyraichu.muscadet_engine``            ``pyraichu.muscadet.engine``
``pyraichu.declare``                    ``pyraichu.muscadet.declare``
``pyraichu.plugins.muscadet``           ``pyraichu.muscadet.plugin``
======================================  ==============================

A relay is not a copy: the old name is bound to the very module object of the
new one, so identity, ``isinstance``, private names and module state are
shared. Importing it warns once, at the importer's line.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
import types
import warnings

import pytest

import pyraichu
import pyraichu.muscadet as authoring
import pyraichu.muscadet.declare as declare
import pyraichu.muscadet.engine as engine
import pyraichu.muscadet.plugin as plugin
from pyraichu.plugins import PLUGINS, expand_model
from test_explore import _pair

#: (old path, new path, parent package, attribute on the parent).
RELAYS = [
    ("pyraichu.muscadet_engine", "pyraichu.muscadet.engine", "pyraichu", "muscadet_engine"),
    ("pyraichu.declare", "pyraichu.muscadet.declare", "pyraichu", "declare"),
    (
        "pyraichu.plugins.muscadet",
        "pyraichu.muscadet.plugin",
        "pyraichu.plugins",
        "muscadet",
    ),
]

RELAY_IDS = [old for old, *_ in RELAYS]


@pytest.fixture
def fresh_relay():
    """Forget one relay, so the next import of it runs its body again.

    The import system caches the aliased module under the old name, which is
    what makes the warning fire once per process; a test of that warning has
    to start from a process that has not imported it yet.
    """
    forgotten = []

    def forget(old: str, parent: str, attribute: str) -> None:
        saved_module = sys.modules.pop(old, None)
        parent_module = sys.modules[parent]
        saved_attribute = parent_module.__dict__.pop(attribute, None)
        forgotten.append((old, saved_module, parent_module, attribute, saved_attribute))

    yield forget
    for old, saved_module, parent_module, attribute, saved_attribute in forgotten:
        if saved_module is not None:
            sys.modules[old] = saved_module
        if saved_attribute is not None:
            setattr(parent_module, attribute, saved_attribute)


def _deprecations(caught) -> list[warnings.WarningMessage]:
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


# --- 1. one module object, one warning ------------------------------------


@pytest.mark.parametrize(("old", "new", "parent", "attribute"), RELAYS, ids=RELAY_IDS)
def test_an_old_path_is_the_new_module_and_warns_once(fresh_relay, old, new, parent, attribute):
    fresh_relay(old, parent, attribute)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = importlib.import_module(old)
        second = importlib.import_module(old)

    assert first is importlib.import_module(new)
    assert second is first
    assert sys.modules[old] is sys.modules[new]
    assert getattr(sys.modules[parent], attribute) is first

    [warning] = _deprecations(caught)
    assert f"`{old}`" in str(warning.message)
    assert f"`{new}`" in str(warning.message)
    # Attributed to the code that imported the old path, not to the relay
    # or the import system, so a test run names the line to change.
    assert warning.filename == __file__


@pytest.mark.parametrize(("old", "new", "parent", "attribute"), RELAYS, ids=RELAY_IDS)
def test_every_import_form_reaches_the_same_module(fresh_relay, old, new, parent, attribute):
    """``from parent import name`` and a bare attribute read go through the
    relay too: both used to find the module the eager imports left bound."""
    fresh_relay(old, parent, attribute)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        through_from = getattr(__import__(parent, fromlist=[attribute]), attribute)
    assert through_from is sys.modules[new]
    assert len(_deprecations(caught)) == 1

    fresh_relay(old, parent, attribute)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        through_attribute = getattr(sys.modules[parent], attribute)
    assert through_attribute is sys.modules[new]
    [warning] = _deprecations(caught)
    assert warning.filename == __file__


def test_an_unknown_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match="no_such_module"):
        pyraichu.no_such_module  # noqa: B018
    with pytest.raises(AttributeError, match="no_such_plugin"):
        pyraichu.plugins.no_such_plugin  # noqa: B018


def test_the_new_paths_import_without_a_warning():
    """What the repository itself imports must not warn: the probe runs in a
    fresh interpreter with deprecations turned into errors."""
    probe = (
        "import pyraichu, pyraichu.muscadet, pyraichu.muscadet.engine,"
        " pyraichu.muscadet.declare, pyraichu.muscadet.plugin, pyraichu.plugins,"
        " pyraichu.mode_objects;"
        "pyraichu.expand_model({'name': 'm', 'components': []})"
    )
    subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c", probe], check=True)


# --- 2. private names and names resolved by string ------------------------


@pytest.mark.parametrize(
    ("old", "name"),
    [
        ("pyraichu.muscadet_engine", "_merge_indicators"),
        ("pyraichu.declare", "_refuse_a_latched_production_a_condition_also_writes"),
        ("pyraichu.plugins.muscadet", "_carry_grafts"),
    ],
)
def test_private_names_resolve_through_the_old_path(old, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        relayed = importlib.import_module(old)
    assert getattr(relayed, name) is getattr(sys.modules[old], name)
    assert getattr(relayed, name).__module__ in {engine.__name__, declare.__name__, plugin.__name__}


def test_the_authoring_layer_keeps_its_names():
    """The package ``__init__`` IS the authoring module moved as is: its
    public and private names, and the module its classes report."""
    for name in ("System", "ObjFlow", "_Capacity", "_matches_flow", "_CMP_OPS"):
        assert hasattr(authoring, name), name
    assert authoring.System.__module__ == "pyraichu.muscadet"
    assert authoring.ObjFlow.__module__ == "pyraichu.muscadet"


def test_a_class_resolved_by_its_dotted_name():
    """cod3s selects the RAICHU system class by name
    (``get_class_by_name("pyraichu.muscadet.System")``): split on the last
    dot, import the module, read the attribute."""
    module_name, class_name = "pyraichu.muscadet.System".rsplit(".", 1)
    resolved = getattr(importlib.import_module(module_name), class_name)
    assert resolved is authoring.System
    # And the name the class reports resolves back to it, which is what a
    # pickle of a system relies on.
    reported = f"{resolved.__module__}.{resolved.__qualname__}"
    assert reported == "pyraichu.muscadet.System"


def test_the_platform_runner_surface_is_there():
    """The names the platform's launcher stub mirrors, read through the paths
    the launcher imports today."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from pyraichu import declare as old_declare
        from pyraichu import muscadet_engine as old_engine
    for name in ("__version__", "ModelError", "SimulationError", "monte_carlo",
                 "run_sequences", "load_model", "model_body", "seal"):
        assert hasattr(pyraichu, name), name
    assert old_engine.build_model is engine.build_model
    assert old_declare.event_automaton is declare.event_automaton


# --- 3. the entry point and the plugin registry ---------------------------


def _advertised_entry_point() -> importlib.metadata.EntryPoint:
    try:
        distribution = importlib.metadata.distribution("pyraichu")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("pyraichu is on the path but not installed as a distribution")
    [entry] = [e for e in distribution.entry_points if e.group == "muscadet.engines"]
    return entry


def _registering_with_a_stand_in(hook) -> dict:
    registered: dict = {}
    stand_in = types.ModuleType("muscadet")
    stand_in.register_engine = lambda **kwargs: registered.update(kwargs)
    previous = sys.modules.get("muscadet")
    sys.modules["muscadet"] = stand_in
    try:
        hook()
    finally:
        if previous is None:
            del sys.modules["muscadet"]
        else:
            sys.modules["muscadet"] = previous
    return registered


def test_the_entry_point_loads_the_new_path_and_registers():
    entry = _advertised_entry_point()
    assert entry.value == "pyraichu.muscadet.engine:register"
    hook = entry.load()
    assert hook is engine.register
    registered = _registering_with_a_stand_in(hook)
    assert registered["name"] == engine.ENGINE_NAME == entry.name
    assert registered["simulate"] is engine.simulate


def test_the_old_entry_point_value_still_loads():
    """Metadata written before the move (an older install's
    ``entry_points.txt``) names the old path; it resolves to the same hook."""
    stale = importlib.metadata.EntryPoint(
        name="raichu", value="pyraichu.muscadet_engine:register", group="muscadet.engines"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert stale.load() is engine.register


def test_the_plugin_registry_key_is_unchanged():
    assert isinstance(PLUGINS["muscadet"], plugin.MuscadetPlugin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from pyraichu.plugins import muscadet as old_plugin
    assert old_plugin.MuscadetPlugin is plugin.MuscadetPlugin


#: Import orders that load the plugin by a different first module. The
#: plugin module and `pyraichu.plugins` import each other, so the registry
#: entry must be written whichever of them is imported first.
IMPORT_ORDERS = {
    "registry first": "import pyraichu.plugins",
    "plugin first": "import pyraichu.muscadet.plugin",
    "declaration reader first": "import pyraichu.muscadet.declare",
    "engine seam first": "import pyraichu.muscadet.engine",
    "old plugin path first": "import pyraichu.plugins.muscadet",
    "package only": "import pyraichu",
}


def _expansion_digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("first_import", IMPORT_ORDERS.values(), ids=IMPORT_ORDERS.keys())
def test_the_expansion_does_not_depend_on_the_import_order(first_import):
    """Byte-identical expansion of a muscadet document in a fresh
    interpreter, whatever module was imported first."""
    document = json.dumps(_pair())
    probe = (
        "import warnings; warnings.simplefilter('ignore', DeprecationWarning)\n"
        f"{first_import}\n"
        "import hashlib, json, sys, pyraichu\n"
        "from pyraichu.plugins import PLUGINS\n"
        "assert type(PLUGINS['muscadet']).__module__ == 'pyraichu.muscadet.plugin'\n"
        "expanded = pyraichu.expand_model(json.loads(sys.stdin.read()))\n"
        "print(hashlib.sha256(json.dumps(expanded, sort_keys=True).encode()).hexdigest())\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], input=document, capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == _expansion_digest(expand_model(_pair()))
