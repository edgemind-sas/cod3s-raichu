"""The published examples, written as component type libraries.

Each example page builds its system component by component in the flat
core schema. Its library under ``examples/<name>/library/`` describes the
same system from reusable types. A library reaches its neighbours through
ports where the flat model reads them by name, so the two documents differ;
these tests prove that they behave identically: same events at the same
dates, same firing counts, same indicator values at every sample instant,
for every seed tried (`library_examples_support.record`).

A catalogued failure mode renames its transitions (``failure.occ`` for
``health.fail``); the comparison translates them back with a rename map.
The documents are checked against the published schemas by the Rust
schema tests (``crates/raichu-library/tests/schema.rs``), which validate
nested expressions in milliseconds where the Python validator takes
tens of seconds.
"""

from __future__ import annotations

import json

import pyraichu
import pytest
from library_examples_support import (
    EXAMPLES,
    differences,
    example_module,
    library_model,
    page_namespace,
    record,
)

SEEDS = range(5)


def assert_same_behaviour(flat, library, *, t_max, instants, seeds=SEEDS, rename=None, **options):
    for seed in seeds:
        expected = record(flat, t_max=t_max, instants=instants, seed=seed, **options)
        observed = record(
            library, t_max=t_max, instants=instants, seed=seed, rename=rename, **options
        )
        lines = differences(expected, observed)
        assert not lines, f"seed {seed}:\n" + "\n".join(lines)


# --- two out of three ------------------------------------------------------

TWO_OUT_OF_THREE_RENAME = {
    f"{pump}.failure.{mode}": f"{pump}.health.{flat}"
    for pump in ("P1", "P2", "P3")
    for mode, flat in (("occ", "fail"), ("rep", "repair"))
}


@pytest.fixture(scope="module")
def two_out_of_three():
    flat = page_namespace("examples/two-out-of-three.md")["model"]
    library = library_model("two_out_of_three", study="study.json")
    return flat, library


def test_two_out_of_three_behaves_as_its_page_model(two_out_of_three):
    flat, library = two_out_of_three
    assert_same_behaviour(
        flat,
        library,
        t_max=1000.0,
        instants=[100.0 * k for k in range(11)],
        rename=TWO_OUT_OF_THREE_RENAME,
    )


def test_two_out_of_three_explores_to_the_same_bounds(two_out_of_three):
    flat, library = two_out_of_three
    options = {"horizon": 100.0, "min_probability": 1e-12, "max_length": 30}
    expected = pyraichu.explore(flat, "system_down", **options)
    observed = pyraichu.explore(library, "system_down", **options)
    assert (observed.lower, observed.upper) == (expected.lower, expected.upper)
    assert len(observed.sequences) == len(expected.sequences)


# --- heated room -----------------------------------------------------------


def test_heated_room_behaves_as_its_page_model():
    flat = page_namespace("examples/heated-room.md")["model"]
    library = library_model("heated_room")
    assert_same_behaviour(
        flat, library, t_max=24.0 * 30, instants=[24.0 * d for d in range(31)], seeds=[12, 0, 1, 2, 3]
    )


# --- heated tank -----------------------------------------------------------


def test_heated_tank_behaves_as_its_page_model():
    # The failure rates read the temperature through a port inside a
    # state-dependent rate: the cumulative hazard must follow it as it
    # follows the direct read.
    flat = page_namespace("examples/heated-tank.md")["model"]
    library = library_model("heated_tank")
    assert_same_behaviour(
        flat, library, t_max=1000.0, instants=[100.0 * k for k in range(11)], seeds=[3, 0, 1, 2, 4]
    )


# --- hydrogen installation -------------------------------------------------


@pytest.fixture(scope="module")
def h2_installation():
    return page_namespace("examples/h2-installation.md")["model"], library_model("h2_installation")


def test_h2_installation_behaves_as_its_page_model(h2_installation):
    # A quarter, sampled daily: the explicit equations run in the order
    # the expander derives, the page model in its own, and every value
    # must agree at every instant.
    flat, library = h2_installation
    assert_same_behaviour(
        flat, library, t_max=2190.0, instants=[24.0 * d for d in range(92)], seeds=[3, 0, 1, 2, 4]
    )


def test_h2_installation_keeps_the_same_cycle_groups(h2_installation):
    flat, library = h2_installation

    def decisions(model):
        return sorted(
            (group["component"], tuple(sorted((d["cancellable"], d.get("memory")) for d in group["targets"])))
            for group in pyraichu.cycle_groups(model)
        )

    assert decisions(library) == decisions(flat)


# --- gas production --------------------------------------------------------


@pytest.mark.parametrize("case", [1, 2, 3, 4])
def test_gas_production_behaves_as_its_flat_model(case):
    # Cases 1 to 3: a unit's repair refills the reservoir by writing its
    # level through a port; case 4 refills it at the units' maximum.
    flat = pyraichu.load_model(example_module("gas_production").build(case))
    library = library_model("gas_production", system=f"system_case{case}.json")
    assert_same_behaviour(
        flat,
        library,
        t_max=1e5,
        instants=[1e4 * k for k in range(11)],
        seeds=range(10),
        # The published campaigns bound the step as well: a long linear
        # horizon otherwise takes steps the size of the horizon's tail.
        max_step=100.0,
    )


def test_the_committed_gas_production_library_is_the_generated_one():
    generated = example_module("gas_production", "library").documents()
    folder = EXAMPLES / "gas_production" / "library"
    assert sorted(p.name for p in folder.glob("*.json")) == sorted(generated)
    for name, document in generated.items():
        assert json.loads((folder / name).read_text()) == document, name


# --- APPRODYN --------------------------------------------------------------

APPRODYN_VARIANTS = [
    ("approdyn_pdmp", "pdmp", True, False),
    ("approdyn_pdmp_noccf", "pdmp", False, False),
    ("approdyn_tables", "tables", True, False),
    ("approdyn_pdmp_perfect_tpa", "pdmp", True, True),
]


@pytest.mark.parametrize("variant, rule, ccf, perfect", APPRODYN_VARIANTS)
def test_approdyn_behaves_as_its_flat_model(variant, rule, ccf, perfect):
    # Dense in simultaneous instantaneous transitions with on-demand draws:
    # any change in which transition fires first at a date shifts every
    # later draw, so many seeds are compared event by event.
    approdyn = example_module("approdyn")
    flat = pyraichu.load_model(approdyn.build(rule, ccf, variant, perfect))
    library = library_model("approdyn", system=f"system_{variant}.json")
    assert_same_behaviour(
        flat,
        library,
        t_max=approdyn.HORIZON,
        instants=[1000.0 * k for k in range(14)],
        seeds=range(20),
    )


def test_the_committed_approdyn_library_is_the_generated_one():
    generated = example_module("approdyn", "library").documents()
    folder = EXAMPLES / "approdyn" / "library"
    assert sorted(p.name for p in folder.glob("*.json")) == sorted(generated)
    for name, document in generated.items():
        assert json.loads((folder / name).read_text()) == document, name
