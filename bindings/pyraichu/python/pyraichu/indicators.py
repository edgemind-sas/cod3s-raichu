"""One rule for putting an indicator into a model body.

More than one writer adds indicators to the same document, and they do not
all know about each other. A document may DECLARE indicators of its own; a
plugin emits some while it expands an object; the muscadet authoring layer
emits one per observable variable it generated, whose name is
``{component}_{variable}`` -- muscadet's own convention, which is exactly the
name a modeller writing a declaration by hand reaches for. Two writers
naming one observation is therefore the normal case, not the exceptional
one.

The engine refuses two indicators of one name, and refuses them rightly: two
observations under a single name is a result whose reader cannot tell which
one it is. That refusal is a floor, not a merge policy, and it names the
indicator rather than the writer that recopied it. So the reconciliation
belongs HERE, above it, where both the entry already present and the entry
being added are still in hand:

* a name nobody holds is **added**;
* a name held by the SAME observation is **one indicator**, and the second
  writer adds nothing: this is the ordinary case of a declaration asking for
  what the layer was going to emit anyway;
* a name held by a DIFFERENT observation is **refused**, by the caller's own
  exception and in the caller's own vocabulary, which is what the
  `collision` argument is for. A refusal here can say which two writers
  disagree; the engine's cannot.

Compared on the WHOLE entry rather than on its ``attr``: a state indicator
carries no ``attr`` at all, and entries compared on a key they have not
would read as equal to every other one of their kind.

Compared in ONE canonical form, though: a threshold that is the identity on
a boolean (``eq true``, ``ne false``) observes the boolean itself, and the
engine returns the same series for it as for the plain ``attribute`` entry,
point for point. cod3s gives every variable indicator a threshold pair and a
platform study often spells ``== True``, so that spelling meets the
generated ``attribute`` entry of the same name as a matter of course
(measured 2026-09-25, the internal instance's "MC DC P2"). Any other
threshold is a different quantity and still collides.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

__all__ = [
    "GENERATED_INDICATORS",
    "generated_indicators",
    "merge_indicators",
    "observation",
]


#: The model-level key by which a model declares that it wants the indicator
#: set this layer GENERATES -- one per observable variable, named
#: ``{component}_{variable}`` -- beside the ones it declares itself.
#:
#: Spelled after ``indicators``, the key that carries the declared ones, and
#: in the model-level style of the format: a flat snake_case noun naming a
#: property of the whole model, as ``evaluation_order`` and ``targets`` are.
#: It says what the document holds rather than what a layer should do, which
#: is why it is not ``generate_indicators``: no key of this format is an
#: instruction.
#:
#: **Absent means false**, and false means the model observes what it
#: declared and nothing else. That is the reading which changes nothing for
#: the corpus of boolean studies: a model asking for the generated set says
#: so. Every writer of a model document states it explicitly all the same
#: (:meth:`pyraichu.muscadet.System.build_dict`), so the two authoring
#: surfaces write one document rather than one document and its default.
GENERATED_INDICATORS = "generated_indicators"


def generated_indicators(
    model: Mapping[str, Any], *, refuse: Callable[[str], Exception] = ValueError
) -> bool:
    """Whether `model` asks for the generated indicator set.

    The one reading of :data:`GENERATED_INDICATORS`, shared by the three
    writers that honour it -- the plugin expansion, the declaration reader and
    the class-based builder -- because what a model observes must not depend
    on which of them assembled it. That is the divergence this key closed: the
    presence of a continuous construct somewhere in the model used to decide,
    on one route only, what every component of it was observed by.

    `refuse` builds the exception, so each reader refuses in its own
    vocabulary; only a genuine boolean is honoured, a string ``"false"``
    being true to Python and false to whoever wrote it.
    """
    value = model.get(GENERATED_INDICATORS, False)
    if not isinstance(value, bool):
        raise refuse(
            f"`{GENERATED_INDICATORS}` says whether the model wants the "
            f"indicator set this layer generates beside the declared ones, so "
            f"it is true or false; got {value!r}"
        )
    return value


def observation(entry: Mapping[str, Any]) -> dict[str, Any]:
    """What an indicator entry observes: the whole entry but its name.

    The name is the handle a reader looks an estimate up by; everything
    else is what the estimate is OF. Two entries with one name and one
    observation are one indicator written twice.
    """
    return {key: value for key, value in entry.items() if key != "name"}


def _canonical(entry: Mapping[str, Any]) -> dict[str, Any]:
    """`observation(entry)`, with the identity threshold on a boolean read
    as the attribute it tests.

    Only for comparing: what a caller reads and what the document carries
    stay as written. A boolean constant can only be compared with a boolean
    attribute (the engine refuses anything else at build time), so the
    threshold's own value is enough to know the attribute is one.
    """
    seen = observation(entry)
    value = seen.get("value")
    if (
        seen.get("target") == "predicate"
        and isinstance(value, Mapping)
        and value.get("kind") == "bool"
        and (seen.get("cmp"), value.get("value")) in {("eq", True), ("ne", False)}
        and set(seen) == {"target", "attr", "cmp", "value"}
    ):
        return {"target": "attribute", "attr": seen["attr"]}
    return seen


def merge_indicators(
    indicators: list[dict[str, Any]],
    incoming: Iterable[Mapping[str, Any]],
    *,
    collision: Callable[[str, dict[str, Any], dict[str, Any]], Exception],
) -> list[dict[str, Any]]:
    """Add `incoming` to `indicators`, in place, at most one entry per name.

    Parameters
    ----------
    indicators : list of dict
        The indicators the body already carries. Extended in place and
        returned, so a caller may write ``body["indicators"] =
        merge_indicators(body.get("indicators") or [], ...)`` whether or not
        the key was there.
    incoming : iterable of mapping
        The entries this writer wants observed. Consumed lazily: an entry
        is built, then reconciled, so a generator of translations is a
        valid argument.
    collision : callable
        ``collision(name, incoming_observation, existing_observation)``
        answers the exception raised when one name covers two observations.
        The caller builds it because the caller is the only one who knows
        which two writers disagreed, and naming them is the whole point of
        refusing here rather than letting the engine refuse the document.

    Nothing already in `indicators` is ever dropped or rewritten. That is
    the property the "replace rather than append" shortcut loses: an
    indicator a modeller renamed, or one on a variable the emitting layer
    generates without observing, has no counterpart among `incoming` and
    would vanish without a word.
    """
    held = {entry["name"]: entry for entry in indicators if "name" in entry}
    for entry in incoming:
        # An entry with no name at all is carried through rather than made
        # into a `KeyError` here: it is a malformed document, and the engine
        # refuses it by name with a message written for that.
        name = entry.get("name")
        already = held.get(name) if name is not None else None
        if already is None:
            entry = dict(entry)
            indicators.append(entry)
            if name is not None:
                held[name] = entry
            continue
        if _canonical(already) != _canonical(entry):
            raise collision(name, observation(entry), observation(already))
    return indicators
