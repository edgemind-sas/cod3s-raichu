"""Plugin system: specialized object schemas over the RAICHU core.

A model file may carry a ``"plugins"`` section whose objects follow a
plugin-specific specification schema; :func:`expand_model` translates
them **deterministically** into ordinary core-model material
(components, connections, indicators) before validation. The expansion
is pure data-to-data: auditable, reproducible, serializable.

Registering a plugin::

    from pyraichu.plugins import PLUGINS
    PLUGINS["my_domain"] = MyDomainPlugin()

A plugin implements ``expand_object(spec: dict, model: dict) ->
(components, connections, indicators)`` where the returned lists are
core-schema fragments appended to the model. A plugin that also derives
a **model-wide** property (the evaluation order, first of them) returns
a fourth element: a dict of model-level keys, listed in
``MODEL_LEVEL_KEYS``. Three-element returns stay valid.
"""

from __future__ import annotations

import copy
from typing import Any, Protocol

from .._pyraichu import MODEL_ENVELOPE_KEY

__all__ = ["MODEL_LEVEL_KEYS", "PLUGINS", "expand_model", "Plugin"]


class Plugin(Protocol):
    """Protocol of a plugin: translate one specialized object into core
    model fragments."""

    def expand_object(
        self, spec: dict[str, Any], model: dict[str, Any]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Return ``(components, connections, indicators)`` fragments,
        optionally followed by a fourth element: a dict of **model-level**
        keys to set on the expanded model (see :func:`expand_model`)."""
        ...  # pragma: no cover


PLUGINS: dict[str, Plugin] = {}

#: Model-level keys a plugin may set through the fourth element of its
#: ``expand_object`` return. They are model-wide properties, so two
#: plugins setting the same one is a contradiction, refused rather than
#: resolved by declaration order.
MODEL_LEVEL_KEYS = frozenset({"evaluation_order"})


def expand_model(model: dict[str, Any]) -> dict[str, Any]:
    """Expand every plugin object of ``model`` into core material.

    Accepts a document in either shape (bare body, or body under the
    format envelope) and returns a new core-schema **body** (the input is
    not mutated); a model without a ``"plugins"`` section is returned
    unchanged (deep-copied). Raises ``KeyError`` for an unknown plugin
    and lets plugin-specific errors propagate with their context.

    Besides the three fragment lists, a plugin may return a fourth
    element: model-level keys (``MODEL_LEVEL_KEYS``) it derives for the
    whole model, the evaluation order being the first of them. Those are
    not lists to extend but single values, so a second plugin setting one
    already set is refused.
    """
    model = copy.deepcopy(model)
    if MODEL_ENVELOPE_KEY in model:
        model = model["model"]
    plugins_section = model.pop("plugins", None)
    if not plugins_section:
        return model

    model.setdefault("components", [])
    model.setdefault("connections", [])
    model.setdefault("indicators", [])
    model.setdefault("targets", [])
    for plugin_name, payload in plugins_section.items():
        plugin = PLUGINS.get(plugin_name)
        if plugin is None:
            raise KeyError(
                f"unknown plugin `{plugin_name}` (registered: "
                f"{sorted(PLUGINS)})"
            )
        for spec in payload.get("objects", []):
            fragments = plugin.expand_object(spec, model)
            components, connections, indicators = fragments[:3]
            model["components"].extend(components)
            model["connections"].extend(connections)
            model["indicators"].extend(indicators)
            updates = fragments[3] if len(fragments) > 3 else None
            _apply_model_level(model, plugin_name, updates)
    return model


def _apply_model_level(
    model: dict[str, Any], plugin_name: str, updates: dict[str, Any] | None
) -> None:
    """Set the model-level keys a plugin derived, refusing an unknown key
    and a second writer of the same one."""
    for key, value in (updates or {}).items():
        if key not in MODEL_LEVEL_KEYS:
            raise KeyError(
                f"plugin `{plugin_name}` sets unknown model-level key "
                f"`{key}` (allowed: {sorted(MODEL_LEVEL_KEYS)})"
            )
        if key in model and model[key] != value:
            raise ValueError(
                f"plugin `{plugin_name}` sets model-level key `{key}`, "
                f"which is already set to a different value: a model-wide "
                f"property has one writer"
            )
        model[key] = value


# Built-in plugins.
from . import muscadet as _muscadet  # noqa: E402  (registry side effect)

PLUGINS["muscadet"] = _muscadet.MuscadetPlugin()
