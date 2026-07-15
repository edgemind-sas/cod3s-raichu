"""Plugin system: specialized object schemas over the RAICHU core.

A model file may carry a ``"plugins"`` section whose objects follow a
plugin-specific specification schema; :func:`expand_model` translates
them **deterministically** into ordinary core-model material
(components, connections, indicators) before validation. The expansion
is pure data-to-data — auditable, reproducible, serializable.

Registering a plugin::

    from pyraichu.plugins import PLUGINS
    PLUGINS["my_domain"] = MyDomainPlugin()

A plugin implements ``expand_object(spec: dict, model: dict) ->
(components, connections, indicators)`` where the returned lists are
core-schema fragments appended to the model.
"""

from __future__ import annotations

import copy
from typing import Any, Protocol

__all__ = ["PLUGINS", "expand_model", "Plugin"]


class Plugin(Protocol):
    """Protocol of a plugin: translate one specialized object into core
    model fragments."""

    def expand_object(
        self, spec: dict[str, Any], model: dict[str, Any]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Return ``(components, connections, indicators)`` fragments."""
        ...  # pragma: no cover


PLUGINS: dict[str, Plugin] = {}


def expand_model(model: dict[str, Any]) -> dict[str, Any]:
    """Expand every plugin object of ``model`` into core material.

    Returns a new core-schema dict (the input is not mutated); a model
    without a ``"plugins"`` section is returned unchanged (deep-copied).
    Raises ``KeyError`` for an unknown plugin and lets plugin-specific
    errors propagate with their context.
    """
    model = copy.deepcopy(model)
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
            components, connections, indicators = plugin.expand_object(spec, model)
            model["components"].extend(components)
            model["connections"].extend(connections)
            model["indicators"].extend(indicators)
    return model


# Built-in plugins.
from . import muscadet as _muscadet  # noqa: E402  (registry side effect)

PLUGINS["muscadet"] = _muscadet.MuscadetPlugin()
