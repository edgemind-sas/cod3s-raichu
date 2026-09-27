"""Deprecated: this module moved to :mod:`pyraichu.muscadet.plugin`.

Kept for one release so existing imports keep working. Importing it emits a
``DeprecationWarning`` once and binds ``pyraichu.plugins.muscadet`` to the very module
object ``pyraichu.muscadet.plugin``, so identity, private names and module state are shared.
"""

from .._relay import alias

alias(__name__, "pyraichu.muscadet.plugin")
