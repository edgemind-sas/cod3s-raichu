"""Deprecated: this module moved to :mod:`pyraichu.muscadet.declare`.

Kept for one release so existing imports keep working. Importing it emits a
``DeprecationWarning`` once and binds ``pyraichu.declare`` to the very module
object ``pyraichu.muscadet.declare``, so identity, private names and module state are shared.
"""

from ._relay import alias

alias(__name__, "pyraichu.muscadet.declare")
