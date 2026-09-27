"""Deprecated: this module moved to :mod:`pyraichu.muscadet.engine`.

Kept for one release so existing imports keep working. Importing it emits a
``DeprecationWarning`` once and binds ``pyraichu.muscadet_engine`` to the very module
object ``pyraichu.muscadet.engine``, so identity, private names and module state are shared.
"""

from ._relay import alias

alias(__name__, "pyraichu.muscadet.engine")
