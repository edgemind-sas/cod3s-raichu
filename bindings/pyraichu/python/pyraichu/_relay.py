"""Deprecated import paths kept working for one release.

A module that moved leaves a relay at its old path. The relay does not copy
the names it used to hold: it replaces itself in :data:`sys.modules` with the
module that moved, so the old path and the new one are ONE module object.
Identity, ``isinstance`` checks, private names and module-level state are
shared, and a monkeypatch through either path is seen through the other.

The warning is emitted once, when the old path is first imported: the import
system caches the aliased module, so the relay's body never runs again.
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from types import FrameType, ModuleType

__all__ = ["alias"]

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
_IMPORTLIB_DIR = os.path.dirname(os.path.abspath(importlib.__file__)) + os.sep


def alias(old: str, new: str) -> ModuleType:
    """Bind the module at ``new`` under the name ``old`` and warn once.

    Called from the body of the relay module ``old``. The warning is
    attributed to the code that imported the old path, whichever route the
    import took (``import``, ``from ... import``, :func:`importlib.import_module`
    or an attribute of the parent package), so the default filters show it
    to a script and a test run names the line to change.
    """
    module = importlib.import_module(new)
    sys.modules[old] = module
    message = (
        f"`{old}` is deprecated and will be removed in a future release: "
        f"import `{new}` instead"
    )
    frame = _importer_frame()
    if frame is None:  # pragma: no cover (no caller outside the import system)
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return module
    module_globals = frame.f_globals
    warnings.warn_explicit(
        message,
        DeprecationWarning,
        frame.f_code.co_filename,
        frame.f_lineno,
        module=module_globals.get("__name__", "<unknown>"),
        registry=module_globals.setdefault("__warningregistry__", {}),
    )
    return module


def _importer_frame() -> FrameType | None:
    """The first frame outside the import system (the frozen bootstrap and the
    standard library's ``importlib``) and outside pyraichu."""
    frame = sys._getframe(2)
    while frame is not None:
        filename = frame.f_code.co_filename
        path = os.path.abspath(filename)
        import_system = filename.startswith("<frozen importlib") or path.startswith(
            _IMPORTLIB_DIR
        )
        if import_system or path.startswith(_PACKAGE_DIR):
            frame = frame.f_back
            continue
        return frame
    return None
