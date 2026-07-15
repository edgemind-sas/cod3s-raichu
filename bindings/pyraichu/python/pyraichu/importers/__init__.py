"""Importers: translate external model descriptions into RAICHU models.

`cod3s_platform` consumes the COD3S-platform artefacts (the model export
JSON + the study YAML dict) and emits a RAICHU plugin-spec model — the thin
`platform-export → core JSON → pyraichu` path (no muscadet/PyCATSHOO
dependency).
"""

from .cod3s_platform import (
    Translation,
    TranslationError,
    translate,
    translate_export,
    translate_study,
)

__all__ = [
    "Translation",
    "TranslationError",
    "translate",
    "translate_export",
    "translate_study",
]
