"""Unsupported-Yielding vs. Rational-Updating: a two-turn diagnostic.

From "Sycophancy Suppression Can Impair Rational Updating: Anti-Sycophancy
Should Preserve the Ability to Update" (Findings of EMNLP 2026).
"""
import sys

if sys.version_info < (3, 9):  # pragma: no cover - the interpreter is too old
    raise SystemExit(
        f"sru needs Python 3.9 or newer; this is {sys.version.split()[0]}. "
        "On a machine whose default python3 is older, run the tool from an "
        "environment with a newer one, for example a venv created by that "
        "interpreter.")

__version__ = "1.0.0"
__all__ = ["__version__"]
