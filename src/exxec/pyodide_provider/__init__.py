"""Pyodide execution environment that runs Python in WASM via Deno."""

from __future__ import annotations

from exxec.pyodide_provider.filesystem import PyodideFS
from exxec.pyodide_provider.provider import PyodideExecutionEnvironment

__all__ = ["PyodideExecutionEnvironment", "PyodideFS"]
