"""Local execution environment that runs code in the same process."""

from __future__ import annotations

from exxec.local_provider.provider import LocalExecutionEnvironment
from exxec.local_provider.pty_manager import LocalPtyManager

__all__ = ["LocalExecutionEnvironment", "LocalPtyManager"]
