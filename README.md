# Exxec

[![PyPI License](https://img.shields.io/pypi/l/exxec.svg)](https://pypi.org/project/exxec/)
[![Package status](https://img.shields.io/pypi/status/exxec.svg)](https://pypi.org/project/exxec/)
[![Monthly downloads](https://img.shields.io/pypi/dm/exxec.svg)](https://pypi.org/project/exxec/)
[![Distribution format](https://img.shields.io/pypi/format/exxec.svg)](https://pypi.org/project/exxec/)
[![Wheel availability](https://img.shields.io/pypi/wheel/exxec.svg)](https://pypi.org/project/exxec/)
[![Python version](https://img.shields.io/pypi/pyversions/exxec.svg)](https://pypi.org/project/exxec/)
[![Implementation](https://img.shields.io/pypi/implementation/exxec.svg)](https://pypi.org/project/exxec/)
[![Releases](https://img.shields.io/github/downloads/phil65/exxec/total.svg)](https://github.com/phil65/exxec/releases)
[![Github Contributors](https://img.shields.io/github/contributors/phil65/exxec)](https://github.com/phil65/exxec/graphs/contributors)
[![Github Forks](https://img.shields.io/github/forks/phil65/exxec)](https://github.com/phil65/exxec/forks)
[![Github Issues](https://img.shields.io/github/issues/phil65/exxec)](https://github.com/phil65/exxec/issues)
[![Github Issues](https://img.shields.io/github/issues-pr/phil65/exxec)](https://github.com/phil65/exxec/pulls)
[![Github Watchers](https://img.shields.io/github/watchers/phil65/exxec)](https://github.com/phil65/exxec/watchers)
[![Github Stars](https://img.shields.io/github/stars/phil65/exxec)](https://github.com/phil65/exxec/stars)
[![Github Repository size](https://img.shields.io/github/repo-size/phil65/exxec)](https://github.com/phil65/exxec)
[![Github last commit](https://img.shields.io/github/last-commit/phil65/exxec)](https://github.com/phil65/exxec/commits)
[![Github release date](https://img.shields.io/github/release-date/phil65/exxec)](https://github.com/phil65/exxec/releases)
[![Github language count](https://img.shields.io/github/languages/count/phil65/exxec)](https://github.com/phil65/exxec)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/exxec)](https://github.com/phil65/exxec)
[![Package status](https://codecov.io/gh/phil65/exxec/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/exxec/)
[![PyUp](https://pyup.io/repos/github/phil65/exxec/shield.svg)](https://pyup.io/repos/github/phil65/exxec/)

[Read the documentation!](https://phil65.github.io/exxec/)
**One async API to run code and shell commands locally, in Docker, over SSH, or in cloud sandboxes.**

exxec gives every backend the same interface: execute Python/JavaScript/TypeScript code,
run and stream shell commands, spawn long-running processes (optionally on a PTY), and read
and write files through an fsspec filesystem. Switch from local execution to Docker or E2B by
changing one argument.

## Installation

```bash
pip install exxec
```

Cloud and container backends need their SDK, installed via extras:

```bash
pip install "exxec[docker]"        # or e2b, daytona, modal, beam, vercel, ssh, ...
pip install "exxec[vt]"            # terminal emulation for TerminalSession
```

## Quickstart

```python
import asyncio

from exxec import get_environment


async def main() -> None:
    async with get_environment("local") as env:
        # Run code: return a value from `main()` or assign `_result`
        result = await env.execute("""
async def main():
    return "Hello from exxec!"
""")
        print(result.success, result.result)  # True Hello from exxec!

        # Run a shell command
        cmd = await env.execute_command("echo hi")
        print(cmd.exit_code, cmd.stdout)  # 0 hi

        # Stream a command's output line by line
        async for line in env.execute_command_stream("echo a; echo b"):
            print(line)


asyncio.run(main())
```

Swap `"local"` for `"docker"`, `"e2b"`, `"modal"`, ... (with the matching extra installed and
credentials configured); the rest of the code stays the same.

## Providers

| Provider | `get_environment()` name | Extra | `spawn()` | PTY |
|---|---|---|---|---|
| Local machine (in-process or `isolated=True` subprocess) | `local` | none | yes | yes (not on Windows) |
| Anthropic sandbox-runtime (`srt` CLI) | `srt` | none (needs `srt`) | yes | yes (not on Windows) |
| Pyodide via Deno | `pyodide` | none (needs `deno`) | | |
| Docker (testcontainers) | `docker` | `docker` | yes | yes |
| SSH | `ssh` | `ssh` | yes | yes |
| E2B | `e2b` | `e2b` | yes | yes |
| Daytona | `daytona` | `daytona` | yes | yes |
| Modal | `modal` | `modal` | yes | yes |
| Vercel Sandbox | `vercel` | `vercel` | yes | yes |
| Microsandbox | `microsandbox` | `microsandbox` | yes | yes |
| Beam | `beam` | `beam` | yes | |
| Cloudflare Sandbox (Workers HTTP API) | `cloudflare` | `cloudflare` | | |
| HopX | `hopx` | `hopx` | | PTY manager only |
| Sprites (Fly.io) | `sprites` | `sprites` | | PTY manager only |

The classes are importable directly as well (`LocalExecutionEnvironment`,
`DockerExecutionEnvironment`, `E2bExecutionEnvironment`, ...). `get_environment()` has typed
overloads, so your editor shows each provider's options. `MockExecutionEnvironment` is
available for tests.

`srt` runs each command inside the sandbox; its `dependencies` are installed with `pip install`
when the environment is entered, outside the sandbox, since the list is the environment's own
configuration.

## Features

### Code execution

`execute()` runs a snippet and returns an `ExecutionResult` with `success`, `result`,
`duration`, `stdout`, `stderr`, `error` and `error_type`. The snippet either defines
`main()` (sync or async), whose return value becomes `result`, or assigns `_result`:

```python
result = await env.execute("import math\n_result = math.pi * 2")
if not result.success:
    print(result.error_type, result.error)
```

`stream_code()` and `stream_command()` yield typed events (`ProcessStartedEvent`,
`OutputEvent`, `ProcessCompletedEvent`, `ProcessErrorEvent`); `execute_stream()` and
`execute_command_stream()` yield plain output lines.

### Processes and terminals

`spawn()` starts a process and returns a `ProcessHandle` (output iterator, `write`, `wait`,
`kill`, `resize`). With `pty=PtySize(...)` the process runs on a pseudo-terminal.
`TerminalSession` (extra `vt`) adds an emulated screen, which lets you drive interactive programs:

```python
from exxec import LocalExecutionEnvironment
from exxec.terminal import TerminalSession

async with (
    LocalExecutionEnvironment() as env,
    await TerminalSession.start(env, "python3", ["-q"]) as term,
):
    await term.wait_for(">>>")
    await term.send_line("6 * 7")
    await term.wait_for("42")
    print(term.text())
    await term.send_keys("C-d")
```

`env.process_manager` manages background processes and `env.get_pty_manager()` manages PTY sessions.

### Filesystem

`env.get_fs()` returns an async fsspec filesystem for the environment (local, Docker, SSH,
and most sandboxes); `set_file_content()` and `get_file_content()` are shortcuts.

### Configuration models

`exxec_config` holds lightweight pydantic models for every provider, a discriminated union
(`ExecutionEnvironmentConfig`) for config files, and `get_provider()` to build the environment:

```python
from exxec_config import DockerExecutionEnvironmentConfig

config = DockerExecutionEnvironmentConfig(image="python:3.13-slim", dependencies=["numpy"])
async with config.get_provider() as env:
    ...
```

### Remote callables and PEP 723 scripts

`create_remote_callable(func, DockerExecutionEnvironment)` wraps a function so that each call
runs inside a fresh environment, with dependencies inferred from its module.
`ExecutionEnvironment.execute_script()` runs a script with PEP 723 inline metadata and installs
its declared dependencies.

## Documentation

Full API reference: [phil65.github.io/exxec](https://phil65.github.io/exxec/)

## License

MIT
