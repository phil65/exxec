# AGENTS.md

## Overview

exxec is an async Python library that runs code and shell commands in interchangeable
execution environments: the local machine (optionally an isolated subprocess), Docker, SSH, Pyodide (via Deno),
Anthropic's sandbox-runtime (`srt`), an ACP client's terminal capabilities, and cloud sandboxes (E2B, Daytona, Modal, Beam, Vercel,
Cloudflare, HopX, Microsandbox, Sprites). Every environment has the same interface:
execute code, run and stream commands, spawn processes (optionally under a PTY), and access
the filesystem through fsspec. `exxec_config` provides pydantic configs that build environments.

## Architecture

- `src/exxec/base.py`: `ExecutionEnvironment` ABC, the core abstraction. Async context
  manager (`__aenter__` sets up the sandbox, `__aexit__` tears it down). Abstract
  `execute()`, `stream_code()`, `stream_command()`; optional `spawn()` (gated by the
  `supports_spawn` / `supports_pty` ClassVars), `get_fs()`, `process_manager`,
  `get_pty_manager()`, file helpers.
- `src/exxec/<name>_provider/`: one package per backend, each with `provider.py`
  (the `XxxExecutionEnvironment` subclass), usually `process_handle.py` (the provider's
  `ProcessHandle` for `spawn()`), sometimes `helpers.py`, `pty_manager.py`, `process_manager.py`.
  Providers: `local`, `docker`, `ssh`, `e2b`, `daytona`, `modal`, `beam`, `vercel`,
  `cloudflare`, `hopx`, `microsandbox`, `sprites`, `srt`, `pyodide`, `acp`, `mock`.
- `src/exxec/process_handle.py`: `ProcessHandle` ABC returned by `spawn()`: output
  iterator, `wait`, `write`, `resize`, `kill` (TERM then KILL after a grace period), exit code
  normalization.
- `src/exxec/process_manager.py`: background processes for the anyenv
  `ProcessManagerProtocol`. `SpawnProcessManager` (built on `spawn()`) and
  `StreamingProcessManager` (fallback built on `stream_command()`); `TerminalTask` record.
- `src/exxec/pty_manager.py`: `PtySize`, `PtyInfo`, `PtyManagerProtocol`, `BasePtyManager`.
  `src/exxec/spawn_pty_manager.py`: the generic PTY manager on top of `spawn(pty=...)`.
- `src/exxec/terminal/`: `TerminalSession` (`session.py`) drives a PTY process through
  a screen emulator (`screen.py`, `GhosttyScreen` via optional `pyghostty`); `keys.py` encodes keys.
- `src/exxec/spawn_stream.py`: `stream_command_via_spawn()` turns `spawn()` into event streams.
  `src/exxec/output_broadcast.py`: fan-out of one byte stream to many subscribers.
- `src/exxec/events.py` (`ExecutionEvent` union: started/output/completed/error),
  `src/exxec/models.py` (`ExecutionResult`, `ServerInfo`, `Language`),
  `src/exxec/parse_output.py` (wraps code in a runner script and parses its result),
  `src/exxec/command_line.py` (shell quoting for local/remote/spawn command lines),
  `src/exxec/pep723.py` (inline script metadata for `execute_script`),
  `src/exxec/remote_callable.py` (`create_remote_callable`: run a Python callable remotely).
- `src/exxec/__init__.py`: re-exports and `get_environment(provider, **kwargs)` with
  typed overloads per provider.
- `src/exxec_config/configs.py`: one pydantic (`schemez.Schema`) config per provider with a
  `type` discriminator and `get_provider()`, which imports the provider lazily;
  `ExecutionEnvironmentConfig` is the discriminated union. Kept import-light on purpose.

Control flow: `get_environment()` or `Config.get_provider()` -> `async with env:` ->
`execute()` / `stream_command()` / `spawn()` -> provider SDK or subprocess -> `ExecutionResult`
or `ExecutionEvent`s.

## Where to look

| Task | Location |
|---|---|
| Add or change a provider | `src/exxec/<name>_provider/`, config in `src/exxec_config/configs.py`, overload + dispatch in `src/exxec/__init__.py`, extra in `pyproject.toml` |
| Shared environment interface | `src/exxec/base.py` |
| Process spawning / signals | `src/exxec/process_handle.py`, provider `process_handle.py` |
| Background processes | `src/exxec/process_manager.py` |
| PTY / terminal automation | `src/exxec/pty_manager.py`, `src/exxec/spawn_pty_manager.py`, `src/exxec/terminal/` |
| Code wrapping / result parsing | `src/exxec/parse_output.py` |
| Tests | `tests/test_<provider>_execution.py`, `tests/test_spawn_<provider>.py` |
| Planned breaking renames | `CHANGES_FOR_NEXT_MAJOR.md` |

## Commands

```bash
uv sync --all-extras                 # setup (CI: --no-group docs --no-group lint for tests)
uv run pytest                        # tests (integration tests deselected by default)
uv run pytest -m integration         # tests needing cloud credentials
uv run ruff check . && uv run ruff format --check .
uv run ty check                      # type check src/ (warnings fail CI)
uv run deptry src                    # dependency declarations
uv run mknodes build          # docs
# duties.py defines lint, lint_check, test, build, serve, update, version (duty tasks)
```

## Conventions

- Python >= 3.14, fully async API. No `from __future__ import annotations` (PEP 649; banned via ruff TID251);
  type-only imports under `TYPE_CHECKING`.
- Ruff with a broad rule set (`pyproject.toml`), line length 100, preview formatting.
  ty is the type checker (`[tool.ty.*]`); use targeted `# ty: ignore[rule]` only.
- Google-style docstrings without types in `Args`.
- Tests are plain pytest functions (no classes), `asyncio_mode = "auto"`.
- Optional provider SDKs are imported lazily inside methods, never at module level:
  `import exxec` must work without any extra installed (it imports all provider modules).

## Specs

Remote sandbox APIs, used through each vendor's Python SDK (versions in `[project.optional-dependencies]`). Local backends (local, docker via testcontainers, ssh, mock) are not listed.

| Spec | URL | Targets | Code | Update check |
|---|---|---|---|---|
| ACP terminals (`terminal/*` client methods) | https://agentclientprotocol.com/protocol/terminals | unpinned (tracks latest) | `src/exxec/acp_provider/` | https://github.com/agentclientprotocol/agent-client-protocol/releases |
| Beam sandboxes (`beam-client>=0.2.187`) | https://docs.beam.cloud/v2/sandbox/overview | unpinned (tracks latest SDK) | `src/exxec/beam_provider/` | same URL |
| Cloudflare Sandbox SDK (HTTP to a Worker) | https://developers.cloudflare.com/sandbox/ | unpinned (tracks latest) | `src/exxec/cloudflare_provider/` | same URL |
| Daytona (`daytona<2`) | https://www.daytona.io/docs/ | daytona SDK 1.x (2.x is Daytona Cloud only) | `src/exxec/daytona_provider/` | same URL |
| E2B (`e2b>=2.0.0`) | https://e2b.dev/docs | unpinned (tracks latest SDK) | `src/exxec/e2b_provider/` | same URL |
| HopX (`hopx-ai`) | https://docs.hopx.ai/ | unpinned (tracks latest SDK) | `src/exxec/hopx_provider/` | same URL |
| microsandbox (`microsandbox`) | https://docs.microsandbox.dev/ | unpinned (tracks latest SDK) | `src/exxec/microsandbox_provider/` | same URL |
| Modal Sandboxes (`modal`) | https://modal.com/docs/guide/sandbox | unpinned (tracks latest SDK) | `src/exxec/modal_provider/` | same URL |
| Pyodide (run under Deno) | https://pyodide.org/en/stable/ | unpinned (tracks latest) | `src/exxec/pyodide_provider/` | same URL |
| Sprites (`sprites-py`) | https://docs.sprites.dev/ | unpinned (tracks latest SDK) | `src/exxec/sprites_provider/` | same URL |
| Anthropic sandbox-runtime (`srt` CLI) | https://github.com/anthropic-experimental/sandbox-runtime | unpinned (tracks latest) | `src/exxec/srt_provider/` | same URL (releases) |
| Vercel Sandbox (`vercel`) | https://vercel.com/docs/vercel-sandbox | unpinned (tracks latest SDK) | `src/exxec/vercel_provider/` | same URL |

## Pitfalls

- Copier-managed project (`.copier-answers.yml`, template `copier-phil65`): workflows,
  `duties.py`, `mkdocs.yml`, `.pre-commit-config.yaml` and most of `pyproject.toml` come from
  the template. Add dependencies below the `# Only add below (Copier)` markers.
- Extras per provider (`beam`, `cloudflare`, `daytona`, `docker`, `e2b`, `hopx`,
  `microsandbox`, `modal`, `sprites`, `ssh`, `vercel`, `vt`). `daytona` is capped `<2`
  (the 2.x SDK changed its API) and limited to Python < 3.14; `pyghostty` (`vt`) is not on Windows.
- `[tool.ty.analysis] allowed-unresolved-imports` lists `acp`, `daytona`, `pyghostty`:
  `acp` ships with agentpool (a dependent of exxec), so it cannot be declared.
- `[tool.uv] override-dependencies` forces `websockets>=15` (beam-client vs daytona conflict).
- Test marker `integration`: needs real cloud credentials (E2B, Beam, Daytona, Modal, Vercel,
  ...). Docker tests run on Linux only and need a working Docker daemon.
- Env vars read by providers include `DAYTONA_API_URL`, `SPRITES_TOKEN`; the cloud SDKs read
  their own credentials (integration tests skip without e.g. `DAYTONA_API_KEY`, `BEAM_TOKEN`, `MODAL_TOKEN_ID`).
- `uv.lock` is gitignored.
