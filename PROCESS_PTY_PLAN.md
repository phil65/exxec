# Process / PTY / Terminal redesign

Status: steps 1–4 are done; see the provider status table for per-provider support.

## Problem

Execution environments expose three independent ways of running a command with I/O:

| | `process_manager` | `get_pty_manager()` | `stream_command()` |
|---|---|---|---|
| Access | property, cached in base | method, cached per provider | method |
| Default | generic (was: local subprocess) | raises `NotImplementedError` | abstract |
| Implemented by | base, local, acp, mock | local, ssh, docker, e2b, modal, daytona, hopx, sprites, mock | all |

Consequences:

- **Duplication**: each PTY-capable provider ships ~250–330 lines of PTY code in addition
  to its `stream_command`, and they diverge:
  - ssh / e2b join `args` with spaces without quoting
  - e2b computes the joined command and discards it
  - daytona sends the command via stdin
  - hopx ignores command, args, cwd and env
- **Capability discovery** happens by catching `NotImplementedError`.
- **Raw byte streams have no single owner**: `read()` and `stream()` compete for bytes
  (local `read()` locks, `stream()` does not), so two consumers (websocket + emulator)
  would each see partial output.
- **Local PTY blocks executor threads**: blocking `ptyprocess.read` in the default
  executor plus `sleep(0.01)` polling instead of `loop.add_reader(fd)`.
- **No exit semantics in the PTY protocol**: no `wait()`; `kill()` deletes the session
  and loses the exit code; status only updates as a side effect of `stream()`/`get_info()`.
- **`stream_command` timeout semantics differ**: local applies the timeout per line
  (inactivity), docker wraps with `timeout N` (total, and `timeout None` when unset),
  and there is no way to request "no timeout" when a default is configured.
- **Local `stream_command` does not kill its subprocess** when the generator is closed.
- **Naming**: "terminal" means *background process* in `TerminalTask` /
  `EnvironmentTerminalManager` / `BaseTerminal` and *PTY* elsewhere.

## Target design

Providers implement one primitive; everything else is generic and written once.

```
Provider:
  spawn(command, args, cwd, env, pty: PtySize | None) -> ProcessHandle
  supports_pty: bool

ProcessHandle (per provider, ~80-120 lines):
  pid, write(bytes), output() -> single-owner byte stream,
  resize(PtySize), wait() -> int, kill(), close stdin

Generic layers (exxec):
  stream_command   = spawn(pty=None) + line splitting + event mapping
                     (default impl; providers may keep an optimized one)
  ProcessManager   = spawn(pty=None) + buffered output, real kill, no implicit timeout
  PtyManager       = spawn(pty=size) + session registry (keeps exit codes after exit)
  TerminalSession  = PTY handle + ScreenEmulator + broadcast to N subscribers
```

### Screen emulation

`ScreenEmulator` protocol (`feed`, `text`, `contents`, `cursor`, `resize`), with a
pyghostty (`libghostty-vt`) implementation behind an optional `exxec[vt]` extra.
pyghostty ships wheels for macOS (arm64/x86_64) and manylinux (x86_64/aarch64) only,
so the protocol keeps the dependency optional and its untyped cffi surface contained.

`TerminalSession` owns the single reader of a PTY, feeds the emulator and fans out raw
bytes to subscribers (e.g. the OpenCode websocket). It provides:

- `screen()` / `scrollback()` / `cursor` — clean text for agents
- `send_keys("<C-c>")` — key-name encoding
- `wait_for_quiet(timeout)` / `wait_for(pattern, timeout)` — settle detection

## Steps

1. **Done** — Background processes run in the environment.
   `LocalProcessManager` (local + subclasses: applies `wrap_command`, cwd, env vars,
   process-group kill) and `StreamingProcessManager` (all other providers, via
   `stream_command`). Environments clean up running processes on exit.
   Known interim limitations of `StreamingProcessManager`: env default timeout applies;
   kill relies on provider cancellation handling.
2. **Done** — `ProcessHandle` (`exxec.process_handle`), `ExecutionEnvironment.spawn()`
   and the `supports_spawn` / `supports_pty` flags. Implemented for:
   - local: pipes via asyncio subprocess; PTYs without a Python-level fork
     (isolated helper in a new session sets the controlling tty and execs), non-blocking master fd
   - ssh: one session channel per process, SSH signal and window-change requests
   - docker: exec API over the hijacked socket (demuxed pipe frames), signals via
     pidfile + `kill` helper exec, exit code from `exec_inspect`
   Also fixed along the way: SSH env connected after OS detection, used an outdated
   sshfs private API, and leaked the filesystem's SSH connection.
3. **Done for spawn-capable environments** — `SpawnProcessManager` and `SpawnPtyManager`
   (with `OutputBroadcaster` fan-out and replay) are the defaults for `supports_spawn` /
   `supports_pty`; `StreamingProcessManager` stays as fallback. Local/ssh/docker PTY
   managers and `LocalProcessManager` were removed. Local `stream_command` is built on
   `spawn()` (kills on close, total timeout, process-group kill, honours cwd).
   microsandbox implements `spawn()` too; `exxec.spawn_stream` provides the shared
   spawn-based `stream_command`.
   Remaining:
   - implement `spawn()` for vercel (credentials currently get HTTP 403), modal,
     daytona, hopx, sprites (and others with a process API), then drop their PTY managers
   - unused native process managers exist for beam, daytona, hopx and vercel
     (`*_provider/process_manager.py`, never wired up); replace with `spawn()` or wire them
   - unify `stream_command` timeout semantics across providers (docker builds
     `timeout None …` when no timeout is configured)
4. **Done** — `exxec.terminal` (optional `exxec[vt]` extra, pyghostty on macOS/Linux):
   `ScreenEmulator` protocol + `GhosttyScreen`, `encode_key()`, and `TerminalSession`
   (spawned PTY + emulator + broadcaster): `text()` / `scrollback()` / `cursor` /
   `alternate_screen`, `send` / `send_line` / `send_keys` / `paste` (bracketed when
   enabled), `wait_for(pattern)` / `wait_for_quiet()`, resize, and automatic answers
   to terminal queries (cursor position, device attributes) unless a real terminal
   is attached (`answer_queries=False`). Works on local, ssh and docker.
5. Renames and protocol cleanups are deferred to the next major version; see
   `CHANGES_FOR_NEXT_MAJOR.md`.

Downstream (agentpool): per-agent terminal registry, `TerminalTools` resource provider,
OpenCode PTY routes reusing `TerminalSession`, optional VFS snapshot mounts.

## Provider status

| Provider | `spawn()` pipes | `spawn()` PTY | Verified against | Notes |
|---|---|---|---|---|
| local (+ srt) | yes | yes (POSIX) | real processes | no Python-level fork; process groups |
| ssh | yes | yes | in-process asyncssh server | SSH signal + window-change requests |
| docker | yes | yes | real container | exec API; signals via pidfile + `kill` |
| microsandbox | yes | yes | real microVMs (local) | CI runners lack a usable hypervisor |
| e2b | yes | yes | real sandboxes | PTY commands exec'd into e2b's login shell after a screen clear |
| daytona | yes | yes | real sandboxes | via Daytona's SSH gateway; SIGTERM reports exit 1, not 143 |
| beam | yes | no (Beam has none) | real sandboxes | no interactive stdin; sync SDK runs in threads |
| vercel | yes | yes | SDK fakes only | credentials get HTTP 403; integration test skips |
| modal | yes | yes | local stand-in for `exec` | no Modal credentials; wrappers for PID/TTY/resize |
| sprites | no | own PTY manager | — | command I/O via sync file objects; no credentials |
| hopx | no | own PTY manager | — | no PID/stdin/output stream for background commands |
| cloudflare | no | no | — | needs a deployed Sandbox Worker (`base_url`) |
| acp, pyodide, mock | n/a | n/a | — | protocol/test environments |

Unused native process managers (beam, daytona, e2b, hopx, vercel) were removed;
per-provider PTY managers remain only for sprites and hopx.

Filesystems (upathtools): microsandbox_fs rewritten on `sandbox.fs` (1.21.0). Gaps:
- e2b_fs: `files.rename` and `files.watch_dir` unused; `_glob`/`_grep` still TODO
- vercel_fs uses `find`/`stat` via commands
- watch APIs (e2b `watch_dir`, modal `filesystem.watch`) could back change notifications
