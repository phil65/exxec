# Changes for the next major version

Breaking changes collected for a future major release. None of them change
behaviour; they make names and types match what the code actually does.

## "Terminal" vs "process" naming

The background-process machinery uses "terminal" names, but these objects are
plain processes with piped output (no PTY). "Terminal" should be reserved for the
PTY/emulator layer (`PtyManagerProtocol`, `TerminalSession`).

| Current | Proposed | Where |
|---|---|---|
| `TerminalTask` | `ProcessRecord` | `exxec.process_manager` |
| `EnvironmentTerminalManager` (deprecated alias) | removed; use `StreamingProcessManager` | `exxec.process_manager` |
| `BaseTerminal` | `BaseProcessRecord` | `anyenv.process_manager` |
| `BaseTerminal.terminal_id` | `process_id` | `anyenv.process_manager` |
| `get_process_info()["terminal_id"]` | `"process_id"` | exxec managers (anyenv's `ProcessManager` already uses `process_id`) |
| `self._terminals` | `self._processes` | `exxec.process_manager` (internal) |

Keep "terminal" where an external protocol defines it: ACP's `terminal/*` methods
and `ACPProcessManager` internals map to ACP terminals by spec.

## Typed process info

`ProcessManagerProtocol.get_process_info()` returns `dict[str, Any]`. Replace it with
a dataclass (e.g. `ProcessInfo`: id, command, args, cwd, created_at, is_running,
exit_code, output_limit) in anyenv and use it in all exxec managers.

## PTY manager protocol

- Add `wait(pty_id) -> int` to `PtyManagerProtocol` (`SpawnPtyManager` has it).
- Make `get_pty_manager()` a property, like `process_manager`.
- Drop the remaining per-provider PTY managers (hopx, sprites) once those
  providers implement `spawn()`, and remove their exports.
