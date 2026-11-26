"""Helpers for building POSIX shell command lines for remote environments."""

import re
import shlex


_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_command_line(command: str, args: list[str] | None) -> str:
    """Build a POSIX shell command line from a command and optional arguments.

    Args:
        command: Program name, or a full shell command string when no args are given
        args: Arguments, shell-quoted when joined

    Returns:
        A command line suitable for execution by a POSIX shell
    """
    return shlex.join([command, *args]) if args else command


def build_remote_command_line(
    command_line: str,
    cwd: str | None,
    env: dict[str, str] | None,
) -> str:
    """Prefix a command line with directory change and environment exports.

    Used for environments where cwd and env can only be passed as part of the
    shell command itself.

    Args:
        command_line: Shell command line to run
        cwd: Working directory to change into first
        env: Environment variables to export first

    Returns:
        Combined POSIX shell command line

    Raises:
        ValueError: If an environment variable name is not a valid identifier
    """
    parts: list[str] = []
    if cwd:
        parts.append(f"cd {shlex.quote(cwd)}")
    if env:
        invalid = [name for name in env if not _ENV_VAR_NAME.match(name)]
        if invalid:
            msg = f"Invalid environment variable names: {', '.join(invalid)}"
            raise ValueError(msg)
        assignments = " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items())
        parts.append(f"export {assignments}")
    parts.append(command_line)
    return " && ".join(parts)


def build_spawn_command_line(
    command: str,
    args: list[str] | None,
    cwd: str | None,
    env: dict[str, str] | None,
) -> str:
    """Build the full command line for spawning a process in a POSIX environment.

    With `args`, the program replaces the shell via `exec`, so signals sent to
    the spawned process reach the program itself rather than a wrapping shell.

    Args:
        command: Program (with `args`) or shell command line (without)
        args: Program arguments
        cwd: Working directory to change into first
        env: Environment variables to export first

    Returns:
        POSIX shell command line

    Raises:
        ValueError: If an environment variable name is not a valid identifier
    """
    line = f"exec {shlex.join([command, *args])}" if args else command
    return build_remote_command_line(line, cwd, env)


def with_timeout(command_line: str, timeout: float | None) -> str:
    """Limit a shell command line's run time with coreutils `timeout`.

    The line runs in its own `sh -c`, so builtins (`cd`, `export`) and compound
    commands (`a && b`) work and are limited as a whole. Exit code 124 means the
    limit was hit.

    Args:
        command_line: Shell command line
        timeout: Limit in seconds, None for no limit (the line is returned unchanged)

    Returns:
        POSIX shell command line
    """
    if timeout is None:
        return command_line
    return f"timeout {timeout:g} sh -c {shlex.quote(command_line)}"
