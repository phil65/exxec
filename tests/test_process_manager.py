"""Tests for EnvironmentTerminalManager (process manager)."""

import asyncio

import pytest

from exxec import LocalExecutionEnvironment


class TestProcessManager:
    """Tests for the process manager functionality."""

    async def test_start_and_kill_short_process(self):
        """Test starting and killing a short-lived process."""
        async with LocalExecutionEnvironment() as env:
            pm = env.process_manager

            # Start a process that prints and exits
            process_id = await pm.start_process("echo", args=["hello", "world"])

            # Wait for it to complete
            exit_code = await pm.wait_for_exit(process_id)

            assert exit_code == 0

            # Get output
            output = await pm.get_output(process_id)
            assert "hello world" in output.stdout

    async def test_start_long_running_process(self):
        """Test starting a long-running daemon-like process."""
        async with LocalExecutionEnvironment() as env:
            pm = env.process_manager

            # Start a process that runs indefinitely (like a daemon)
            # Use a simple Python script that prints periodically
            script = "import time; print('started', flush=True); time.sleep(10)"
            process_id = await pm.start_process("python", args=["-c", script])

            # Give it time to start and print
            await asyncio.sleep(0.5)

            # Check it's running
            info = await pm.get_process_info(process_id)
            assert info["is_running"] is True

            # Get output so far
            output = await pm.get_output(process_id)
            assert "started" in output.stdout

            # Kill it
            await pm.kill_process(process_id)

            # Verify it's dead
            info = await pm.get_process_info(process_id)
            assert info["is_running"] is False
            assert info["exit_code"] == 130  # SIGINT  # noqa: PLR2004

    async def test_process_without_output_timeout(self):
        """Test that a process without immediate output doesn't timeout.

        This is critical for daemon processes like LSP servers that may not
        produce output until a client connects.
        """
        async with LocalExecutionEnvironment() as env:
            pm = env.process_manager

            # Start a process that waits before producing output
            # This simulates a daemon waiting for connections
            script = "import time; time.sleep(2); print('delayed output', flush=True)"
            process_id = await pm.start_process("python", args=["-c", script])

            # The process should still be running after 1 second
            await asyncio.sleep(1)
            info = await pm.get_process_info(process_id)
            assert info["is_running"] is True

            # Wait for completion
            exit_code = await pm.wait_for_exit(process_id)
            assert exit_code == 0

            # Should have the delayed output
            output = await pm.get_output(process_id)
            assert "delayed output" in output.stdout

    async def test_list_processes(self):
        """Test listing running processes."""
        async with LocalExecutionEnvironment() as env:
            pm = env.process_manager

            # Start a few processes
            script = "import time; time.sleep(5)"
            pid1 = await pm.start_process("python", args=["-c", script])
            pid2 = await pm.start_process("python", args=["-c", script])

            # List should contain both
            processes = await pm.list_processes()
            assert pid1 in processes
            assert pid2 in processes

            # Kill them
            await pm.kill_process(pid1)
            await pm.kill_process(pid2)

    async def test_release_process(self):
        """Test releasing process resources."""
        async with LocalExecutionEnvironment() as env:
            pm = env.process_manager

            # Start a process
            script = "import time; time.sleep(10)"
            process_id = await pm.start_process("python", args=["-c", script])

            # Release it (should kill and remove)
            await pm.release_process(process_id)

            # Should no longer be in list
            processes = await pm.list_processes()
            assert process_id not in processes

            # Accessing it should raise
            with pytest.raises(ValueError, match="not found"):
                await pm.get_output(process_id)

    async def test_process_with_stderr(self):
        """Test that stderr is captured."""
        async with LocalExecutionEnvironment() as env:
            pm = env.process_manager

            # Start a process that writes to stderr
            script = "import sys; print('stderr output', file=sys.stderr, flush=True)"
            process_id = await pm.start_process("python", args=["-c", script])

            # Wait for completion
            await pm.wait_for_exit(process_id)

            # stderr should be in combined output (stderr="stdout" mode)
            output = await pm.get_output(process_id)
            assert "stderr output" in output.stdout

    async def test_process_exit_code(self):
        """Test that exit codes are properly captured."""
        async with LocalExecutionEnvironment() as env:
            pm = env.process_manager

            # Start a process that exits with non-zero code
            script = "import sys; sys.exit(42)"
            process_id = await pm.start_process("python", args=["-c", script])

            # Wait for completion
            exit_code = await pm.wait_for_exit(process_id)
            assert exit_code == 42  # noqa: PLR2004

            info = await pm.get_process_info(process_id)
            assert info["exit_code"] == 42  # noqa: PLR2004


class TestProcessManagerDaemonSupport:
    """Tests specifically for daemon/server process support."""

    async def test_socket_server_process(self):
        """Test running a simple socket server as a background process."""
        async with LocalExecutionEnvironment() as env:
            pm = env.process_manager

            # Start a simple socket server
            server_script = """
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('127.0.0.1', 0))
sock.listen(1)
port = sock.getsockname()[1]
print(f"listening:{port}", flush=True)

# Accept one connection then exit
conn, addr = sock.accept()
data = conn.recv(1024)
conn.send(b"pong")
conn.close()
sock.close()
print("done", flush=True)
"""
            process_id = await pm.start_process("python", args=["-c", server_script])

            # Wait for server to start and print port
            await asyncio.sleep(0.5)
            output = await pm.get_output(process_id)

            # Extract port from output
            port = None
            for line in output.stdout.split("\n"):
                if line.startswith("listening:"):
                    port = int(line.split(":")[1])
                    break

            assert port is not None, f"Server didn't print port. Output: {output.stdout}"

            # Connect to the server
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"ping")
            await writer.drain()
            response = await reader.read(1024)
            writer.close()
            await writer.wait_closed()

            assert response == b"pong"

            # Wait for server to finish
            exit_code = await pm.wait_for_exit(process_id)
            assert exit_code == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
