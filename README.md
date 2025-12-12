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
[![Github Discussions](https://img.shields.io/github/discussions/phil65/exxec)](https://github.com/phil65/exxec/discussions)
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



### Basic Usage

Use the `get_environment()` function to create execution environments:

```python
from exxec import get_environment

# Local execution (same process)
env = get_environment("local")

# Subprocess execution (separate process when executing python code)
env = get_environment("local", isolated=True)

# Docker execution (containerized)
env = get_environment("docker")

# Execute code
async with env:
    result = await env.execute("""
    async def main():
        return "Hello from execution environment!"
    """)
    print(result.result)  # "Hello from execution environment!"
```

## Available Providers

### Local Provider
Executes code in the same Python process. Fastest option but offers no isolation.

```python
env = get_environment("local", timeout=30.0)
```

**Parameters:**
- `timeout` (float): Execution timeout in seconds (default: 30.0)
- `isolated` (bool): Whether to execute code in a separate process (default: False)
- `language` (Language): Programming language (default: "python")


### Docker Provider
Executes code in Docker containers for strong isolation and reproducible environments.

```python
env = get_environment(
    "docker",
    image="python:3.13-slim",
    timeout=60.0,
    language="python"
)
```

**Parameters:**
- `lifespan_handler`: Tool server context manager (optional)
- `image` (str): Docker image to use (default: "python:3.13-slim")
- `timeout` (float): Execution timeout in seconds (default: 60.0)
- `language` (Language): Programming language (default: "python")


### Daytona Provider
Executes code in remote Daytona sandboxes for cloud-based development environments.

```python
env = get_environment(
    "daytona",
    api_url="https://api.daytona.io",
    api_key="your-api-key",
    timeout=300.0,
    keep_alive=False
)
```

**Parameters:**
- `api_url` (str): Daytona API URL (optional, uses env vars if not provided)
- `api_key` (str): API key for authentication (optional)
- `target` (str): Target configuration (optional)
- `image` (str): Container image (default: "python:3.13-slim")
- `timeout` (float): Execution timeout in seconds (default: 300.0)
- `keep_alive` (bool): Keep sandbox running after execution (default: False)


### E2B Provider
Executes code in E2B sandboxes for secure, ephemeral execution environments.

```python
env = get_environment(
    "e2b",
    template="python",
    timeout=300.0,
    keep_alive=False,
    language="python"
)
```

**Parameters:**
- `template` (str): E2B template to use (optional)
- `timeout` (float): Execution timeout in seconds (default: 300.0)
- `keep_alive` (bool): Keep sandbox running after execution (default: False)
- `language` (Language): Programming language (default: "python")


### Beam Provider
Executes code in Beam cloud sandboxes for scalable, serverless execution environments.

```python
env = get_environment(
    "beam",
    cpu=1.0,
    memory=128,
    keep_warm_seconds=600,
    timeout=300.0,
    language="python"
)
```

**Parameters:**
- `cpu` (float | str): CPU cores allocated to the container (default: 1.0)
- `memory` (int | str): Memory allocated to the container in MiB (default: 128)
- `keep_warm_seconds` (int): Seconds to keep sandbox alive, -1 for no timeout (default: 600)
- `timeout` (float): Execution timeout in seconds (default: 300.0)
- `language` (Language): Programming language (default: "python")


### MCP Provider
Executes Python code with Model Context Protocol support for AI integrations.

```python
env = get_environment(
    "mcp",
    dependencies=["requests", "numpy"],
    allow_networking=True,
    timeout=30.0
)
```

**Parameters:**
- `dependencies` (list[str]): Python packages to install (optional)
- `allow_networking` (bool): Allow network access (default: True)
- `timeout` (float): Execution timeout in seconds (default: 30.0)


## Code Execution Patterns

All providers support two execution patterns:

### 1. Main Function Pattern
```python
code = """
async def main():
    # Your code here
    return "result"
"""
```

### 2. Result Variable Pattern
```python
code = """
import math
_result = math.pi * 2
"""
```

## Error Handling

Execution results include comprehensive error information:

```python
async with env:
    result = await env.execute(code)
    if result.success:
        print(f"Result: {result.result}")
        print(f"Duration: {result.duration:.3f}s")
    else:
        print(f"Error: {result.error}")
        print(f"Error Type: {result.error_type}")
```

## Multi-Language Support

Some providers support multiple programming languages:

```python
# JavaScript execution
env = get_environment("subprocess", language="javascript", executable="node")

# TypeScript execution
env = get_environment("docker", language="typescript", image="node:18")
```

## Advanced Usage

### Context Managers
All environments are async context managers for proper resource cleanup:

```python
async with get_environment("docker") as env:
    result1 = await env.execute(code1)
    result2 = await env.execute(code2)  # Reuses same container
# Container automatically cleaned up
```

### Custom Configurations
Each provider supports environment-specific customization:

```python
# Docker with custom image and networking
env = get_environment(
    "docker",
    image="tensorflow/tensorflow:latest-py3",
    timeout=600.0
)

# Subprocess with specific Python version
env = get_environment(
    "subprocess",
    executable="/usr/bin/python3.11",
    timeout=120.0
)
```

### Streaming Output

Some providers support streaming output line by line, useful for long-running processes:

```python
from exxec import get_environment

# Stream output from subprocess execution
env = get_environment("subprocess")

async with env:
    async for line in env.execute_stream("""
    import time
    for i in range(5):
        print(f"Processing step {i+1}...")
        time.sleep(1)
    print("Done!")
    """):
        print(f"Live output: {line}")

# Also works with Docker execution
env = get_environment("docker")
async with env:
    async for line in env.execute_stream(code):
        # Process each line as it's produced
        if "ERROR" in line:
            print(f"⚠️  {line}")
        else:
            print(f"✓ {line}")
```

**Supported providers:** `docker`, `local`, `beam`, `e2b`, `modal`, `vercel`, `ssh`, `daytona`
