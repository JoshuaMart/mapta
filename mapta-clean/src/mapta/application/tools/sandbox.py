"""Tools that execute code inside the scan's sandbox."""

import uuid

from .base import ToolContext, tool

__all__ = ["sandbox_run_command", "sandbox_run_python"]


@tool
async def sandbox_run_command(ctx: ToolContext, command: str, timeout: int = 120) -> str:
    """Run a shell command inside an ephemeral sandbox and return stdout/stderr/exit code.

    Args:
        command: Shell command to execute (e.g., "ls -la").
        timeout: Max seconds to wait before timing out the command.
    """
    result = await ctx.sandbox.run(command, timeout=timeout)
    return result.render(max_chars=ctx.max_tool_output_chars)


@tool
async def sandbox_run_python(ctx: ToolContext, python_code: str, timeout: int = 120) -> str:
    """Run Python code inside the sandbox and return stdout/stderr/exit code.

    Output longer than 30,000 characters is truncated before being returned.

    Args:
        python_code: Python code to execute (e.g., "print('Hello World')").
        timeout: Max seconds to wait before timing out the code execution.
    """
    script_path = f"/home/user/temp_script_{uuid.uuid4().hex[:8]}.py"
    await ctx.sandbox.write_file(script_path, python_code)
    result = await ctx.sandbox.run(
        f"source .venv/bin/activate 2>/dev/null; python3 {script_path}", timeout=timeout
    )
    return result.render(max_chars=ctx.max_tool_output_chars)
