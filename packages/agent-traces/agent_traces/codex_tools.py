"""Explicit model calls and independent native execution details."""

import orjson

from agent_traces import codex_records as cr
from agent_traces import events as ev

FAILED_STATUSES: frozenset[str] = frozenset({"failed", "declined", "cancelled"})


TOOL_KINDS: dict[str, ev.ToolKind] = {
    "exec": "shell",
    "exec_command": "shell",
    "shell": "shell",
    "shell_command": "shell",
    "write_stdin": "shell",
    "js": "shell",
    "apply_patch": "file_edit",
    "patch": "file_edit",
    "edit_file": "file_edit",
    "write_file": "file_edit",
    "spawn_agent": "subagent",
    "send_message": "subagent",
    "send_input": "subagent",
    "followup_task": "subagent",
    "wait": "subagent",
    "wait_agent": "subagent",
    "list_agents": "subagent",
    "resume_agent": "subagent",
    "close_agent": "subagent",
    "interrupt_agent": "subagent",
    "view_image": "image",
    "open_image": "image",
    "update_plan": "plan",
    "plan": "plan",
    "web_search": "web_search",
    "web": "web_search",
    "search_query": "web_search",
}
"""Native model tool names mapped to the shared kind vocabulary."""


def model_tool(item: cr.ModelTool, calls: dict[str, cr.ModelCall]) -> ev.ToolCall | ev.ToolResult:
    """Preserve raw call input and output, joining only the call identifier."""
    call: cr.ModelCall | None = calls.get(item.call_id)
    name: str = call.name if call is not None and call.name else "unknown"
    kind: ev.ToolKind = "mcp" if name.startswith(("mcp__", "mcp/")) else TOOL_KINDS.get(name.rsplit(".", 1)[-1], "other")
    name = ev.tool_path(name)
    match item:
        case cr.FunctionCall(arguments=arguments):
            return ev.ToolCall(name, item.call_id, arguments, kind)
        case cr.CustomToolCall(input=text):
            return ev.ToolCall(name, item.call_id, text, kind)
        case cr.ToolCallOutput(output=output):
            text_output: str = output if isinstance(output, str) else orjson.dumps(output).decode()
            return ev.ToolResult(name, item.call_id, text_output, "", kind, None)


def execution(item: cr.CommandExecution | cr.FileChange | cr.McpToolCall | cr.OtherTool, raw_json: str, input_json: str, entity_name: str, call_id: str) -> ev.Execution:
    """Keep native execution output, inputs and explicit relationships."""
    text: str = ""
    is_error: bool = False
    if isinstance(item, cr.CommandExecution):
        text = item.aggregated_output or item.stdout + item.stderr or item.formatted_output
        is_error = item.status in FAILED_STATUSES or item.exit_code not in {None, 0}
    elif isinstance(item, cr.FileChange):
        text = item.stdout + item.stderr
        is_error = item.status in FAILED_STATUSES
    elif isinstance(item, cr.McpToolCall):
        text = item.error.message if item.error else ""
        is_error = item.error is not None or item.status in FAILED_STATUSES
    return ev.Execution(entity_name, item.id, text, input_json, raw_json, call_id, is_error)
