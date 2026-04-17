import logging
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPManager:
    """
    Manages connections to multiple MCP servers and provides a unified interface
    for accessing tools from all connected servers.
    """

    def __init__(self, servers: dict[str, MCPServerConfig]) -> None:
        """
        Initialize the MCP manager with a set of server configurations.

        Args:
            servers: Dictionary mapping server names to their configurations.
                     Only enabled servers will be connected to.
        """
        self._servers = {k: v for k, v in servers.items() if v.enabled}
        self._sessions: dict[str, ClientSession] = {}
        # tool_name -> (server_name, tool_object)
        self._tools: dict[str, tuple[str, object]] = {}
        self._exit_stack = AsyncExitStack()

    async def start(self) -> None:
        """
        Connect to all enabled MCP servers and populate the tool registry.
        Failed servers are logged and skipped gracefully.
        """
        await self._exit_stack.__aenter__()
        for name, cfg in self._servers.items():
            try:
                session = await self._connect(name, cfg)
                self._sessions[name] = session
                result = await session.list_tools()
                for tool in result.tools:
                    self._tools[tool.name] = (name, tool)
                logger.info("MCP server '%s' connected with %d tool(s)", name, len(result.tools))
            except Exception as exc:
                logger.warning("MCP server '%s' failed to connect: %s", name, exc)

    async def _connect(self, name: str, cfg: MCPServerConfig) -> ClientSession:
        """
        Establish a connection to an MCP server based on its configuration.

        Args:
            name: The server name (for logging).
            cfg: The server configuration specifying type and connection details.

        Returns:
            An initialized ClientSession for the server.

        Raises:
            ValueError: If the server type is not recognized.
        """
        if cfg.type == "stdio":
            params = StdioServerParameters(command=cfg.command[0], args=cfg.command[1:])
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        elif cfg.type == "sse":
            read, write = await self._exit_stack.enter_async_context(sse_client(cfg.url))
        elif cfg.type == "http":
            read, write, _ = await self._exit_stack.enter_async_context(
                streamablehttp_client(cfg.url)
            )
        else:
            raise ValueError(f"Unknown MCP server type '{cfg.type}' for server '{name}'")

        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    def get_tool_definitions(self) -> list[dict]:
        """
        Return tools in the format Ollama's /api/chat expects.

        Returns:
            A list of tool definitions compatible with Ollama's tool use format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for _, (_, tool) in self._tools.items()
        ]

    async def call_tool(self, name: str, arguments: dict) -> str:
        """
        Call a tool by name with the given arguments.

        Args:
            name: The name of the tool to call.
            arguments: Dictionary of arguments to pass to the tool.

        Returns:
            The result of the tool call as a string. If the tool is not found,
            returns an error message.
        """
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        server_name, _ = self._tools[name]
        session = self._sessions[server_name]
        result = await session.call_tool(name, arguments)
        parts = [
            c.text if hasattr(c, "text") else str(c)
            for c in result.content
        ]
        return "\n".join(parts)

    def list_tools_summary(self) -> str:
        """
        Generate a human-readable summary of all available tools.

        Returns:
            A formatted string listing all tools and their descriptions.
        """
        if not self._tools:
            return "No MCP tools available."
        lines = [
            f"• **{tool_name}** ({server_name}): {tool.description or '(no description)'}"
            for tool_name, (server_name, tool) in self._tools.items()
        ]
        return "\n".join(lines)

    async def stop(self) -> None:
        """
        Gracefully shut down all MCP server connections.
        """
        await self._exit_stack.__aexit__(None, None, None)
