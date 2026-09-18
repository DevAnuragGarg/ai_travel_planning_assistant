import json
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ------------------------------------------------------------
# MCP Server Configuration
# ------------------------------------------------------------
# Define how the MCP client should start the MCP server.
#
# The server is launched as a separate Python process and communicates
# with this client through standard input/output (stdio).
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server/travel_server.py"],
)


def convert_mcp_tools_to_gemini_tools(mcp_tools):
    """
    Convert MCP tool definitions into the function-declaration format
    expected by the Gemini API.

    MCP is responsible for describing the tools exposed by the server.
    Gemini needs those tool definitions so that it can decide which
    tool to call and generate the required arguments.

    Flow:
        MCP tool definitions
                ↓
        Gemini function declarations
                ↓
        LLM tool selection
    """

    declarations = []

    for tool in mcp_tools:
        # MCP SDK exposes the tool schema through input_schema.
        # This schema is passed to Gemini so that the model knows which
        # arguments are required by the tool.
        declarations.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
        )

    return declarations


def is_valid_mcp_tool(tool_name, mcp_tools):
    """
    Validate that the tool selected by the LLM actually exists
    on the connected MCP server.

    This validation prevents the application from attempting to
    execute an invented or unavailable tool name.
    """

    return any(tool.name == tool_name for tool in mcp_tools)


# ------------------------------------------------------------
# MCP Session Management
# ------------------------------------------------------------
# asynccontextmanager allows us to create our own async resource
# manager using the "async with mcp_session()" syntax.
#
# The MCP connection remains alive for the entire duration of
# the block where the session is being used.
@asynccontextmanager
async def mcp_session():

    # Start the MCP server and establish stdio communication.
    #
    # "read" is used to receive messages from the MCP server.
    # "write" is used to send messages to the MCP server.
    async with stdio_client(server_params) as (read, write):

        # Create an MCP ClientSession using the communication
        # streams established above.
        async with ClientSession(read, write) as session:

            # Perform the MCP protocol initialization handshake.
            await session.initialize()

            # Yield the initialized session to the caller.
            #
            # "yield" temporarily hands control back to the caller
            # while keeping the MCP connection and session alive.
            #
            # When the surrounding "async with" block finishes,
            # the session and stdio connection are automatically closed.
            yield session


async def call_mcp_tool(session, tool_name, tool_args):
    """
    Execute an MCP tool and convert its result into a Python dictionary.

    Errors are converted into structured dictionaries so that the
    final LLM can be informed that current information could not
    be retrieved instead of the application fabricating a result.
    """

    try:
        # Ask the MCP server to execute the selected tool.
        tool_result = await session.call_tool(
            tool_name,
            arguments=tool_args,
        )

        # MCP can indicate that the tool execution itself failed.
        if tool_result.is_error:
            return {
                "error": f"MCP tool '{tool_name}' failed",
                "details": tool_result.content[0].text,
            }

        # The MCP server returns the tool result as content.
        tool_text = tool_result.content[0].text

        try:
            # Our MCP tools return JSON, so convert the response into
            # a normal Python dictionary for the rest of the application.
            return json.loads(tool_text)

        except json.JSONDecodeError:
            # Do not attempt to interpret or invent a value if the
            # MCP server returned something other than valid JSON.
            return {
                "error": "MCP tool returned invalid JSON",
                "raw_result": tool_text,
            }

    except Exception as e:
        # Handle connection errors, unavailable tools, server failures,
        # or other unexpected MCP client errors.
        #
        # Returning a structured error allows assistant.py to pass
        # the failure information to the final LLM.
        return {"error": f"MCP tool '{tool_name}' failed: {str(e)}"}
