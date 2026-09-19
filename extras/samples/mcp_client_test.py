import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["mcp_server/weather_server.py"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # Initialize MCP connection
            await session.initialize()

            # Discover available tools
            tools = await session.list_tools()

            print("Available tools:")

            for tool in tools.tools:
                print(f"- {tool.name}")
                print(f"  Description: {tool.description}")
                print(f"  Input schema: {tool.input_schema}")

            # Call weather tool
            result = await session.call_tool(
                "get_weather",
                arguments={
                    "latitude": 1.3521,
                    "longitude": 103.8198,
                },
            )

            print("\nWeather tool result:")
            text = result.content[0].text
            weather_data = json.loads(text)
            print(weather_data["current"]["temperature_2m"])
            print(result)


if __name__ == "__main__":
    asyncio.run(main())