import requests
from mcp.server import MCPServer


# ------------------------------------------------------------
# MCP Server
# ------------------------------------------------------------
# This MCP server exposes travel-related tools that provide
# current information which should not be stored in the static
# Knowledge Base.
#
# Current tools:
#   1. get_weather()       -> current/forecast weather
#   2. convert_currency()  -> current exchange-rate conversion
mcp = MCPServer(
    "Singapore Travel Server"
)


# ------------------------------------------------------------
# Weather Tool
# ------------------------------------------------------------
@mcp.tool()
def get_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Get current weather and a daily weather forecast for a location.

    The tool uses the Open-Meteo API as the external data source.

    Parameters:
        latitude: Latitude of the requested location.
        longitude: Longitude of the requested location.
        start_date: First forecast date in YYYY-MM-DD format.
        end_date: Last forecast date in YYYY-MM-DD format.

    Returns:
        Weather information returned by Open-Meteo.
    """

    # Open-Meteo provides the current and forecast weather data
    # required by the travel assistant.
    url = "https://api.open-meteo.com/v1/forecast"

    # Define exactly which weather information the application needs.
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "daily": (
            "weather_code,"
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_probability_max"
        ),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }

    # Call the external weather API.
    #
    # timeout prevents the MCP tool from waiting indefinitely
    # if the external service does not respond.
    response = requests.get(
        url,
        params=params,
        timeout=10,
    )

    # Convert HTTP failures into exceptions so the MCP client
    # can report that current information could not be retrieved.
    response.raise_for_status()

    # Return the API response to the MCP client.
    return response.json()


# ------------------------------------------------------------
# Currency Conversion Tool
# ------------------------------------------------------------
@mcp.tool()
def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> dict:
    """
    Convert an amount from one currency to another.

    The tool uses the Frankfurter API to obtain the latest
    available exchange-rate information.

    Parameters:
        amount: Amount to convert.
        from_currency: Source currency code, e.g. INR.
        to_currency: Target currency code, e.g. USD.

    Returns:
        Currency conversion result returned by Frankfurter.
    """

    # Frankfurter provides exchange-rate information that can
    # be retrieved dynamically through the MCP tool.
    url = "https://api.frankfurter.app/latest"

    # Normalize currency codes to uppercase so that inputs such
    # as "inr" and "INR" are handled consistently.
    params = {
        "amount": amount,
        "from": from_currency.upper(),
        "to": to_currency.upper(),
    }

    # Call the external currency API.
    response = requests.get(
        url,
        params=params,
        timeout=10,
    )

    # Raise an exception for HTTP/API failures instead of
    # returning an unreliable or incomplete result.
    response.raise_for_status()

    # Return the conversion result to the MCP client.
    return response.json()


# ------------------------------------------------------------
# Start MCP Server
# ------------------------------------------------------------
if __name__ == "__main__":
    # Start the MCP server and wait for requests from the client.
    #
    # The MCP client starts this Python process using stdio
    # communication.
    mcp.run()