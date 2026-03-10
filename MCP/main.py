# server.py

from fastmcp import FastMCP

# Create the MCP server instance
mcp = FastMCP("Calculator Server")

# Addition tool
@mcp.tool
def add(a: float, b: float) -> float:
    """Returns the sum of a and b."""
    return a + b

# Subtraction tool
@mcp.tool
def subtract(a: float, b: float) -> float:
    """Returns the difference of a and b."""
    return a - b

# Multiplication tool
@mcp.tool
def multiply(a: float, b: float) -> float:
    """Returns the product of a and b."""
    return a * b

# Division tool
@mcp.tool
def divide(a: float, b: float) -> float:
    """Returns the quotient of a and b."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

# Run the server
if __name__ == "__main__":
    mcp.run()  # Defaults to STDIO transport for local testing