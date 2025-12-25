"""
Example: Model Context Protocol (MCP) Implementation
====================================================

This example shows how to implement an MCP server that provides AI models
with access to external resources (files, databases, APIs) in a standardized way.

MCP enables:
- Standardized resource access
- Tool calling for AI models
- Prompt template management
- Secure context sharing
"""

import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import os

# Configure logging
logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources available through MCP"""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    MEMORY = "memory"


@dataclass
class Resource:
    """Represents a resource accessible through MCP"""
    uri: str
    type: ResourceType
    name: str
    description: str
    metadata: Dict[str, Any]


@dataclass
class Tool:
    """Represents a tool that AI can call"""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]


class MCPServer:
    """
    MCP Server implementation providing resources and tools to AI models.
    
    This follows the Model Context Protocol specification for standardized
    AI-to-resource communication.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.resources: Dict[str, Resource] = {}
        self.tools: Dict[str, Tool] = {}
        self.prompts: Dict[str, str] = {}
        
        logger.info(f"MCP Server '{name}' initialized")
    
    def register_resource(self, resource: Resource):
        """Register a resource that AI can access"""
        self.resources[resource.uri] = resource
        logger.info(f"Registered resource: {resource.name} ({resource.uri})")
    
    def register_tool(self, tool: Tool):
        """Register a tool that AI can call"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_prompt(self, name: str, template: str):
        """Register a reusable prompt template"""
        self.prompts[name] = template
        logger.info(f"Registered prompt: {name}")
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all available resources"""
        return [
            {
                "uri": r.uri,
                "type": r.type.value,
                "name": r.name,
                "description": r.description
            }
            for r in self.resources.values()
        ]
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
            for t in self.tools.values()
        ]
    
    def get_resource(self, uri: str) -> Optional[Any]:
        """Retrieve a resource by URI"""
        if uri not in self.resources:
            return None
        
        resource = self.resources[uri]
        
        # Handle different resource types
        if resource.type == ResourceType.FILE:
            return self._read_file(uri)
        elif resource.type == ResourceType.DATABASE:
            return self._query_database(uri)
        elif resource.type == ResourceType.API:
            return self._call_api(uri)
        elif resource.type == ResourceType.MEMORY:
            return self._read_memory(uri)
    
    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool with given parameters"""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        tool = self.tools[tool_name]
        
        try:
            result = tool.handler(**parameters)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_prompt(self, name: str, variables: Dict[str, str] = None) -> str:
        """Get a prompt template with variable substitution"""
        if name not in self.prompts:
            return None
        
        template = self.prompts[name]
        
        if variables:
            for key, value in variables.items():
                template = template.replace(f"{{{key}}}", str(value))
        
        return template
    
    # Resource handlers
    
    def _read_file(self, uri: str) -> Dict[str, Any]:
        """Read file content"""
        filepath = uri.replace("file://", "")
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            return {
                "uri": uri,
                "content": content,
                "size": len(content),
                "type": "text"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _query_database(self, uri: str) -> Dict[str, Any]:
        """Query database"""
        # Simulated database query
        return {
            "uri": uri,
            "data": [
                {"id": 1, "name": "Record 1"},
                {"id": 2, "name": "Record 2"}
            ],
            "count": 2
        }
    
    def _call_api(self, uri: str) -> Dict[str, Any]:
        """Call external API"""
        # Simulated API call
        return {
            "uri": uri,
            "response": {"status": "success", "data": "API response data"}
        }
    
    def _read_memory(self, uri: str) -> Dict[str, Any]:
        """Read from memory store"""
        # Simulated memory read
        return {
            "uri": uri,
            "value": "Stored memory content"
        }


class MCPClient:
    """
    Client for interacting with MCP server.
    
    This is used by AI models to access resources and call tools.
    """
    
    def __init__(self, server: MCPServer):
        self.server = server
        logger.info(f"MCP Client connected to '{server.name}'")
    
    def discover_resources(self) -> List[Dict[str, Any]]:
        """Discover available resources"""
        return self.server.list_resources()
    
    def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover available tools"""
        return self.server.list_tools()
    
    def access_resource(self, uri: str) -> Any:
        """Access a resource"""
        return self.server.get_resource(uri)
    
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Use a tool"""
        return self.server.call_tool(tool_name, kwargs)
    
    def get_prompt_template(self, name: str, **variables) -> str:
        """Get a prompt template"""
        return self.server.get_prompt(name, variables)


# Example Tools

def search_files(query: str, directory: str = ".") -> List[str]:
    """Search for files matching query"""
    import glob
    pattern = f"{directory}/**/*{query}*"
    results = glob.glob(pattern, recursive=True)
    return results[:10]  # Limit to 10 results


def calculate(expression: str) -> float:
    """Safely calculate mathematical expression"""
    try:
        # In production, use a safe expression evaluator
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def get_weather(city: str) -> Dict[str, Any]:
    """Get weather information (simulated)"""
    # In production, call actual weather API
    return {
        "city": city,
        "temperature": 72,
        "condition": "Sunny",
        "humidity": 45
    }


def main():
    """
    Demonstrate MCP server and client usage.
    """
    # Configure logging for the demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    print("\n" + "="*60)
    print("MODEL CONTEXT PROTOCOL (MCP) DEMONSTRATION")
    print("="*60)
    
    # Initialize MCP Server
    print("\n[1] Initializing MCP Server...")
    server = MCPServer(name="DocumentAnalysisServer")
    
    # Register resources
    print("\n[2] Registering Resources...")
    server.register_resource(Resource(
        uri="file://./documents/report.txt",
        type=ResourceType.FILE,
        name="Q4 Report",
        description="Quarterly business report",
        metadata={"category": "financial"}
    ))
    
    server.register_resource(Resource(
        uri="db://customers",
        type=ResourceType.DATABASE,
        name="Customer Database",
        description="Customer records and information",
        metadata={"table": "customers"}
    ))
    
    server.register_resource(Resource(
        uri="api://weather",
        type=ResourceType.API,
        name="Weather API",
        description="Current weather information",
        metadata={"provider": "WeatherAPI"}
    ))
    
    # Register tools
    print("\n[3] Registering Tools...")
    server.register_tool(Tool(
        name="search_files",
        description="Search for files by name",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "directory": {"type": "string", "description": "Directory to search"}
        },
        handler=search_files
    ))
    
    server.register_tool(Tool(
        name="calculate",
        description="Perform mathematical calculations",
        parameters={
            "expression": {"type": "string", "description": "Mathematical expression"}
        },
        handler=calculate
    ))
    
    server.register_tool(Tool(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "city": {"type": "string", "description": "City name"}
        },
        handler=get_weather
    ))
    
    # Register prompts
    print("\n[4] Registering Prompt Templates...")
    server.register_prompt(
        "analyze_document",
        "Analyze the following document and extract:\n"
        "1. Main topics\n"
        "2. Key findings\n"
        "3. Action items\n\n"
        "Document: {document_content}"
    )
    
    server.register_prompt(
        "summarize_with_context",
        "Using the following context:\n{context}\n\n"
        "Summarize: {text}"
    )
    
    # Initialize client
    print("\n[5] Initializing MCP Client...")
    client = MCPClient(server)
    
    # Demonstrate resource discovery
    print("\n[6] Discovering Resources...")
    resources = client.discover_resources()
    print(f"\nAvailable Resources ({len(resources)}):")
    for r in resources:
        print(f"  • {r['name']}: {r['uri']}")
        print(f"    {r['description']}")
    
    # Demonstrate tool discovery
    print("\n[7] Discovering Tools...")
    tools = client.discover_tools()
    print(f"\nAvailable Tools ({len(tools)}):")
    for t in tools:
        print(f"  • {t['name']}: {t['description']}")
    
    # Demonstrate resource access
    print("\n[8] Accessing Resources...")
    print("\nReading file resource:")
    file_content = client.access_resource("file://./documents/report.txt")
    if file_content and "content" in file_content:
        print(f"  Content preview: {file_content['content'][:100]}...")
    
    # Demonstrate tool usage
    print("\n[9] Using Tools...")
    
    # Tool 1: Calculate
    print("\nCalculating 25 * 4 + 10:")
    calc_result = client.use_tool("calculate", expression="25 * 4 + 10")
    print(f"  Result: {calc_result}")
    
    # Tool 2: Get weather
    print("\nGetting weather for London:")
    weather_result = client.use_tool("get_weather", city="London")
    print(f"  Result: {json.dumps(weather_result, indent=2)}")
    
    # Demonstrate prompt templates
    print("\n[10] Using Prompt Templates...")
    prompt = client.get_prompt_template(
        "analyze_document",
        document_content="Sample document content here..."
    )
    print(f"\nGenerated prompt:\n{prompt[:200]}...")
    
    # Demonstrate AI using MCP
    print("\n[11] AI Model Using MCP...")
    demonstrate_ai_mcp_usage(client)
    
    print("\n" + "="*60)
    print("MCP DEMONSTRATION COMPLETE")
    print("="*60)
    print("\n✓ Resources: AI can access files, databases, APIs")
    print("✓ Tools: AI can call functions and utilities")
    print("✓ Prompts: Reusable templates for common tasks")
    print("✓ Standardized: Consistent interface for all resources")


def demonstrate_ai_mcp_usage(client: MCPClient):
    """
    Simulate how an AI model would use MCP to accomplish a task.
    """
    print("\n📋 Task: AI needs to analyze a document and check weather")
    
    # Step 1: AI discovers available resources
    print("\n  1. AI discovers resources...")
    resources = client.discover_resources()
    print(f"     Found {len(resources)} resources")
    
    # Step 2: AI selects relevant resource
    print("\n  2. AI accesses document resource...")
    doc = client.access_resource("file://./documents/report.txt")
    print(f"     Retrieved document (simulated)")
    
    # Step 3: AI gets prompt template
    print("\n  3. AI gets analysis prompt template...")
    prompt = client.get_prompt_template(
        "analyze_document",
        document_content="Document content here..."
    )
    print(f"     Generated analysis prompt")
    
    # Step 4: AI uses tools
    print("\n  4. AI calls weather tool...")
    weather = client.use_tool("get_weather", city="London")
    print(f"     Weather: {weather['result']['temperature']}°F, {weather['result']['condition']}")
    
    print("\n  ✓ AI successfully used MCP to:")
    print("    - Access external resources")
    print("    - Use tools for calculations")
    print("    - Generate contextual prompts")
    print("    - Combine information from multiple sources")


if __name__ == "__main__":
    main()
    
    print("\n💡 KEY TAKEAWAYS:")
    print("="*60)
    print("• MCP provides standardized interface for AI-to-resource communication")
    print("• Resources can be files, databases, APIs, or memory stores")
    print("• Tools allow AI to perform actions (search, calculate, etc.)")
    print("• Prompt templates enable reusable, context-aware prompts")
    print("• This enables AI models to work with external data safely")
    print("\n📚 See MCP documentation: ./docs/protocols/MCP.md")
