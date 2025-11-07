# MCP - Model Context Protocol

## Overview

**Model Context Protocol** (MCP) is an open standard for sharing context between AI models, tools, and applications. It provides a standardized way for language models to access external data sources, tools, and services securely and efficiently.

## Key Characteristics

- **Type**: Communication protocol
- **Purpose**: Context sharing and tool integration
- **Architecture**: Client-Server
- **Transport**: JSON-RPC 2.0
- **Providers**: Anthropic, others

## What Problem Does MCP Solve?

**Before MCP:**
```
AI Model → Custom Integration 1 → Database
         → Custom Integration 2 → File System
         → Custom Integration 3 → API Service
         → Custom Integration 4 → Web Browser
         
Problem: Each integration is custom, non-portable, and hard to maintain
```

**With MCP:**
```
AI Model → MCP Client → MCP Protocol → Multiple Servers
                                        ↓
                        ┌───────────────┼───────────────┐
                        ▼               ▼               ▼
                   Database         File System     API Service
                   
Benefit: Standardized, portable, reusable integrations
```

## How It Works

```
┌──────────────┐
│   AI Model   │
│  (Claude,    │
│   GPT, etc)  │
└──────┬───────┘
       │
┌──────▼───────┐
│  MCP Client  │ ← Embedded in app (Claude Desktop, IDEs, etc)
└──────┬───────┘
       │ JSON-RPC 2.0
       │
┌──────▼───────┐
│  MCP Server  │ ← Provides tools, resources, prompts
└──────┬───────┘
       │
┌──────▼───────┐
│   External   │
│   Resources  │ ← Databases, APIs, filesystems, etc
└──────────────┘
```

## Architecture Components

### 1. MCP Client
- Embedded in AI applications
- Discovers available servers
- Routes requests to appropriate server
- Handles responses

### 2. MCP Server
- Exposes capabilities via protocol
- Implements tools, resources, or prompts
- Handles authentication
- Manages state

### 3. Transport Layer
- Standard I/O (stdio)
- HTTP with SSE (Server-Sent Events)
- WebSocket (future)

## Core Concepts

### Resources
Read-only data sources the model can access.

```json
{
  "uri": "file:///Users/documents/report.pdf",
  "name": "Quarterly Report",
  "mimeType": "application/pdf",
  "description": "Q3 2024 financial report"
}
```

### Tools
Functions the model can execute.

```json
{
  "name": "search_database",
  "description": "Search customer database",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "limit": {"type": "number"}
    },
    "required": ["query"]
  }
}
```

### Prompts
Pre-configured prompt templates.

```json
{
  "name": "code_review",
  "description": "Review code for issues",
  "arguments": [
    {
      "name": "language",
      "description": "Programming language",
      "required": true
    }
  ]
}
```

## Examples with Input/Output

### Example 1: File System Access

**MCP Server Configuration:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/projects"]
    }
  }
}
```

**Interaction:**

**User:** "Read the README.md file from my project"

**AI → MCP Client → MCP Server:**
```json
{
  "jsonrpc": "2.0",
  "method": "resources/read",
  "params": {
    "uri": "file:///Users/projects/myapp/README.md"
  },
  "id": 1
}
```

**MCP Server Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "contents": [
      {
        "uri": "file:///Users/projects/myapp/README.md",
        "mimeType": "text/markdown",
        "text": "# My Application\n\nThis is a sample app..."
      }
    ]
  },
  "id": 1
}
```

**AI Output:** "Your README.md contains..."

---

### Example 2: Database Query Tool

**MCP Server: Database Tool**

**User:** "How many users signed up last month?"

**AI → MCP:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "execute_sql",
    "arguments": {
      "query": "SELECT COUNT(*) FROM users WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH)"
    }
  },
  "id": 2
}
```

**MCP Server Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Query result: 1,247 users"
      }
    ]
  },
  "id": 2
}
```

**AI Output:** "Last month, 1,247 new users signed up to your service."

---

### Example 3: API Integration

**MCP Server: Weather API**

**User:** "What's the weather in Paris?"

**AI → MCP:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "location": "Paris, France"
    }
  },
  "id": 3
}
```

**MCP Server → External API → Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Temperature: 15°C, Conditions: Partly cloudy, Humidity: 65%"
      }
    ]
  },
  "id": 3
}
```

**AI Output:** "In Paris, it's currently 15°C (59°F) with partly cloudy conditions and 65% humidity."

---

### Example 4: Multi-Server Workflow

**Configuration:**
```json
{
  "mcpServers": {
    "github": {
      "command": "mcp-server-github",
      "args": ["--token", "${GITHUB_TOKEN}"]
    },
    "slack": {
      "command": "mcp-server-slack",
      "args": ["--token", "${SLACK_TOKEN}"]
    }
  }
}
```

**User:** "Check our GitHub issues and post a summary to #dev-team on Slack"

**Step 1: AI → GitHub MCP Server**
```json
{
  "method": "tools/call",
  "params": {
    "name": "list_issues",
    "arguments": {
      "repo": "company/product",
      "state": "open"
    }
  }
}
```

**Response:** 
```json
{
  "result": {
    "content": [{
      "type": "text",
      "text": "Found 23 open issues: 5 bugs, 12 features, 6 documentation..."
    }]
  }
}
```

**Step 2: AI → Slack MCP Server**
```json
{
  "method": "tools/call",
  "params": {
    "name": "post_message",
    "arguments": {
      "channel": "#dev-team",
      "text": "📊 GitHub Issues Summary:\n• 23 open issues\n• 5 bugs\n• 12 features\n• 6 documentation"
    }
  }
}
```

**Response:**
```json
{
  "result": {
    "content": [{
      "type": "text",
      "text": "Message posted successfully"
    }]
  }
}
```

**AI Output:** "I've checked your GitHub repo and posted a summary to the #dev-team Slack channel. You have 23 open issues: 5 bugs, 12 feature requests, and 6 documentation items."

---

### Example 5: Prompt Templates

**MCP Server: Code Review Prompts**

**Available Prompt:**
```json
{
  "name": "security_review",
  "description": "Review code for security issues",
  "arguments": [
    {
      "name": "code",
      "description": "Code to review",
      "required": true
    },
    {
      "name": "language",
      "description": "Programming language"
    }
  ]
}
```

**User:** "Review this code for security issues"

**AI → MCP:**
```json
{
  "method": "prompts/get",
  "params": {
    "name": "security_review",
    "arguments": {
      "code": "def login(username, password):\n    query = f\"SELECT * FROM users WHERE username='{username}' AND password='{password}'\"",
      "language": "python"
    }
  }
}
```

**MCP Response:**
```json
{
  "result": {
    "messages": [
      {
        "role": "user",
        "content": {
          "type": "text",
          "text": "Review the following Python code for security vulnerabilities:\n\n```python\ndef login(username, password):\n    query = f\"SELECT * FROM users WHERE username='{username}' AND password='{password}'\"\n```\n\nFocus on: SQL injection, authentication, password handling."
        }
      }
    ]
  }
}
```

**AI Output:** 
```
⚠️ Critical Security Issues Found:

1. SQL Injection Vulnerability (High Risk)
   - Using f-strings to build SQL queries allows SQL injection attacks
   - An attacker could input: username = "admin' --" to bypass authentication
   
2. Plain Text Password (High Risk)
   - Password is being compared in plain text
   - Should use bcrypt or similar hashing

Recommended fix:
```python
import bcrypt
from db import get_db_connection

def login(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Use parameterized query
    cursor.execute(
        "SELECT * FROM users WHERE username = ?", 
        (username,)
    )
    user = cursor.fetchone()
    
    # Verify hashed password
    if user and bcrypt.checkpw(password.encode(), user['password_hash']):
        return user
    return None
```
```

---

## Building an MCP Server

### Python Example

```python
# mcp_server_example.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import sqlite3

app = Server("my-database-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_users",
            description="Query users database",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum results"
                    }
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_users":
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        
        limit = arguments.get("limit", 10)
        cursor.execute(f"SELECT * FROM users LIMIT {limit}")
        results = cursor.fetchall()
        
        return [TextContent(
            type="text",
            text=f"Found {len(results)} users"
        )]

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    
    asyncio.run(stdio_server(app))
```

### TypeScript Example

```typescript
// mcp-server-example.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "my-file-server",
  version: "1.0.0"
});

// List available tools
server.setRequestHandler("tools/list", async () => {
  return {
    tools: [
      {
        name: "read_file",
        description: "Read contents of a file",
        inputSchema: {
          type: "object",
          properties: {
            path: { type: "string" }
          },
          required: ["path"]
        }
      }
    ]
  };
});

// Handle tool calls
server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "read_file") {
    const fs = require("fs");
    const content = fs.readFileSync(request.params.arguments.path, "utf-8");
    
    return {
      content: [
        {
          type: "text",
          text: content
        }
      ]
    };
  }
});

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
```

## MCP Server Registry

Popular open-source MCP servers:

| Server | Purpose | Provider |
|--------|---------|----------|
| filesystem | File system access | Anthropic |
| github | GitHub integration | Anthropic |
| gitlab | GitLab integration | Anthropic |
| google-maps | Maps and location | Anthropic |
| postgres | PostgreSQL database | Anthropic |
| sqlite | SQLite database | Anthropic |
| puppeteer | Web automation | Anthropic |
| slack | Slack integration | Anthropic |
| fetch | HTTP requests | Anthropic |

## Use Cases

✅ **Best For:**
- Connecting AI to databases
- File system access
- API integration
- Multi-tool workflows
- Reusable integrations
- Secure context sharing
- Custom tool development

❌ **Not Suitable For:**
- Direct model-to-model communication (use A2A)
- Real-time streaming
- Peer-to-peer connections
- Low-latency requirements

## Advantages

- **Standardization**: One protocol for all integrations
- **Security**: Controlled access to resources
- **Portability**: Works across different AI models
- **Composability**: Combine multiple servers
- **Maintainability**: Centralized tool management
- **Open Standard**: Community-driven

## Limitations

- Relatively new (as of 2024)
- Limited to Anthropic's Claude initially
- Requires server development
- Transport overhead (JSON-RPC)
- No built-in authentication standard yet

## MCP vs Other Approaches

| Approach | MCP | Function Calling | Plugins |
|----------|-----|------------------|---------|
| Standard | ✅ Open | ❌ Vendor-specific | ❌ Platform-specific |
| Portability | ✅ High | ❌ Low | ❌ Low |
| Security | ✅ Controlled | ⚠️ Varies | ⚠️ Varies |
| Complexity | Medium | Low | High |

## Configuration Example

**Claude Desktop Configuration:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/Documents"
      ]
    },
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_TOKEN": "your-token-here"
      }
    },
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://localhost/mydb"
      ]
    }
  }
}
```

## Future of MCP

- Broader model support (beyond Claude)
- Enhanced authentication/authorization
- WebSocket transport
- Streaming support
- GraphQL-style queries
- Built-in caching
- Server marketplace

## Getting Started

1. **Install MCP SDK:**
   ```bash
   npm install @modelcontextprotocol/sdk
   ```

2. **Create a simple server:**
   ```bash
   npx create-mcp-server my-server
   ```

3. **Configure your AI app:**
   Add server to config file

4. **Test it:**
   Ask your AI to use the new tool

---

**Related:**  
- [A2A - Agent-to-Agent Communication](./A2A.md) →
- [Multi-Agent Orchestration](./ORCHESTRATION.md) →
