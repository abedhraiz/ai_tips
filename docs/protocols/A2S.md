# A2S - Agent-to-System Communication

## Overview

Agent-to-System (A2S) communication defines how AI agents interact with external systems, databases, APIs, and infrastructure. This protocol enables AI agents to read from and write to external systems while maintaining security, reliability, and observability.

## Key Principles

1. **Authentication & Authorization**: Secure access control
2. **Idempotency**: Safe retries without side effects
3. **Error Handling**: Graceful degradation and recovery
4. **Logging & Monitoring**: Observable operations
5. **Rate Limiting**: Respect system constraints

## Communication Patterns

### 1. Database Operations

AI agents querying and updating databases.

```python
import openai
import psycopg2
import json
import os

class DatabaseAgent:
    def __init__(self, db_config):
        self.client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        
    def natural_language_query(self, question):
        """Convert natural language to SQL and execute"""
        # Get database schema
        schema = self._get_schema()
        
        # Generate SQL from natural language
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a SQL expert. Convert natural language to SQL queries.
                    
Database schema:
{schema}

Rules:
- Return only the SQL query, no explanation
- Use proper joins and WHERE clauses
- Ensure query is safe (read-only when possible)"""
                },
                {"role": "user", "content": question}
            ]
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Validate query safety
        if self._is_safe_query(sql_query):
            # Execute query
            self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            
            # Format results
            return self._format_results(results, sql_query)
        else:
            return {"error": "Query contains potentially unsafe operations"}
    
    def _get_schema(self):
        """Get database schema"""
        self.cursor.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """)
        
        schema = {}
        for table, column, dtype in self.cursor.fetchall():
            if table not in schema:
                schema[table] = []
            schema[table].append(f"{column} ({dtype})")
        
        return "\n".join([f"{table}: {', '.join(cols)}" for table, cols in schema.items()])
    
    def _is_safe_query(self, query):
        """Check if query is safe"""
        unsafe_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        query_upper = query.upper()
        return not any(keyword in query_upper for keyword in unsafe_keywords)
    
    def _format_results(self, results, query):
        """Format query results"""
        return {
            "query": query,
            "row_count": len(results),
            "results": results[:100]  # Limit to 100 rows
        }
    
    def close(self):
        """Close database connection"""
        self.cursor.close()
        self.conn.close()

# Usage
db_config = {
    "host": "localhost",
    "database": "myapp",
    "user": "user",
    "password": "password"
}

agent = DatabaseAgent(db_config)

# Natural language database query
result = agent.natural_language_query(
    "Show me the top 10 customers by total purchase amount this year"
)

print(f"SQL: {result['query']}")
print(f"Found {result['row_count']} results")
for row in result['results'][:5]:
    print(row)

agent.close()
```

**Benefits**: Natural language data access, automatic SQL generation, safe operations

---

### 2. REST API Integration

AI agents calling external APIs.

```python
import openai
import requests
import json
from typing import Dict, Any

class APIAgent:
    def __init__(self):
        self.client = openai.OpenAI()
        self.api_specs = {}
        
    def register_api(self, name: str, base_url: str, spec: Dict[str, Any]):
        """Register an API with its specification"""
        self.api_specs[name] = {
            "base_url": base_url,
            "spec": spec
        }
    
    def call_api(self, instruction: str):
        """Call API based on natural language instruction"""
        # Determine which API and endpoint to call
        api_selection = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""Select the appropriate API and endpoint for the user's request.
                    
Available APIs:
{json.dumps(self.api_specs, indent=2)}

Return JSON with: {{"api": "name", "endpoint": "path", "method": "GET/POST", "params": {{}}}}"""
                },
                {"role": "user", "content": instruction}
            ],
            response_format={"type": "json_object"}
        )
        
        api_call = json.loads(api_selection.choices[0].message.content)
        
        # Execute API call
        return self._execute_api_call(api_call)
    
    def _execute_api_call(self, call_spec: Dict[str, Any]):
        """Execute the API call"""
        api_name = call_spec['api']
        api_config = self.api_specs[api_name]
        
        url = f"{api_config['base_url']}{call_spec['endpoint']}"
        method = call_spec['method']
        params = call_spec.get('params', {})
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=params, timeout=30)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            response.raise_for_status()
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": response.json()
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e)
            }

# Usage
agent = APIAgent()

# Register weather API
agent.register_api(
    "weather",
    "https://api.openweathermap.org/data/2.5",
    {
        "endpoints": {
            "/weather": "Get current weather for a city",
            "/forecast": "Get weather forecast"
        }
    }
)

# Natural language API call
result = agent.call_api("Get the current weather in London")
print(json.dumps(result, indent=2))
```

**Benefits**: Natural language API access, automatic endpoint selection, error handling

---

### 3. Event-Driven Integration

AI agents responding to system events.

```python
import asyncio
import json
from datetime import datetime

class EventDrivenAgent:
    def __init__(self):
        self.client = openai.OpenAI()
        self.event_handlers = {}
        self.event_queue = asyncio.Queue()
        
    def register_handler(self, event_type: str, handler):
        """Register event handler"""
        self.event_handlers[event_type] = handler
    
    async def process_event(self, event):
        """Process incoming event"""
        event_type = event.get('type')
        
        if event_type in self.event_handlers:
            handler = self.event_handlers[event_type]
            await handler(event)
        else:
            # Use AI to determine how to handle unknown event
            await self._ai_handle_event(event)
    
    async def _ai_handle_event(self, event):
        """AI determines how to handle event"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze this system event and suggest appropriate actions. Return JSON with 'analysis' and 'suggested_actions'."
                },
                {"role": "user", "content": json.dumps(event)}
            ],
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        
        # Log analysis
        print(f"[{datetime.now()}] Event Analysis:")
        print(f"  Type: {event.get('type')}")
        print(f"  Analysis: {analysis['analysis']}")
        print(f"  Suggested Actions: {analysis['suggested_actions']}")
        
        return analysis
    
    async def start_processing(self):
        """Start processing events from queue"""
        while True:
            event = await self.event_queue.get()
            await self.process_event(event)
            self.event_queue.task_done()

# Example handlers
async def handle_error_event(event):
    """Handle system errors"""
    error_msg = event.get('message')
    severity = event.get('severity', 'medium')
    
    print(f"[ERROR] {severity.upper()}: {error_msg}")
    
    if severity == 'critical':
        # Alert on-call engineer
        pass  # Send alert via PagerDuty, Slack, etc.

async def handle_user_signup(event):
    """Handle new user signups"""
    user_id = event.get('user_id')
    email = event.get('email')
    
    print(f"[SIGNUP] New user: {user_id} ({email})")
    
    # Send welcome email
    # Add to onboarding sequence
    pass

# Usage
agent = EventDrivenAgent()

# Register handlers
agent.register_handler('error', handle_error_event)
agent.register_handler('user_signup', handle_user_signup)

# Simulate events
async def simulate_events():
    # Error event
    await agent.event_queue.put({
        'type': 'error',
        'message': 'Database connection timeout',
        'severity': 'high',
        'timestamp': datetime.now().isoformat()
    })
    
    # User signup
    await agent.event_queue.put({
        'type': 'user_signup',
        'user_id': '12345',
        'email': 'user@example.com',
        'timestamp': datetime.now().isoformat()
    })
    
    # Unknown event type
    await agent.event_queue.put({
        'type': 'payment_processed',
        'amount': 99.99,
        'currency': 'USD',
        'timestamp': datetime.now().isoformat()
    })

# Run
# asyncio.run(simulate_events())
# asyncio.run(agent.start_processing())
```

**Benefits**: Reactive system integration, automatic event analysis, flexible handling

---

### 4. File System Operations

AI agents reading and writing files.

```python
import openai
import os
from pathlib import Path
import json

class FileSystemAgent:
    def __init__(self, workspace_dir):
        self.client = openai.OpenAI()
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
    def process_files(self, instruction: str):
        """Process files based on instruction"""
        # List available files
        files = list(self.workspace.glob('*'))
        file_list = "\n".join([f"- {f.name}" for f in files])
        
        # Determine operations needed
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a file system assistant. Available files:
{file_list}

Determine what file operations are needed. Return JSON with 'operations' array.
Each operation: {{"action": "read/write/delete/create", "file": "filename", "content": "..."}}"""
                },
                {"role": "user", "content": instruction}
            ],
            response_format={"type": "json_object"}
        )
        
        plan = json.loads(response.choices[0].message.content)
        
        # Execute operations
        results = []
        for op in plan['operations']:
            result = self._execute_operation(op)
            results.append(result)
        
        return results
    
    def _execute_operation(self, operation):
        """Execute a single file operation"""
        action = operation['action']
        filename = operation['file']
        filepath = self.workspace / filename
        
        try:
            if action == 'read':
                content = filepath.read_text()
                return {"action": "read", "file": filename, "content": content}
                
            elif action == 'write':
                content = operation.get('content', '')
                filepath.write_text(content)
                return {"action": "write", "file": filename, "success": True}
                
            elif action == 'delete':
                filepath.unlink()
                return {"action": "delete", "file": filename, "success": True}
                
            elif action == 'create':
                content = operation.get('content', '')
                filepath.write_text(content)
                return {"action": "create", "file": filename, "success": True}
                
        except Exception as e:
            return {"action": action, "file": filename, "error": str(e)}
    
    def analyze_file(self, filename: str):
        """Analyze file content"""
        filepath = self.workspace / filename
        
        if not filepath.exists():
            return {"error": "File not found"}
        
        content = filepath.read_text()
        
        # AI analyzes the file
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze this file and provide insights. Return JSON with 'summary', 'key_points', and 'suggestions'."
                },
                {"role": "user", "content": f"File: {filename}\n\n{content}"}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

# Usage
agent = FileSystemAgent(workspace_dir="./workspace")

# Process files with natural language
results = agent.process_files("Create a summary of all .txt files and save it to summary.md")

for result in results:
    print(f"Operation: {result}")

# Analyze specific file
analysis = agent.analyze_file("report.txt")
print(json.dumps(analysis, indent=2))
```

**Benefits**: Natural language file operations, intelligent content analysis, safe operations

---

### 5. System Monitoring and Alerting

AI agents monitoring system health.

```python
import psutil
import openai
from datetime import datetime
import time

class MonitoringAgent:
    def __init__(self, alert_threshold=80):
        self.client = openai.OpenAI()
        self.alert_threshold = alert_threshold
        self.alert_history = []
        
    def check_system_health(self):
        """Check system resources"""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check for issues
        issues = []
        if metrics['cpu_percent'] > self.alert_threshold:
            issues.append(f"High CPU usage: {metrics['cpu_percent']}%")
        if metrics['memory_percent'] > self.alert_threshold:
            issues.append(f"High memory usage: {metrics['memory_percent']}%")
        if metrics['disk_percent'] > self.alert_threshold:
            issues.append(f"High disk usage: {metrics['disk_percent']}%")
        
        if issues:
            self._handle_alert(metrics, issues)
        
        return metrics
    
    def _handle_alert(self, metrics, issues):
        """Handle system alerts with AI analysis"""
        # AI analyzes the issue
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a system administrator. Analyze these metrics and issues, then suggest remediation steps."
                },
                {
                    "role": "user",
                    "content": f"Metrics: {metrics}\nIssues: {issues}"
                }
            ]
        )
        
        analysis = response.choices[0].message.content
        
        alert = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "issues": issues,
            "analysis": analysis
        }
        
        self.alert_history.append(alert)
        
        # Send alert (email, Slack, PagerDuty, etc.)
        print(f"\n🚨 ALERT at {alert['timestamp']}")
        print(f"Issues: {', '.join(issues)}")
        print(f"\nAnalysis:\n{analysis}")
        
        return alert
    
    def continuous_monitoring(self, interval=60):
        """Continuously monitor system"""
        print("Starting continuous monitoring...")
        try:
            while True:
                metrics = self.check_system_health()
                print(f"[{metrics['timestamp']}] CPU: {metrics['cpu_percent']}% | RAM: {metrics['memory_percent']}% | Disk: {metrics['disk_percent']}%")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

# Usage
agent = MonitoringAgent(alert_threshold=80)

# Single check
metrics = agent.check_system_health()
print(f"System Status: {metrics}")

# Continuous monitoring
# agent.continuous_monitoring(interval=60)
```

**Benefits**: Intelligent system monitoring, automatic issue detection, AI-powered remediation suggestions

---

## A2S Best Practices

### 1. Authentication and Security

```python
class SecureSystemAgent:
    def __init__(self, api_key, system_credentials):
        self.api_key = api_key
        self.credentials = system_credentials
        
    def authenticate_request(self, request):
        """Authenticate before system access"""
        # Verify API key
        # Check permissions
        # Log access
        pass
    
    def encrypt_sensitive_data(self, data):
        """Encrypt sensitive information"""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        cipher = Fernet(key)
        return cipher.encrypt(data.encode())
```

### 2. Rate Limiting

```python
from functools import wraps
import time

def rate_limit(calls_per_minute=60):
    """Rate limiting decorator"""
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    return decorator

@rate_limit(calls_per_minute=30)
def call_external_api(endpoint):
    # API call implementation
    pass
```

### 3. Retry Logic with Exponential Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def reliable_system_call(system_function):
    """Retry failed system calls"""
    return system_function()
```

### 4. Comprehensive Logging

```python
import logging
from datetime import datetime

class LoggingSystemAgent:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agent_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SystemAgent')
    
    def execute_with_logging(self, operation, **kwargs):
        """Execute operation with comprehensive logging"""
        self.logger.info(f"Starting operation: {operation}")
        self.logger.debug(f"Parameters: {kwargs}")
        
        try:
            result = self._perform_operation(operation, **kwargs)
            self.logger.info(f"Operation {operation} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Operation {operation} failed: {str(e)}", exc_info=True)
            raise
```

### 5. Health Checks

```python
class HealthCheckAgent:
    def health_check(self):
        """Comprehensive system health check"""
        checks = {
            "database": self._check_database(),
            "api": self._check_external_apis(),
            "disk_space": self._check_disk_space(),
            "memory": self._check_memory()
        }
        
        overall_status = "healthy" if all(checks.values()) else "unhealthy"
        
        return {
            "status": overall_status,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
```

## Error Handling Patterns

```python
class RobustSystemAgent:
    def safe_system_operation(self, operation):
        """Execute system operation with comprehensive error handling"""
        try:
            return operation()
        
        except ConnectionError:
            # Network issues
            return {"error": "connection_failed", "retry": True}
        
        except PermissionError:
            # Access denied
            return {"error": "permission_denied", "retry": False}
        
        except TimeoutError:
            # Operation timeout
            return {"error": "timeout", "retry": True}
        
        except Exception as e:
            # Unexpected error
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            return {"error": "unexpected", "message": str(e), "retry": False}
```

---

**Related Protocols**: [MCP](./MCP.md) | [A2A](./A2A.md) | [A2P](./A2P.md) | [Orchestration](./ORCHESTRATION.md)
