"""
Manager Assistant - Multi-Agent AI System
=========================================

Demonstrates an intelligent manager assistant where multiple specialized agents
collaborate autonomously to handle scheduling, emails, data analysis, research,
and reporting without human intervention.

This showcases Agent-to-Agent (A2A) communication for complex business workflows.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import openai

# Configure logging
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles of agents in the manager assistant system"""
    COORDINATOR = "coordinator"
    SCHEDULING = "scheduling"
    EMAIL = "email"
    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"
    REPORT = "report"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class Task:
    """Task structure for agent workflow"""
    task_id: str
    description: str
    assigned_to: AgentRole
    priority: TaskPriority
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentMessage:
    """Message structure for A2A communication"""
    sender: AgentRole
    recipient: AgentRole
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message_type: str = "request"  # request, response, notification, collaboration


class BaseAssistantAgent:
    """Base class for all manager assistant agents"""
    
    def __init__(self, role: AgentRole, name: str, llm_client):
        self.role = role
        self.name = name
        self.llm_client = llm_client
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.messages: List[AgentMessage] = []
        
        print(f"  ✓ {self.name} initialized")
    
    async def send_message(self, recipient: AgentRole, content: Dict[str, Any], 
                          message_type: str = "request"):
        """Send A2A message to another agent"""
        message = AgentMessage(
            sender=self.role,
            recipient=recipient,
            content=content,
            message_type=message_type
        )
        self.messages.append(message)
        print(f"    📤 {self.name} → {recipient.value}: {message_type}")
        return message
    
    async def process_with_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Process task using LLM"""
        try:
            messages = [
                {"role": "system", "content": system_prompt or self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_system_prompt(self) -> str:
        return f"You are a {self.name} in a manager assistant system."
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Override in subclasses"""
        raise NotImplementedError


class CoordinatorAgent(BaseAssistantAgent):
    """
    Coordinates all other agents and manages workflow.
    
    Analyzes manager requests and distributes tasks to appropriate agents.
    """
    
    def __init__(self, llm_client):
        super().__init__(AgentRole.COORDINATOR, "Coordinator Agent", llm_client)
    
    def get_system_prompt(self) -> str:
        return """You are a coordinator agent for a manager assistant system.
        Analyze manager requests and break them down into tasks for specialized agents:
        - SCHEDULING: Calendar management, meetings, appointments
        - EMAIL: Email triage, drafting, responses
        - DATA_ANALYSIS: Reports, metrics, business intelligence
        - RESEARCH: Information gathering, competitive intelligence
        - REPORT: Compile information into actionable reports
        
        Return JSON with tasks array: [{"agent": "...", "action": "...", "priority": 1-5}]"""
    
    async def analyze_request(self, manager_request: str) -> List[Task]:
        """Analyze manager request and create task plan"""
        
        print(f"\n🎯 {self.name} analyzing request...")
        
        prompt = f"""Analyze this manager request and create a task plan:
        
        Request: "{manager_request}"
        
        Break it down into specific tasks for each agent. Return JSON array."""
        
        response = await self.process_with_llm(prompt)
        
        try:
            task_plan = json.loads(response)
            
            tasks = []
            for idx, task_def in enumerate(task_plan):
                task = Task(
                    task_id=f"TASK_{idx+1}_{datetime.now().timestamp()}",
                    description=task_def.get("action", ""),
                    assigned_to=AgentRole[task_def.get("agent", "COORDINATOR").upper()],
                    priority=TaskPriority(task_def.get("priority", 3))
                )
                tasks.append(task)
                print(f"  ✓ Created task: {task.assigned_to.value} - {task.description}")
            
            return tasks
        
        except Exception as e:
            print(f"  ⚠️ Error parsing task plan: {e}")
            # Return default task
            return [Task(
                task_id=f"TASK_DEFAULT_{datetime.now().timestamp()}",
                description=manager_request,
                assigned_to=AgentRole.DATA_ANALYSIS,
                priority=TaskPriority.MEDIUM
            )]


class SchedulingAgent(BaseAssistantAgent):
    """
    Manages calendar, scheduling, and meeting coordination.
    """
    
    def __init__(self, llm_client):
        super().__init__(AgentRole.SCHEDULING, "Scheduling Agent", llm_client)
        
        # Simulated calendar data
        self.calendar = {
            "today": [
                {"time": "09:00", "title": "Team Standup", "duration": 30},
                {"time": "14:00", "title": "Client Review", "duration": 60}
            ],
            "week": [
                {"day": "Monday", "slots_available": 4},
                {"day": "Tuesday", "slots_available": 6},
                {"day": "Wednesday", "slots_available": 3},
                {"day": "Thursday", "slots_available": 5},
                {"day": "Friday", "slots_available": 2}
            ]
        }
    
    def get_system_prompt(self) -> str:
        return """You are a scheduling agent. Manage calendar, find meeting times,
        coordinate schedules, and handle time-related tasks. Be precise with dates
        and times."""
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute scheduling task"""
        
        print(f"\n📅 {self.name} processing: {task.description}")
        
        prompt = f"""Handle this scheduling task:
        
        Task: {task.description}
        
        Calendar Data:
        {json.dumps(self.calendar, indent=2)}
        
        Provide a clear response with specific times and dates."""
        
        response = await self.process_with_llm(prompt)
        
        task.status = "completed"
        task.completed_at = datetime.now().isoformat()
        task.result = {
            "agent": "scheduling",
            "response": response,
            "calendar_checked": True
        }
        
        return task.result


class EmailAgent(BaseAssistantAgent):
    """
    Handles email triage, drafting, and responses.
    """
    
    def __init__(self, llm_client):
        super().__init__(AgentRole.EMAIL, "Email Agent", llm_client)
        
        # Simulated inbox
        self.inbox = [
            {"from": "client@company.com", "subject": "Project Update Needed", "priority": "high"},
            {"from": "team@company.com", "subject": "Budget Approval", "priority": "medium"},
            {"from": "vendor@supplier.com", "subject": "Invoice #12345", "priority": "low"}
        ]
    
    def get_system_prompt(self) -> str:
        return """You are an email management agent. Triage emails, draft responses,
        and handle email-related tasks. Be professional and concise."""
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute email task"""
        
        print(f"\n📧 {self.name} processing: {task.description}")
        
        prompt = f"""Handle this email task:
        
        Task: {task.description}
        
        Inbox:
        {json.dumps(self.inbox, indent=2)}
        
        Provide email summary or draft response as appropriate."""
        
        response = await self.process_with_llm(prompt)
        
        task.status = "completed"
        task.completed_at = datetime.now().isoformat()
        task.result = {
            "agent": "email",
            "response": response,
            "emails_processed": len(self.inbox)
        }
        
        return task.result


class DataAnalysisAgent(BaseAssistantAgent):
    """
    Analyzes business data, metrics, and KPIs.
    """
    
    def __init__(self, llm_client):
        super().__init__(AgentRole.DATA_ANALYSIS, "Data Analysis Agent", llm_client)
        
        # Simulated business data
        self.data = {
            "sales": {
                "current_month": 250000,
                "last_month": 230000,
                "growth": "+8.7%"
            },
            "customers": {
                "total": 1250,
                "new_this_month": 45,
                "churn_rate": "2.1%"
            },
            "team": {
                "size": 15,
                "productivity_score": 87,
                "satisfaction": "4.2/5"
            }
        }
    
    def get_system_prompt(self) -> str:
        return """You are a data analysis agent. Analyze business metrics, identify
        trends, and provide data-driven insights. Be specific with numbers and
        provide actionable recommendations."""
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute data analysis task"""
        
        print(f"\n📊 {self.name} analyzing: {task.description}")
        
        prompt = f"""Analyze this data request:
        
        Task: {task.description}
        
        Available Data:
        {json.dumps(self.data, indent=2)}
        
        Provide analysis with insights and recommendations."""
        
        response = await self.process_with_llm(prompt)
        
        task.status = "completed"
        task.completed_at = datetime.now().isoformat()
        task.result = {
            "agent": "data_analysis",
            "response": response,
            "data_points_analyzed": len(self.data)
        }
        
        return task.result


class ResearchAgent(BaseAssistantAgent):
    """
    Gathers information and competitive intelligence.
    """
    
    def __init__(self, llm_client):
        super().__init__(AgentRole.RESEARCH, "Research Agent", llm_client)
    
    def get_system_prompt(self) -> str:
        return """You are a research agent. Gather information, analyze trends,
        and provide competitive intelligence. Be thorough and cite sources."""
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute research task"""
        
        print(f"\n🔍 {self.name} researching: {task.description}")
        
        prompt = f"""Research this topic:
        
        Task: {task.description}
        
        Provide comprehensive information with key findings."""
        
        response = await self.process_with_llm(prompt)
        
        task.status = "completed"
        task.completed_at = datetime.now().isoformat()
        task.result = {
            "agent": "research",
            "response": response,
            "sources_consulted": "simulated"
        }
        
        return task.result


class ReportAgent(BaseAssistantAgent):
    """
    Compiles information from other agents into actionable reports.
    """
    
    def __init__(self, llm_client):
        super().__init__(AgentRole.REPORT, "Report Agent", llm_client)
    
    def get_system_prompt(self) -> str:
        return """You are a report compilation agent. Combine information from
        multiple sources into clear, actionable reports. Use professional formatting
        and highlight key insights."""
    
    async def compile_report(self, manager_request: str, 
                            task_results: List[Dict[str, Any]]) -> str:
        """Compile results from all agents into final report"""
        
        print(f"\n📋 {self.name} compiling final report...")
        
        prompt = f"""Compile a comprehensive report for this manager request:
        
        Request: "{manager_request}"
        
        Agent Results:
        {json.dumps(task_results, indent=2)}
        
        Create a well-structured report with:
        1. Executive Summary
        2. Key Findings
        3. Detailed Analysis
        4. Recommendations
        5. Next Steps"""
        
        report = await self.process_with_llm(prompt)
        
        return report


class ManagerAssistantOrchestrator:
    """
    Orchestrates the multi-agent manager assistant system.
    
    Coordinates agent collaboration to handle manager requests autonomously.
    """
    
    def __init__(self, api_key: str):
        self.llm_client = openai.OpenAI(api_key=api_key)
        
        # Initialize all agents
        self.coordinator = CoordinatorAgent(self.llm_client)
        self.scheduling_agent = SchedulingAgent(self.llm_client)
        self.email_agent = EmailAgent(self.llm_client)
        self.data_agent = DataAnalysisAgent(self.llm_client)
        self.research_agent = ResearchAgent(self.llm_client)
        self.report_agent = ReportAgent(self.llm_client)
        
        # Agent registry
        self.agents = {
            AgentRole.COORDINATOR: self.coordinator,
            AgentRole.SCHEDULING: self.scheduling_agent,
            AgentRole.EMAIL: self.email_agent,
            AgentRole.DATA_ANALYSIS: self.data_agent,
            AgentRole.RESEARCH: self.research_agent,
            AgentRole.REPORT: self.report_agent
        }
        
        print("\n" + "="*60)
        print("🏢 MANAGER ASSISTANT SYSTEM INITIALIZED")
        print("="*60)
    
    async def handle_request(self, manager_request: str) -> Dict[str, Any]:
        """
        Process manager request through multi-agent system.
        
        Workflow:
        1. Coordinator analyzes request and creates task plan
        2. Tasks distributed to specialized agents
        3. Agents execute tasks (may collaborate via A2A)
        4. Report agent compiles results
        5. Final report delivered to manager
        """
        
        print(f"\n\n{'='*60}")
        print(f"👔 MANAGER REQUEST")
        print(f"{'='*60}")
        print(f"Request: {manager_request}\n")
        
        start_time = datetime.now()
        
        # Phase 1: Task Planning
        print("[Phase 1] Task Planning...")
        tasks = await self.coordinator.analyze_request(manager_request)
        
        # Phase 2: Task Execution
        print(f"\n[Phase 2] Executing {len(tasks)} Tasks...")
        task_results = []
        
        for task in tasks:
            agent = self.agents.get(task.assigned_to)
            if agent and hasattr(agent, 'execute_task'):
                result = await agent.execute_task(task)
                task_results.append(result)
        
        # Phase 3: Report Compilation
        print(f"\n[Phase 3] Compiling Final Report...")
        final_report = await self.report_agent.compile_report(
            manager_request,
            task_results
        )
        
        # Calculate metrics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        result = {
            "request": manager_request,
            "tasks_completed": len(tasks),
            "agents_involved": list(set([t.assigned_to.value for t in tasks])),
            "processing_time_seconds": processing_time,
            "report": final_report,
            "timestamp": end_time.isoformat()
        }
        
        print(f"\n{'='*60}")
        print("✅ REQUEST COMPLETED")
        print(f"{'='*60}")
        print(f"Tasks Completed: {result['tasks_completed']}")
        print(f"Agents Involved: {', '.join(result['agents_involved'])}")
        print(f"Processing Time: {processing_time:.2f}s")
        print(f"\n📄 FINAL REPORT:")
        print("="*60)
        print(final_report)
        print("="*60)
        
        return result


async def main():
    """
    Demonstrate manager assistant multi-agent system.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    import os
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    # Initialize orchestrator
    orchestrator = ManagerAssistantOrchestrator(api_key)
    
    # Test scenarios
    scenarios = [
        "Give me my morning briefing with today's schedule, important emails, and key metrics",
        "Schedule a team meeting next week to discuss Q4 performance",
        "Analyze our sales performance and suggest improvements",
        "Find the best time to schedule a client presentation and draft an invitation email"
    ]
    
    print("\n" + "="*60)
    print("🎬 RUNNING MANAGER ASSISTANT SCENARIOS")
    print("="*60)
    
    # Execute scenarios
    for idx, scenario in enumerate(scenarios, 1):
        print(f"\n\n📋 SCENARIO {idx}")
        await orchestrator.handle_request(scenario)
        
        if idx < len(scenarios):
            await asyncio.sleep(2)  # Brief pause between scenarios
    
    print("\n" + "="*60)
    print("✨ ALL SCENARIOS COMPLETED")
    print("="*60)
    print("\n💡 Key Takeaways:")
    print("  • Multiple agents collaborate autonomously")
    print("  • Coordinator orchestrates workflow")
    print("  • Specialized agents handle specific domains")
    print("  • Agents communicate via A2A protocol")
    print("  • Results compiled into actionable reports")
    print("  • Manager gets comprehensive assistance without micromanaging")


if __name__ == "__main__":
    asyncio.run(main())
