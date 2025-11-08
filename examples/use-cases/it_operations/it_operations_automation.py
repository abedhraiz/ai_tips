"""
IT Operations Automation - Multi-Agent A2A System
=================================================

Demonstrates autonomous IT operations where agents collaborate to:
- Monitor systems continuously
- Detect and triage incidents
- Diagnose root causes
- Execute automated remediation
- Report on incidents and resolutions

Agents communicate via A2A protocol without human intervention.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random


class Severity(Enum):
    """Incident severity levels"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    EMERGENCY = 5


class IncidentType(Enum):
    """Types of IT incidents"""
    PERFORMANCE = "performance"
    DATABASE = "database"
    NETWORK = "network"
    DISK = "disk"
    SECURITY = "security"
    APPLICATION = "application"


class AgentRole(Enum):
    """IT operations agent roles"""
    MONITORING = "monitoring"
    TRIAGE = "triage"
    DIAGNOSTIC = "diagnostic"
    DATABASE = "database"
    NETWORK = "network"
    REMEDIATION = "remediation"
    REPORTING = "reporting"


@dataclass
class Incident:
    """IT incident structure"""
    incident_id: str
    type: IncidentType
    severity: Severity
    description: str
    detected_at: str
    metrics: Dict[str, Any]
    status: str = "open"  # open, investigating, resolving, resolved, escalated
    assigned_to: Optional[AgentRole] = None
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    resolved_at: Optional[str] = None


@dataclass
class SystemMetrics:
    """System health metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    database_connections: int
    error_rate: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MonitoringAgent:
    """
    Continuously monitors system health and detects anomalies.
    """
    
    def __init__(self, name: str = "Monitoring Agent"):
        self.name = name
        self.baseline_metrics = {
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "disk_usage": 65.0,
            "network_latency": 25.0,
            "database_connections": 50,
            "error_rate": 0.5
        }
        print(f"  ✓ {self.name} initialized - continuous monitoring active")
    
    def collect_metrics(self) -> SystemMetrics:
        """Simulate collecting system metrics"""
        # Simulate normal operation with occasional spikes
        return SystemMetrics(
            cpu_usage=random.uniform(40, 95),
            memory_usage=random.uniform(55, 85),
            disk_usage=random.uniform(60, 92),
            network_latency=random.uniform(20, 150),
            database_connections=random.randint(40, 150),
            error_rate=random.uniform(0.1, 5.0)
        )
    
    def detect_anomaly(self, metrics: SystemMetrics) -> Optional[Incident]:
        """Detect if metrics indicate an issue"""
        
        incidents = []
        
        # CPU check
        if metrics.cpu_usage > 85:
            incidents.append(Incident(
                incident_id=f"INC_{datetime.now().timestamp()}",
                type=IncidentType.PERFORMANCE,
                severity=Severity.ERROR if metrics.cpu_usage > 90 else Severity.WARNING,
                description=f"High CPU usage detected: {metrics.cpu_usage:.1f}%",
                detected_at=datetime.now().isoformat(),
                metrics={"cpu_usage": metrics.cpu_usage}
            ))
        
        # Disk check
        if metrics.disk_usage > 85:
            incidents.append(Incident(
                incident_id=f"INC_{datetime.now().timestamp()}",
                type=IncidentType.DISK,
                severity=Severity.CRITICAL if metrics.disk_usage > 90 else Severity.WARNING,
                description=f"Disk space critical: {metrics.disk_usage:.1f}% used",
                detected_at=datetime.now().isoformat(),
                metrics={"disk_usage": metrics.disk_usage}
            ))
        
        # Network latency check
        if metrics.network_latency > 100:
            incidents.append(Incident(
                incident_id=f"INC_{datetime.now().timestamp()}",
                type=IncidentType.NETWORK,
                severity=Severity.ERROR,
                description=f"High network latency: {metrics.network_latency:.1f}ms",
                detected_at=datetime.now().isoformat(),
                metrics={"network_latency": metrics.network_latency}
            ))
        
        # Database connections check
        if metrics.database_connections > 120:
            incidents.append(Incident(
                incident_id=f"INC_{datetime.now().timestamp()}",
                type=IncidentType.DATABASE,
                severity=Severity.ERROR,
                description=f"Database connection pool exhausted: {metrics.database_connections} connections",
                detected_at=datetime.now().isoformat(),
                metrics={"database_connections": metrics.database_connections}
            ))
        
        # Error rate check
        if metrics.error_rate > 2.0:
            incidents.append(Incident(
                incident_id=f"INC_{datetime.now().timestamp()}",
                type=IncidentType.APPLICATION,
                severity=Severity.ERROR,
                description=f"High error rate: {metrics.error_rate:.2f}%",
                detected_at=datetime.now().isoformat(),
                metrics={"error_rate": metrics.error_rate}
            ))
        
        return incidents[0] if incidents else None


class TriageAgent:
    """
    Classifies incidents and routes to appropriate specialist agents.
    """
    
    def __init__(self, name: str = "Triage Agent"):
        self.name = name
        print(f"  ✓ {self.name} initialized")
    
    def classify_and_route(self, incident: Incident) -> AgentRole:
        """Classify incident and determine which agent should handle it"""
        
        print(f"\n🔍 {self.name} classifying incident...")
        print(f"  Type: {incident.type.value}")
        print(f"  Severity: {incident.severity.name}")
        
        # Route based on incident type
        routing = {
            IncidentType.PERFORMANCE: AgentRole.DIAGNOSTIC,
            IncidentType.DATABASE: AgentRole.DATABASE,
            IncidentType.NETWORK: AgentRole.NETWORK,
            IncidentType.DISK: AgentRole.DIAGNOSTIC,
            IncidentType.SECURITY: AgentRole.DIAGNOSTIC,
            IncidentType.APPLICATION: AgentRole.DIAGNOSTIC
        }
        
        assigned_agent = routing.get(incident.type, AgentRole.DIAGNOSTIC)
        incident.assigned_to = assigned_agent
        incident.status = "investigating"
        
        print(f"  ✓ Routed to: {assigned_agent.value}")
        
        return assigned_agent


class DiagnosticAgent:
    """
    Investigates incidents and determines root causes.
    """
    
    def __init__(self, name: str = "Diagnostic Agent"):
        self.name = name
        print(f"  ✓ {self.name} initialized")
    
    async def investigate(self, incident: Incident) -> str:
        """Investigate incident and determine root cause"""
        
        print(f"\n🔬 {self.name} investigating: {incident.description}")
        
        # Simulate investigation
        await asyncio.sleep(0.5)
        
        # Determine root cause based on incident type
        root_causes = {
            IncidentType.PERFORMANCE: "Resource-intensive process consuming excessive CPU",
            IncidentType.DISK: "Log files not being rotated, accumulating disk space",
            IncidentType.NETWORK: "Network congestion due to traffic spike",
            IncidentType.DATABASE: "Connection pool size insufficient for current load",
            IncidentType.APPLICATION: "Memory leak in application service"
        }
        
        root_cause = root_causes.get(
            incident.type,
            "Unknown issue requiring further investigation"
        )
        
        incident.root_cause = root_cause
        print(f"  ✓ Root cause identified: {root_cause}")
        
        return root_cause


class DatabaseAgent:
    """
    Handles database-specific incidents.
    """
    
    def __init__(self, name: str = "Database Agent"):
        self.name = name
        print(f"  ✓ {self.name} initialized")
    
    async def diagnose_and_fix(self, incident: Incident) -> str:
        """Diagnose database issue and apply fix"""
        
        print(f"\n🗄️ {self.name} handling database incident...")
        
        await asyncio.sleep(0.5)
        
        # Simulate database diagnostics
        if "connection" in incident.description.lower():
            print(f"  → Checking connection pool configuration...")
            print(f"  → Current connections: {incident.metrics.get('database_connections')}")
            print(f"  → Increasing connection pool size from 100 to 200")
            
            resolution = "Increased database connection pool size to handle load"
        else:
            resolution = "Optimized slow queries and rebuilt indexes"
        
        print(f"  ✓ {resolution}")
        
        return resolution


class NetworkAgent:
    """
    Handles network-related incidents.
    """
    
    def __init__(self, name: str = "Network Agent"):
        self.name = name
        print(f"  ✓ {self.name} initialized")
    
    async def diagnose_and_fix(self, incident: Incident) -> str:
        """Diagnose network issue and apply fix"""
        
        print(f"\n🌐 {self.name} handling network incident...")
        
        await asyncio.sleep(0.5)
        
        # Simulate network diagnostics
        if "latency" in incident.description.lower():
            print(f"  → Checking network routes...")
            print(f"  → Current latency: {incident.metrics.get('network_latency')}ms")
            print(f"  → Reconfiguring load balancer...")
            
            resolution = "Reconfigured load balancer to optimize traffic distribution"
        else:
            resolution = "Cleared network congestion and updated routing tables"
        
        print(f"  ✓ {resolution}")
        
        return resolution


class RemediationAgent:
    """
    Executes automated remediation actions.
    """
    
    def __init__(self, name: str = "Remediation Agent"):
        self.name = name
        print(f"  ✓ {self.name} initialized")
    
    async def execute_remediation(self, incident: Incident, resolution_plan: str) -> bool:
        """Execute remediation actions"""
        
        print(f"\n🔧 {self.name} executing remediation...")
        print(f"  Plan: {resolution_plan}")
        
        incident.status = "resolving"
        
        # Simulate remediation execution
        await asyncio.sleep(1.0)
        
        # Remediation steps based on incident type
        steps = {
            IncidentType.PERFORMANCE: [
                "Identifying high-CPU processes",
                "Restarting resource-intensive service",
                "Verifying CPU usage normalized"
            ],
            IncidentType.DISK: [
                "Archiving old log files",
                "Compressing archived logs",
                "Clearing temporary files",
                "Verifying disk space freed"
            ],
            IncidentType.NETWORK: [
                "Analyzing traffic patterns",
                "Updating load balancer config",
                "Restarting network services",
                "Verifying latency improved"
            ],
            IncidentType.DATABASE: [
                "Updating connection pool settings",
                "Restarting database service",
                "Verifying connections stable"
            ]
        }
        
        for step in steps.get(incident.type, ["Executing generic fix"]):
            print(f"    → {step}")
            await asyncio.sleep(0.3)
        
        incident.status = "resolved"
        incident.resolved_at = datetime.now().isoformat()
        incident.resolution = resolution_plan
        
        print(f"  ✓ Remediation completed successfully")
        
        return True


class ReportingAgent:
    """
    Documents incidents and generates reports.
    """
    
    def __init__(self, name: str = "Reporting Agent"):
        self.name = name
        self.incidents_log: List[Incident] = []
        print(f"  ✓ {self.name} initialized")
    
    def generate_incident_report(self, incident: Incident) -> str:
        """Generate detailed incident report"""
        
        print(f"\n📋 {self.name} generating incident report...")
        
        self.incidents_log.append(incident)
        
        duration = "N/A"
        if incident.resolved_at:
            start = datetime.fromisoformat(incident.detected_at)
            end = datetime.fromisoformat(incident.resolved_at)
            duration = f"{(end - start).total_seconds():.1f}s"
        
        report = f"""
╔════════════════════════════════════════════════════════════╗
║                    INCIDENT REPORT                         ║
╚════════════════════════════════════════════════════════════╝

Incident ID:      {incident.incident_id}
Type:             {incident.type.value.upper()}
Severity:         {incident.severity.name}
Status:           {incident.status.upper()}

Description:      {incident.description}
Detected At:      {incident.detected_at}
Resolved At:      {incident.resolved_at or 'In Progress'}
Resolution Time:  {duration}

Root Cause:       {incident.root_cause or 'Under investigation'}
Resolution:       {incident.resolution or 'Pending'}

Assigned To:      {incident.assigned_to.value if incident.assigned_to else 'Unassigned'}

Metrics:          {json.dumps(incident.metrics, indent=18)}
"""
        
        print(report)
        
        return report


class ITOperationsOrchestrator:
    """
    Orchestrates multi-agent IT operations automation.
    
    Coordinates agents to handle incidents autonomously.
    """
    
    def __init__(self):
        # Initialize all agents
        self.monitoring = MonitoringAgent()
        self.triage = TriageAgent()
        self.diagnostic = DiagnosticAgent()
        self.database = DatabaseAgent()
        self.network = NetworkAgent()
        self.remediation = RemediationAgent()
        self.reporting = ReportingAgent()
        
        self.specialist_agents = {
            AgentRole.DATABASE: self.database,
            AgentRole.NETWORK: self.network
        }
        
        print("\n" + "="*60)
        print("🏢 IT OPERATIONS AUTOMATION SYSTEM INITIALIZED")
        print("="*60)
    
    async def handle_incident(self, incident: Incident) -> Dict[str, Any]:
        """
        Process incident through multi-agent workflow.
        
        Workflow:
        1. Triage classifies and routes
        2. Diagnostic agent investigates
        3. Specialist agent (if needed) provides domain expertise
        4. Remediation agent executes fix
        5. Reporting agent documents
        """
        
        print(f"\n\n{'='*60}")
        print(f"🚨 INCIDENT DETECTED")
        print(f"{'='*60}")
        print(f"Description: {incident.description}")
        print(f"Severity: {incident.severity.name}")
        
        # Phase 1: Triage
        print(f"\n[Phase 1] Triage and Classification")
        assigned_agent = self.triage.classify_and_route(incident)
        
        # Phase 2: Investigation
        print(f"\n[Phase 2] Root Cause Analysis")
        root_cause = await self.diagnostic.investigate(incident)
        
        # Phase 3: Specialist Consultation (if needed)
        resolution_plan = None
        if assigned_agent in [AgentRole.DATABASE, AgentRole.NETWORK]:
            print(f"\n[Phase 3] Specialist Agent Consultation")
            specialist = self.specialist_agents[assigned_agent]
            resolution_plan = await specialist.diagnose_and_fix(incident)
        else:
            resolution_plan = f"Standard remediation for {incident.type.value}"
        
        # Phase 4: Remediation
        print(f"\n[Phase 4] Automated Remediation")
        success = await self.remediation.execute_remediation(incident, resolution_plan)
        
        # Phase 5: Reporting
        print(f"\n[Phase 5] Incident Documentation")
        report = self.reporting.generate_incident_report(incident)
        
        return {
            "incident_id": incident.incident_id,
            "status": incident.status,
            "root_cause": root_cause,
            "resolution": resolution_plan,
            "resolved": success,
            "report": report
        }
    
    async def monitoring_loop(self, duration_seconds: int = 10, interval: float = 2.0):
        """
        Simulate continuous monitoring with incident detection.
        """
        
        print(f"\n\n{'='*60}")
        print(f"📡 STARTING CONTINUOUS MONITORING")
        print(f"{'='*60}")
        print(f"Duration: {duration_seconds}s | Interval: {interval}s")
        
        start_time = datetime.now()
        incidents_handled = 0
        
        while (datetime.now() - start_time).total_seconds() < duration_seconds:
            # Collect metrics
            metrics = self.monitoring.collect_metrics()
            
            print(f"\n⏱️ [{datetime.now().strftime('%H:%M:%S')}] System Health Check")
            print(f"  CPU: {metrics.cpu_usage:.1f}% | Memory: {metrics.memory_usage:.1f}% | Disk: {metrics.disk_usage:.1f}%")
            print(f"  Network: {metrics.network_latency:.1f}ms | DB Conn: {metrics.database_connections} | Errors: {metrics.error_rate:.2f}%")
            
            # Check for anomalies
            incident = self.monitoring.detect_anomaly(metrics)
            
            if incident:
                incidents_handled += 1
                await self.handle_incident(incident)
            else:
                print(f"  ✓ All systems normal")
            
            # Wait before next check
            await asyncio.sleep(interval)
        
        print(f"\n\n{'='*60}")
        print(f"📊 MONITORING SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Duration: {duration_seconds}s")
        print(f"Incidents Detected: {incidents_handled}")
        print(f"Incidents Resolved: {incidents_handled}")
        print(f"Automation Rate: 100%")
        print(f"Average Resolution Time: <2s")


async def main():
    """
    Demonstrate IT operations automation with A2A agents.
    """
    
    orchestrator = ITOperationsOrchestrator()
    
    # Run continuous monitoring simulation
    await orchestrator.monitoring_loop(duration_seconds=15, interval=3.0)
    
    print("\n" + "="*60)
    print("✨ IT OPERATIONS DEMONSTRATION COMPLETE")
    print("="*60)
    print("\n💡 Key Takeaways:")
    print("  • Agents monitor systems continuously")
    print("  • Incidents detected and classified automatically")
    print("  • Specialized agents handle domain-specific issues")
    print("  • Remediation executed without human intervention")
    print("  • Complete audit trail maintained")
    print("  • Scalable to monitor thousands of systems")


if __name__ == "__main__":
    asyncio.run(main())
