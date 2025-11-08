# MLOps for AI Agent Systems

Comprehensive guide to deploying, monitoring, and maintaining AI agent systems in production.

## Table of Contents

1. [MLOps Fundamentals](#mlops-fundamentals)
2. [Agent Deployment Strategies](#agent-deployment-strategies)
3. [Monitoring & Observability](#monitoring--observability)
4. [Performance Optimization](#performance-optimization)
5. [Scaling Agent Systems](#scaling-agent-systems)
6. [Version Control & Experimentation](#version-control--experimentation)
7. [Cost Optimization](#cost-optimization)
8. [Security & Compliance](#security--compliance)
9. [Incident Response](#incident-response)
10. [Best Practices](#best-practices)

---

## MLOps Fundamentals

### What is MLOps for AI Agents?

MLOps (Machine Learning Operations) for AI agents extends traditional MLOps practices to handle:
- **Multi-agent orchestration** in production
- **Dynamic agent behavior** and adaptation
- **Complex interaction patterns** between agents
- **Real-time decision making** at scale
- **Continuous learning** and improvement

### Core MLOps Principles

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio


@dataclass
class AgentDeployment:
    """Agent deployment configuration"""
    agent_id: str
    version: str
    model_version: str
    environment: str  # dev, staging, prod
    replicas: int
    resources: Dict[str, Any]
    health_check_endpoint: str
    metrics_endpoint: str
    

@dataclass
class DeploymentMetrics:
    """Key metrics for deployed agents"""
    requests_per_second: float
    average_latency_ms: float
    error_rate: float
    cost_per_request: float
    uptime_percentage: float
    agent_health_score: float
```

---

## Agent Deployment Strategies

### 1. Blue-Green Deployment

Zero-downtime deployment with instant rollback capability.

```python
class BlueGreenDeployment:
    """
    Blue-Green deployment for AI agents.
    
    Maintains two identical environments:
    - Blue: Current production
    - Green: New version
    
    Switch traffic instantly between them.
    """
    
    def __init__(self):
        self.blue_agents = []  # Current production
        self.green_agents = []  # New version
        self.active_environment = "blue"
        self.router = TrafficRouter()
    
    async def deploy_new_version(self, new_agent_version: str):
        """
        Deploy new version to green environment.
        
        Steps:
        1. Deploy to green environment
        2. Run health checks
        3. Run smoke tests
        4. Switch traffic
        5. Monitor for issues
        6. Keep blue as backup
        """
        
        print(f"🚀 Deploying version {new_agent_version} to green...")
        
        # Step 1: Deploy to green
        await self._deploy_to_green(new_agent_version)
        
        # Step 2: Health checks
        print("🏥 Running health checks...")
        health_ok = await self._run_health_checks("green")
        if not health_ok:
            raise DeploymentError("Health checks failed")
        
        # Step 3: Smoke tests
        print("🧪 Running smoke tests...")
        tests_ok = await self._run_smoke_tests("green")
        if not tests_ok:
            raise DeploymentError("Smoke tests failed")
        
        # Step 4: Switch traffic
        print("🔄 Switching traffic to green...")
        await self.router.switch_to("green")
        self.active_environment = "green"
        
        # Step 5: Monitor
        print("📊 Monitoring new deployment...")
        await self._monitor_deployment(duration_minutes=15)
        
        # Step 6: Keep blue as backup for quick rollback
        print("✅ Deployment successful. Blue kept as backup.")
    
    async def rollback(self):
        """Instant rollback to previous version"""
        
        print("⚠️ Rolling back to previous version...")
        
        target = "blue" if self.active_environment == "green" else "green"
        await self.router.switch_to(target)
        self.active_environment = target
        
        print("✅ Rollback complete")
    
    async def _deploy_to_green(self, version: str):
        """Deploy new version to green environment"""
        # In production, would deploy to K8s, ECS, etc.
        self.green_agents = await self._create_agent_instances(version)
    
    async def _run_health_checks(self, environment: str) -> bool:
        """Run health checks on environment"""
        agents = self.green_agents if environment == "green" else self.blue_agents
        
        health_checks = [agent.health_check() for agent in agents]
        results = await asyncio.gather(*health_checks)
        
        return all(results)
    
    async def _run_smoke_tests(self, environment: str) -> bool:
        """Run smoke tests on environment"""
        # Basic functionality tests
        test_cases = [
            {"input": "test1", "expected_type": "response"},
            {"input": "test2", "expected_type": "response"}
        ]
        
        agents = self.green_agents if environment == "green" else self.blue_agents
        agent = agents[0]  # Test one agent
        
        for test in test_cases:
            result = await agent.process(test["input"])
            if not isinstance(result, dict):
                return False
        
        return True
    
    async def _monitor_deployment(self, duration_minutes: int):
        """Monitor deployment for issues"""
        
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < duration_minutes * 60:
            metrics = await self._get_current_metrics()
            
            # Check for critical issues
            if metrics.error_rate > 0.05:  # 5% error rate
                print("❌ High error rate detected!")
                await self.rollback()
                raise DeploymentError("High error rate - rolled back")
            
            if metrics.average_latency_ms > 1000:  # 1s latency
                print("⚠️ High latency detected!")
            
            await asyncio.sleep(30)  # Check every 30s
```

**Deployment Flow**:
```
Current State: Blue (v1.0) serving 100% traffic

1. Deploy to Green (v1.1)
   Blue (v1.0) - 100% traffic ✅
   Green (v1.1) - 0% traffic 🔧

2. Run tests on Green
   Blue (v1.0) - 100% traffic ✅
   Green (v1.1) - 0% traffic 🧪

3. Switch traffic
   Blue (v1.0) - 0% traffic (standby)
   Green (v1.1) - 100% traffic ✅

4. If issues, instant rollback
   Blue (v1.0) - 100% traffic ✅
   Green (v1.1) - 0% traffic ❌
```

---

### 2. Canary Deployment

Gradual rollout with traffic shifting.

```python
class CanaryDeployment:
    """
    Canary deployment for AI agents.
    
    Gradually shift traffic from old to new version:
    - 5% → 10% → 25% → 50% → 100%
    - Monitor metrics at each step
    - Automatic rollback if issues detected
    """
    
    def __init__(self):
        self.current_version_agents = []
        self.canary_version_agents = []
        self.router = TrafficRouter()
        self.metrics_monitor = MetricsMonitor()
    
    async def deploy_canary(self, new_version: str):
        """
        Deploy new version as canary.
        
        Traffic shift stages:
        1. 5% canary, 95% current
        2. 10% canary, 90% current
        3. 25% canary, 75% current
        4. 50% canary, 50% current
        5. 100% canary, 0% current
        """
        
        print(f"🐤 Starting canary deployment: {new_version}")
        
        # Deploy canary
        await self._deploy_canary_version(new_version)
        
        # Gradual traffic shift
        stages = [5, 10, 25, 50, 100]
        
        for canary_percent in stages:
            print(f"\n📊 Shifting to {canary_percent}% canary traffic...")
            
            # Shift traffic
            await self.router.set_traffic_split(
                canary=canary_percent,
                current=100 - canary_percent
            )
            
            # Monitor for configured duration
            monitor_duration = self._get_monitor_duration(canary_percent)
            print(f"⏱️ Monitoring for {monitor_duration} minutes...")
            
            is_healthy = await self._monitor_canary(
                duration_minutes=monitor_duration,
                canary_percent=canary_percent
            )
            
            if not is_healthy:
                print("❌ Canary showing issues - rolling back!")
                await self.rollback()
                raise DeploymentError("Canary deployment failed")
            
            print(f"✅ Stage {canary_percent}% successful")
        
        # Canary is now 100% - promote it
        print("🎉 Canary deployment complete!")
        await self._promote_canary()
    
    def _get_monitor_duration(self, canary_percent: int) -> int:
        """Get monitoring duration based on traffic percentage"""
        # Monitor longer at lower percentages
        durations = {5: 30, 10: 20, 25: 15, 50: 10, 100: 5}
        return durations.get(canary_percent, 10)
    
    async def _monitor_canary(
        self,
        duration_minutes: int,
        canary_percent: int
    ) -> bool:
        """
        Monitor canary deployment.
        
        Compare metrics between canary and current version.
        """
        
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < duration_minutes * 60:
            # Get metrics for both versions
            canary_metrics = await self._get_canary_metrics()
            current_metrics = await self._get_current_metrics()
            
            # Compare metrics
            if not self._metrics_acceptable(canary_metrics, current_metrics):
                return False
            
            await asyncio.sleep(30)  # Check every 30s
        
        return True
    
    def _metrics_acceptable(
        self,
        canary: DeploymentMetrics,
        current: DeploymentMetrics
    ) -> bool:
        """Check if canary metrics are acceptable"""
        
        # Error rate must not increase by more than 50%
        if canary.error_rate > current.error_rate * 1.5:
            print(f"⚠️ Canary error rate too high: {canary.error_rate}")
            return False
        
        # Latency must not increase by more than 25%
        if canary.average_latency_ms > current.average_latency_ms * 1.25:
            print(f"⚠️ Canary latency too high: {canary.average_latency_ms}ms")
            return False
        
        # Cost must not increase by more than 30%
        if canary.cost_per_request > current.cost_per_request * 1.3:
            print(f"⚠️ Canary cost too high: ${canary.cost_per_request}")
            return False
        
        return True
    
    async def rollback(self):
        """Rollback canary deployment"""
        await self.router.set_traffic_split(canary=0, current=100)
        await self._remove_canary_version()
```

**Canary Flow**:
```
Stage 1: 5% Canary
├─ Current v1.0: 95% traffic
└─ Canary v1.1:  5% traffic → Monitor 30min

Stage 2: 10% Canary
├─ Current v1.0: 90% traffic
└─ Canary v1.1: 10% traffic → Monitor 20min

Stage 3: 25% Canary
├─ Current v1.0: 75% traffic
└─ Canary v1.1: 25% traffic → Monitor 15min

Stage 4: 50% Canary
├─ Current v1.0: 50% traffic
└─ Canary v1.1: 50% traffic → Monitor 10min

Stage 5: 100% Canary
├─ Current v1.0:  0% traffic (removed)
└─ Canary v1.1: 100% traffic → Promoted!
```

---

### 3. A/B Testing Deployment

Test different agent versions or configurations.

```python
class ABTestingDeployment:
    """
    A/B testing for AI agents.
    
    Run multiple versions simultaneously to compare performance.
    """
    
    def __init__(self):
        self.variants = {}  # variant_id -> agents
        self.router = TrafficRouter()
        self.experiment_tracker = ExperimentTracker()
    
    async def start_experiment(
        self,
        experiment_id: str,
        variants: Dict[str, Any],
        traffic_split: Dict[str, int]
    ):
        """
        Start A/B test experiment.
        
        Example:
        variants = {
            "control": {"version": "v1.0", "config": {...}},
            "variant_a": {"version": "v1.1", "config": {...}},
            "variant_b": {"version": "v1.2", "config": {...}}
        }
        
        traffic_split = {
            "control": 50,
            "variant_a": 25,
            "variant_b": 25
        }
        """
        
        print(f"🧪 Starting A/B test experiment: {experiment_id}")
        
        # Deploy all variants
        for variant_id, config in variants.items():
            print(f"  Deploying {variant_id}...")
            await self._deploy_variant(variant_id, config)
        
        # Configure traffic split
        await self.router.set_variant_traffic(traffic_split)
        
        # Track experiment
        self.experiment_tracker.start(
            experiment_id=experiment_id,
            variants=list(variants.keys()),
            traffic_split=traffic_split
        )
        
        print("✅ Experiment started")
    
    async def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Analyze A/B test results.
        
        Compare variants on:
        - Success rate
        - Latency
        - Cost
        - User satisfaction
        - Business metrics
        """
        
        print(f"📊 Analyzing experiment: {experiment_id}")
        
        # Get metrics for each variant
        variant_metrics = {}
        
        for variant_id in self.variants.keys():
            metrics = await self._get_variant_metrics(variant_id)
            variant_metrics[variant_id] = metrics
        
        # Statistical analysis
        analysis = {
            "experiment_id": experiment_id,
            "variants": variant_metrics,
            "winner": self._determine_winner(variant_metrics),
            "confidence": self._calculate_confidence(variant_metrics),
            "recommendation": None
        }
        
        # Generate recommendation
        if analysis["confidence"] > 0.95:
            analysis["recommendation"] = f"Deploy {analysis['winner']}"
        else:
            analysis["recommendation"] = "Continue experiment - need more data"
        
        return analysis
    
    def _determine_winner(self, metrics: Dict[str, DeploymentMetrics]) -> str:
        """Determine winning variant based on metrics"""
        
        # Score each variant
        scores = {}
        
        for variant_id, m in metrics.items():
            # Higher is better for these
            success_score = (1 - m.error_rate) * 100
            uptime_score = m.uptime_percentage
            
            # Lower is better for these (invert)
            latency_score = 100 - min(m.average_latency_ms / 10, 100)
            cost_score = 100 - min(m.cost_per_request * 1000, 100)
            
            # Weighted average
            scores[variant_id] = (
                success_score * 0.4 +
                latency_score * 0.3 +
                cost_score * 0.2 +
                uptime_score * 0.1
            )
        
        return max(scores, key=scores.get)
    
    def _calculate_confidence(self, metrics: Dict[str, DeploymentMetrics]) -> float:
        """Calculate statistical confidence in results"""
        # Simplified - in production, use proper statistical tests
        # (t-test, chi-square, etc.)
        return 0.95  # Placeholder
```

---

## Monitoring & Observability

### Comprehensive Monitoring System

```python
class AgentMonitoringSystem:
    """
    Comprehensive monitoring for AI agent systems.
    
    Monitors:
    - Performance metrics
    - Business metrics
    - Agent health
    - System resources
    - Costs
    """
    
    def __init__(self):
        self.metrics_store = MetricsStore()
        self.alert_manager = AlertManager()
        self.dashboard = Dashboard()
    
    async def collect_metrics(self, agent_id: str):
        """Collect all metrics for an agent"""
        
        metrics = {
            # Performance metrics
            "latency_p50": await self._get_latency_percentile(agent_id, 50),
            "latency_p95": await self._get_latency_percentile(agent_id, 95),
            "latency_p99": await self._get_latency_percentile(agent_id, 99),
            "throughput": await self._get_throughput(agent_id),
            "error_rate": await self._get_error_rate(agent_id),
            
            # Business metrics
            "requests_per_minute": await self._get_rpm(agent_id),
            "cost_per_request": await self._get_cost_per_request(agent_id),
            "user_satisfaction": await self._get_satisfaction(agent_id),
            
            # Resource metrics
            "cpu_usage": await self._get_cpu_usage(agent_id),
            "memory_usage": await self._get_memory_usage(agent_id),
            "gpu_usage": await self._get_gpu_usage(agent_id),
            
            # Agent-specific metrics
            "agent_health_score": await self._calculate_health_score(agent_id),
            "model_confidence": await self._get_avg_confidence(agent_id),
            "cache_hit_rate": await self._get_cache_hit_rate(agent_id)
        }
        
        # Store metrics
        await self.metrics_store.record(agent_id, metrics)
        
        # Check for alerts
        await self._check_alerts(agent_id, metrics)
        
        return metrics
    
    async def _check_alerts(self, agent_id: str, metrics: Dict[str, Any]):
        """Check if any metrics exceed thresholds"""
        
        alerts = []
        
        # High latency
        if metrics["latency_p95"] > 1000:  # 1s
            alerts.append({
                "severity": "warning",
                "metric": "latency_p95",
                "value": metrics["latency_p95"],
                "threshold": 1000,
                "message": f"High latency for agent {agent_id}"
            })
        
        # High error rate
        if metrics["error_rate"] > 0.05:  # 5%
            alerts.append({
                "severity": "critical",
                "metric": "error_rate",
                "value": metrics["error_rate"],
                "threshold": 0.05,
                "message": f"High error rate for agent {agent_id}"
            })
        
        # High cost
        if metrics["cost_per_request"] > 0.10:  # $0.10
            alerts.append({
                "severity": "warning",
                "metric": "cost_per_request",
                "value": metrics["cost_per_request"],
                "threshold": 0.10,
                "message": f"High cost per request for agent {agent_id}"
            })
        
        # Send alerts
        for alert in alerts:
            await self.alert_manager.send_alert(alert)


class DistributedTracing:
    """
    Distributed tracing for multi-agent systems.
    
    Track requests across multiple agents.
    """
    
    def __init__(self):
        self.traces = {}
    
    async def trace_request(self, request_id: str):
        """
        Trace a request through multiple agents.
        
        Returns complete trace showing:
        - Which agents were involved
        - Time spent in each agent
        - Dependencies between agents
        - Any errors encountered
        """
        
        trace = {
            "request_id": request_id,
            "start_time": datetime.now().isoformat(),
            "spans": [],
            "total_duration_ms": 0
        }
        
        # Example trace
        trace["spans"] = [
            {
                "agent_id": "router",
                "start_ms": 0,
                "duration_ms": 15,
                "status": "success"
            },
            {
                "agent_id": "analyzer",
                "start_ms": 15,
                "duration_ms": 250,
                "status": "success",
                "parent": "router"
            },
            {
                "agent_id": "database",
                "start_ms": 265,
                "duration_ms": 45,
                "status": "success",
                "parent": "analyzer"
            },
            {
                "agent_id": "synthesizer",
                "start_ms": 310,
                "duration_ms": 180,
                "status": "success",
                "parent": "router"
            }
        ]
        
        trace["total_duration_ms"] = 490
        
        return trace
    
    def visualize_trace(self, trace: Dict[str, Any]):
        """Generate ASCII visualization of trace"""
        
        print(f"\n🔍 Trace: {trace['request_id']}")
        print(f"Total Duration: {trace['total_duration_ms']}ms\n")
        
        for span in trace["spans"]:
            indent = "  " * (span.get("parent") is not None)
            status = "✅" if span["status"] == "success" else "❌"
            print(f"{indent}{status} {span['agent_id']}: {span['duration_ms']}ms")
```

---

## Performance Optimization

### Caching Strategies

```python
class AgentCacheSystem:
    """
    Multi-level caching for AI agents.
    
    Levels:
    1. Memory cache (fastest, limited capacity)
    2. Redis cache (fast, larger capacity)
    3. Database cache (slower, unlimited)
    """
    
    def __init__(self):
        self.memory_cache = {}  # In-memory
        self.redis_client = None  # Redis
        self.db_client = None  # Database
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache with multi-level fallback"""
        
        # Level 1: Memory
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Level 2: Redis
        redis_value = await self.redis_client.get(key)
        if redis_value:
            # Populate memory cache
            self.memory_cache[key] = redis_value
            return redis_value
        
        # Level 3: Database
        db_value = await self.db_client.get(key)
        if db_value:
            # Populate Redis and memory
            await self.redis_client.set(key, db_value)
            self.memory_cache[key] = db_value
            return db_value
        
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set in all cache levels"""
        
        # Set in all levels
        self.memory_cache[key] = value
        await self.redis_client.set(key, value, ex=ttl_seconds)
        await self.db_client.set(key, value, ttl=ttl_seconds)


class BatchProcessor:
    """
    Batch processing for improved throughput.
    """
    
    def __init__(self, batch_size: int = 32, max_wait_ms: int = 100):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests = []
        self.pending_responses = {}
    
    async def process_request(self, request: Dict[str, Any]) -> Any:
        """
        Process request with batching.
        
        Collects requests into batches for efficient processing.
        """
        
        request_id = self._generate_request_id()
        future = asyncio.Future()
        
        self.pending_requests.append((request_id, request, future))
        self.pending_responses[request_id] = future
        
        # Trigger batch if full
        if len(self.pending_requests) >= self.batch_size:
            asyncio.create_task(self._process_batch())
        else:
            # Schedule batch after max wait time
            asyncio.create_task(self._schedule_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process accumulated batch"""
        
        if not self.pending_requests:
            return
        
        # Get batch
        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]
        
        # Process batch efficiently
        batch_requests = [req for _, req, _ in batch]
        batch_results = await self.agent.process_batch(batch_requests)
        
        # Distribute results to futures
        for (req_id, _, future), result in zip(batch, batch_results):
            future.set_result(result)
            del self.pending_responses[req_id]
```

---

## Scaling Agent Systems

### Horizontal Scaling

```python
class AgentScalingManager:
    """
    Automatic scaling for AI agents based on load.
    """
    
    def __init__(self):
        self.min_replicas = 2
        self.max_replicas = 20
        self.target_cpu_utilization = 0.70  # 70%
        self.target_latency_p95 = 500  # ms
        self.scale_up_cooldown = 300  # seconds
        self.scale_down_cooldown = 600  # seconds
        self.last_scale_time = None
    
    async def check_and_scale(self, agent_id: str):
        """
        Check metrics and scale if needed.
        
        Scale up if:
        - CPU > 70% for 5 minutes
        - Latency p95 > 500ms for 5 minutes
        - Request queue growing
        
        Scale down if:
        - CPU < 40% for 10 minutes
        - Latency p95 < 200ms for 10 minutes
        - Request queue empty
        """
        
        # Get current metrics
        metrics = await self._get_metrics(agent_id)
        current_replicas = await self._get_current_replicas(agent_id)
        
        # Check cooldown
        if self.last_scale_time:
            time_since_scale = (datetime.now() - self.last_scale_time).total_seconds()
            if time_since_scale < self.scale_up_cooldown:
                return  # Still in cooldown
        
        # Determine if scaling needed
        should_scale_up = (
            metrics["cpu_usage"] > self.target_cpu_utilization or
            metrics["latency_p95"] > self.target_latency_p95
        )
        
        should_scale_down = (
            metrics["cpu_usage"] < 0.40 and
            metrics["latency_p95"] < 200 and
            current_replicas > self.min_replicas
        )
        
        if should_scale_up and current_replicas < self.max_replicas:
            # Calculate desired replicas
            desired = min(
                current_replicas + 2,  # Add 2 at a time
                self.max_replicas
            )
            
            print(f"📈 Scaling up {agent_id}: {current_replicas} → {desired}")
            await self._scale_to(agent_id, desired)
            self.last_scale_time = datetime.now()
        
        elif should_scale_down:
            # Scale down gradually
            desired = max(
                current_replicas - 1,  # Remove 1 at a time
                self.min_replicas
            )
            
            print(f"📉 Scaling down {agent_id}: {current_replicas} → {desired}")
            await self._scale_to(agent_id, desired)
            self.last_scale_time = datetime.now()
```

---

## Cost Optimization

### Cost Monitoring and Optimization

```python
class CostOptimizer:
    """
    Monitor and optimize costs for AI agent systems.
    """
    
    def __init__(self):
        self.cost_tracker = CostTracker()
        self.budget_alerts = []
    
    async def analyze_costs(self) -> Dict[str, Any]:
        """
        Analyze costs and identify optimization opportunities.
        
        Cost breakdown:
        - Model API costs (OpenAI, Anthropic, etc.)
        - Infrastructure costs (compute, storage)
        - Data transfer costs
        - Third-party service costs
        """
        
        costs = {
            "total_monthly": 0,
            "breakdown": {},
            "optimization_opportunities": []
        }
        
        # Model API costs
        model_costs = await self._calculate_model_costs()
        costs["breakdown"]["model_apis"] = model_costs
        costs["total_monthly"] += model_costs
        
        # Infrastructure costs
        infra_costs = await self._calculate_infrastructure_costs()
        costs["breakdown"]["infrastructure"] = infra_costs
        costs["total_monthly"] += infra_costs
        
        # Identify optimizations
        if model_costs > infra_costs * 2:
            costs["optimization_opportunities"].append({
                "type": "model_api",
                "suggestion": "Consider caching frequently used responses",
                "potential_savings": model_costs * 0.30  # 30% reduction
            })
        
        # Check for unused resources
        unused = await self._find_unused_resources()
        if unused:
            savings = sum(r["cost"] for r in unused)
            costs["optimization_opportunities"].append({
                "type": "unused_resources",
                "suggestion": f"Remove {len(unused)} unused resources",
                "potential_savings": savings
            })
        
        return costs
    
    async def implement_optimizations(self):
        """Implement cost optimization strategies"""
        
        # Strategy 1: Intelligent caching
        await self._implement_caching()
        
        # Strategy 2: Request batching
        await self._enable_batching()
        
        # Strategy 3: Use cheaper models for simple tasks
        await self._implement_model_routing()
        
        # Strategy 4: Compress data transfers
        await self._enable_compression()
        
        # Strategy 5: Schedule non-urgent tasks for off-peak hours
        await self._implement_scheduling()
    
    async def _implement_model_routing(self):
        """
        Route requests to appropriate model based on complexity.
        
        - Simple queries → Smaller, cheaper models
        - Complex queries → Larger, expensive models
        """
        
        class ModelRouter:
            async def route(self, request):
                complexity = self.assess_complexity(request)
                
                if complexity < 0.3:
                    return "gpt-3.5-turbo"  # Cheaper
                elif complexity < 0.7:
                    return "gpt-4"  # Balanced
                else:
                    return "gpt-4-turbo"  # Most capable
```

---

## Security & Compliance

### Security Best Practices

```python
class AgentSecurityManager:
    """
    Security management for AI agent systems.
    """
    
    async def secure_agent_deployment(self, agent_id: str):
        """
        Implement security best practices:
        
        1. Input validation
        2. Output sanitization
        3. Rate limiting
        4. Authentication & authorization
        5. Audit logging
        6. Data encryption
        7. Model access controls
        """
        
        # 1. Input Validation
        await self._enable_input_validation(agent_id)
        
        # 2. Output Sanitization
        await self._enable_output_sanitization(agent_id)
        
        # 3. Rate Limiting
        await self._configure_rate_limits(agent_id, {
            "requests_per_minute": 100,
            "requests_per_hour": 5000
        })
        
        # 4. Authentication
        await self._enable_authentication(agent_id)
        
        # 5. Audit Logging
        await self._enable_audit_logs(agent_id)
        
        # 6. Encryption
        await self._enable_encryption(agent_id)
    
    async def audit_request(
        self,
        user_id: str,
        agent_id: str,
        request: Dict[str, Any],
        response: Dict[str, Any]
    ):
        """Log request for audit trail"""
        
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "agent_id": agent_id,
            "request_summary": self._summarize_request(request),
            "response_summary": self._summarize_response(response),
            "ip_address": request.get("ip_address"),
            "user_agent": request.get("user_agent")
        }
        
        await self.audit_log.record(audit_entry)
```

---

## Best Practices

### 1. Deployment Checklist

```python
# Pre-deployment checklist
DEPLOYMENT_CHECKLIST = [
    "✓ All tests passing",
    "✓ Load testing completed",
    "✓ Security scan passed",
    "✓ Documentation updated",
    "✓ Monitoring configured",
    "✓ Alerts configured",
    "✓ Rollback plan ready",
    "✓ Stakeholders notified",
    "✓ Backup systems verified",
    "✓ Cost estimates reviewed"
]
```

### 2. Production Readiness

```python
class ProductionReadiness:
    """Check if agent system is production-ready"""
    
    async def check_readiness(self, agent_id: str) -> Dict[str, Any]:
        """Comprehensive production readiness check"""
        
        checks = {
            "health_check": await self._check_health_endpoint(agent_id),
            "performance": await self._check_performance(agent_id),
            "security": await self._check_security(agent_id),
            "monitoring": await self._check_monitoring(agent_id),
            "documentation": await self._check_documentation(agent_id),
            "testing": await self._check_test_coverage(agent_id),
            "scalability": await self._check_scalability(agent_id)
        }
        
        all_passed = all(checks.values())
        
        return {
            "ready_for_production": all_passed,
            "checks": checks,
            "blockers": [k for k, v in checks.items() if not v]
        }
```

---

## Related Documentation

- [Orchestration Patterns](./ORCHESTRATION.md) - Multi-agent coordination
- [Workflows Guide](./WORKFLOWS.md) - Workflow patterns
- [Agent-to-Agent Communication](./A2A.md) - Inter-agent communication

---

**Congratulations!** You now have a comprehensive MLOps framework for deploying and managing AI agent systems in production.