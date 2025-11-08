# AI Agent Workflows Guide

Comprehensive guide to designing and implementing workflows for AI agent systems.

## Table of Contents

1. [Workflow Fundamentals](#workflow-fundamentals)
2. [Common Workflow Patterns](#common-workflow-patterns)
3. [Business Intelligence Workflows](#business-intelligence-workflows)
4. [Data Pipeline Workflows](#data-pipeline-workflows)
5. [Real-Time Processing Workflows](#real-time-processing-workflows)
6. [Decision Workflows](#decision-workflows)
7. [Monitoring & Error Handling](#monitoring--error-handling)
8. [Best Practices](#best-practices)

---

## Workflow Fundamentals

### What is an AI Agent Workflow?

A workflow defines how multiple AI agents collaborate to complete complex tasks through:
- **Sequential steps** - Tasks completed in order
- **Parallel execution** - Multiple tasks at once
- **Conditional logic** - Decisions based on outcomes
- **Error handling** - Recovery and fallback strategies

### Core Workflow Components

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
from enum import Enum
import asyncio


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    step_id: str
    agent: Any  # Agent responsible for this step
    action: Callable  # Function to execute
    inputs: Dict[str, Any]
    dependencies: List[str] = None  # IDs of prerequisite steps
    retry_count: int = 3
    timeout_seconds: int = 300
    
    
@dataclass
class WorkflowDefinition:
    """Complete workflow specification"""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    error_handler: Optional[Callable] = None
    success_callback: Optional[Callable] = None
```

### Basic Workflow Engine

```python
class WorkflowEngine:
    """
    Simple workflow execution engine.
    """
    
    def __init__(self):
        self.workflows = {}
        self.execution_history = []
    
    async def execute_workflow(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """
        Execute a workflow with dependency management.
        """
        
        print(f"\n🚀 Executing workflow: {workflow.name}")
        
        status = WorkflowStatus.RUNNING
        results = {}
        completed_steps = set()
        
        try:
            # Build dependency graph
            steps_by_id = {step.step_id: step for step in workflow.steps}
            
            while len(completed_steps) < len(workflow.steps):
                # Find steps ready to execute
                ready_steps = [
                    step for step in workflow.steps
                    if step.step_id not in completed_steps
                    and all(dep in completed_steps for dep in (step.dependencies or []))
                ]
                
                if not ready_steps:
                    raise RuntimeError("Workflow deadlock detected")
                
                # Execute ready steps in parallel
                tasks = [
                    self._execute_step(step, results)
                    for step in ready_steps
                ]
                
                step_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for step, result in zip(ready_steps, step_results):
                    if isinstance(result, Exception):
                        raise result
                    
                    results[step.step_id] = result
                    completed_steps.add(step.step_id)
                    print(f"  ✓ Completed step: {step.step_id}")
            
            status = WorkflowStatus.COMPLETED
            
            # Call success callback
            if workflow.success_callback:
                await workflow.success_callback(results)
            
            print(f"✅ Workflow '{workflow.name}' completed successfully")
            
        except Exception as e:
            status = WorkflowStatus.FAILED
            print(f"❌ Workflow '{workflow.name}' failed: {e}")
            
            # Call error handler
            if workflow.error_handler:
                await workflow.error_handler(e, results)
        
        return {
            "workflow_id": workflow.workflow_id,
            "status": status.value,
            "results": results,
            "completed_steps": list(completed_steps)
        }
    
    async def _execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a single workflow step with retry logic"""
        
        for attempt in range(step.retry_count):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    step.action(**step.inputs, context=context),
                    timeout=step.timeout_seconds
                )
                return result
            
            except asyncio.TimeoutError:
                if attempt == step.retry_count - 1:
                    raise
                print(f"  ⚠️ Step {step.step_id} timed out, retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            except Exception as e:
                if attempt == step.retry_count - 1:
                    raise
                print(f"  ⚠️ Step {step.step_id} failed: {e}, retrying...")
                await asyncio.sleep(2 ** attempt)
```

---

## Common Workflow Patterns

### 1. Sequential Processing

**Use Case**: Document processing pipeline, multi-step analysis

```python
class SequentialWorkflow:
    """
    Execute steps one after another.
    
    Example: Document Analysis
    1. Extract text from document
    2. Analyze sentiment
    3. Generate summary
    4. Create report
    """
    
    async def document_analysis_workflow(self, document_path: str):
        """Complete document analysis workflow"""
        
        # Step 1: Text Extraction
        print("Step 1: Extracting text...")
        text = await self.text_extractor.extract(document_path)
        
        # Step 2: Sentiment Analysis
        print("Step 2: Analyzing sentiment...")
        sentiment = await self.sentiment_analyzer.analyze(text)
        
        # Step 3: Generate Summary
        print("Step 3: Generating summary...")
        summary = await self.summarizer.summarize(text)
        
        # Step 4: Create Report
        print("Step 4: Creating report...")
        report = await self.report_generator.create({
            "text": text,
            "sentiment": sentiment,
            "summary": summary
        })
        
        return report


# Workflow Definition
workflow = WorkflowDefinition(
    workflow_id="doc_analysis_001",
    name="Document Analysis Pipeline",
    description="Extract, analyze, and summarize documents",
    steps=[
        WorkflowStep(
            step_id="extract",
            agent=text_extractor,
            action=text_extractor.extract,
            inputs={"document_path": doc_path}
        ),
        WorkflowStep(
            step_id="sentiment",
            agent=sentiment_analyzer,
            action=sentiment_analyzer.analyze,
            inputs={"text": "{{extract.result}}"},
            dependencies=["extract"]
        ),
        WorkflowStep(
            step_id="summarize",
            agent=summarizer,
            action=summarizer.summarize,
            inputs={"text": "{{extract.result}}"},
            dependencies=["extract"]
        ),
        WorkflowStep(
            step_id="report",
            agent=report_generator,
            action=report_generator.create,
            inputs={
                "sentiment": "{{sentiment.result}}",
                "summary": "{{summarize.result}}"
            },
            dependencies=["sentiment", "summarize"]
        )
    ]
)
```

**Flow Diagram**:
```
Document → [Extract] → [Sentiment] → [Report]
                     → [Summarize] ↗
```

---

### 2. Parallel Execution

**Use Case**: Market research, multi-source data gathering

```python
class ParallelWorkflow:
    """
    Execute multiple independent tasks simultaneously.
    
    Example: Market Research
    - Competitor analysis
    - Customer sentiment analysis
    - Pricing research
    All happen at the same time
    """
    
    async def market_research_workflow(self, query: str):
        """Parallel market research workflow"""
        
        print("🚀 Starting parallel market research...")
        
        # Execute all research tasks in parallel
        results = await asyncio.gather(
            self.competitor_agent.analyze(query),
            self.sentiment_agent.analyze(query),
            self.pricing_agent.analyze(query),
            self.trend_agent.analyze(query)
        )
        
        competitor_data, sentiment_data, pricing_data, trend_data = results
        
        # Synthesize results
        print("📊 Synthesizing research findings...")
        synthesis = await self.synthesis_agent.combine({
            "competitor": competitor_data,
            "sentiment": sentiment_data,
            "pricing": pricing_data,
            "trends": trend_data
        })
        
        return synthesis


# Workflow Definition
workflow = WorkflowDefinition(
    workflow_id="market_research_001",
    name="Parallel Market Research",
    description="Gather market intelligence from multiple sources",
    steps=[
        # All these run in parallel (no dependencies)
        WorkflowStep(
            step_id="competitor",
            agent=competitor_agent,
            action=competitor_agent.analyze,
            inputs={"query": query}
        ),
        WorkflowStep(
            step_id="sentiment",
            agent=sentiment_agent,
            action=sentiment_agent.analyze,
            inputs={"query": query}
        ),
        WorkflowStep(
            step_id="pricing",
            agent=pricing_agent,
            action=pricing_agent.analyze,
            inputs={"query": query}
        ),
        WorkflowStep(
            step_id="trends",
            agent=trend_agent,
            action=trend_agent.analyze,
            inputs={"query": query}
        ),
        # Synthesis depends on all parallel steps
        WorkflowStep(
            step_id="synthesize",
            agent=synthesis_agent,
            action=synthesis_agent.combine,
            inputs={
                "competitor": "{{competitor.result}}",
                "sentiment": "{{sentiment.result}}",
                "pricing": "{{pricing.result}}",
                "trends": "{{trends.result}}"
            },
            dependencies=["competitor", "sentiment", "pricing", "trends"]
        )
    ]
)
```

**Flow Diagram**:
```
                  ┌─→ [Competitor] ─┐
                  │                  │
Query → Dispatch ─┼─→ [Sentiment]  ─┼─→ [Synthesize] → Report
                  │                  │
                  ├─→ [Pricing]    ─┤
                  │                  │
                  └─→ [Trends]     ─┘
```

---

### 3. Conditional Workflow

**Use Case**: Adaptive processing based on data characteristics

```python
class ConditionalWorkflow:
    """
    Branch based on conditions and intermediate results.
    
    Example: Content Moderation
    - If content is safe → Approve
    - If content is unsafe → Reject
    - If content is unclear → Human review
    """
    
    async def content_moderation_workflow(self, content: str):
        """Content moderation with conditional routing"""
        
        # Step 1: Initial analysis
        print("Step 1: Analyzing content...")
        analysis = await self.analyzer.analyze(content)
        
        # Conditional routing based on analysis
        if analysis.confidence > 0.95:
            if analysis.is_safe:
                print("✅ Content approved automatically")
                return await self.auto_approve(content, analysis)
            else:
                print("❌ Content rejected automatically")
                return await self.auto_reject(content, analysis)
        else:
            print("👤 Flagging for human review")
            return await self.human_review_queue.add(content, analysis)


# Workflow Definition with Conditions
workflow = WorkflowDefinition(
    workflow_id="moderation_001",
    name="Content Moderation Pipeline",
    description="Moderate content with conditional routing",
    steps=[
        WorkflowStep(
            step_id="analyze",
            agent=analyzer,
            action=analyzer.analyze,
            inputs={"content": content}
        ),
        # Conditional steps
        WorkflowStep(
            step_id="approve",
            agent=approver,
            action=approver.approve,
            inputs={"content": content, "analysis": "{{analyze.result}}"},
            dependencies=["analyze"],
            condition=lambda ctx: ctx["analyze"]["is_safe"] and ctx["analyze"]["confidence"] > 0.95
        ),
        WorkflowStep(
            step_id="reject",
            agent=rejector,
            action=rejector.reject,
            inputs={"content": content, "analysis": "{{analyze.result}}"},
            dependencies=["analyze"],
            condition=lambda ctx: not ctx["analyze"]["is_safe"] and ctx["analyze"]["confidence"] > 0.95
        ),
        WorkflowStep(
            step_id="human_review",
            agent=human_queue,
            action=human_queue.add,
            inputs={"content": content, "analysis": "{{analyze.result}}"},
            dependencies=["analyze"],
            condition=lambda ctx: ctx["analyze"]["confidence"] <= 0.95
        )
    ]
)
```

**Flow Diagram**:
```
                    ┌─→ [Approve] (if safe & confident)
                    │
Content → [Analyze] ┼─→ [Reject] (if unsafe & confident)
                    │
                    └─→ [Human Review] (if uncertain)
```

---

## Business Intelligence Workflows

### ETL Workflow for BI

**Extract, Transform, Load** pattern for business data:

```python
class ETLWorkflow:
    """
    ETL workflow for business intelligence.
    """
    
    async def bi_etl_workflow(self, sources: List[str]):
        """
        Complete ETL workflow for BI dashboard.
        
        Steps:
        1. Extract data from multiple sources
        2. Clean and validate data
        3. Transform and enrich data
        4. Load into data warehouse
        5. Generate reports
        """
        
        # Extract Phase (parallel)
        print("📥 Phase 1: Extracting data...")
        extract_tasks = [
            self.extract_agent.extract(source)
            for source in sources
        ]
        raw_data = await asyncio.gather(*extract_tasks)
        
        # Transform Phase
        print("🔄 Phase 2: Transforming data...")
        cleaned_data = await self.cleaner.clean(raw_data)
        validated_data = await self.validator.validate(cleaned_data)
        enriched_data = await self.enricher.enrich(validated_data)
        
        # Load Phase
        print("💾 Phase 3: Loading data...")
        await self.loader.load_to_warehouse(enriched_data)
        
        # Report Phase (parallel)
        print("📊 Phase 4: Generating reports...")
        report_tasks = [
            self.sales_reporter.generate(enriched_data),
            self.marketing_reporter.generate(enriched_data),
            self.finance_reporter.generate(enriched_data)
        ]
        reports = await asyncio.gather(*report_tasks)
        
        return {
            "status": "success",
            "records_processed": len(enriched_data),
            "reports_generated": len(reports)
        }


# Workflow Definition
etl_workflow = WorkflowDefinition(
    workflow_id="bi_etl_001",
    name="Business Intelligence ETL",
    description="Extract, transform, and load business data",
    steps=[
        # Extract phase (parallel)
        WorkflowStep(
            step_id="extract_sales",
            agent=extract_agent,
            action=extract_agent.extract,
            inputs={"source": "salesforce"}
        ),
        WorkflowStep(
            step_id="extract_marketing",
            agent=extract_agent,
            action=extract_agent.extract,
            inputs={"source": "hubspot"}
        ),
        WorkflowStep(
            step_id="extract_finance",
            agent=extract_agent,
            action=extract_agent.extract,
            inputs={"source": "quickbooks"}
        ),
        
        # Transform phase (sequential)
        WorkflowStep(
            step_id="clean",
            agent=cleaner,
            action=cleaner.clean,
            inputs={"data": "{{extract_*.result}}"},
            dependencies=["extract_sales", "extract_marketing", "extract_finance"]
        ),
        WorkflowStep(
            step_id="validate",
            agent=validator,
            action=validator.validate,
            inputs={"data": "{{clean.result}}"},
            dependencies=["clean"]
        ),
        WorkflowStep(
            step_id="enrich",
            agent=enricher,
            action=enricher.enrich,
            inputs={"data": "{{validate.result}}"},
            dependencies=["validate"]
        ),
        
        # Load phase
        WorkflowStep(
            step_id="load",
            agent=loader,
            action=loader.load_to_warehouse,
            inputs={"data": "{{enrich.result}}"},
            dependencies=["enrich"]
        ),
        
        # Report phase (parallel)
        WorkflowStep(
            step_id="sales_report",
            agent=sales_reporter,
            action=sales_reporter.generate,
            inputs={"data": "{{enrich.result}}"},
            dependencies=["load"]
        ),
        WorkflowStep(
            step_id="marketing_report",
            agent=marketing_reporter,
            action=marketing_reporter.generate,
            inputs={"data": "{{enrich.result}}"},
            dependencies=["load"]
        ),
        WorkflowStep(
            step_id="finance_report",
            agent=finance_reporter,
            action=finance_reporter.generate,
            inputs={"data": "{{enrich.result}}"},
            dependencies=["load"]
        )
    ]
)
```

**Flow Diagram**:
```
┌─ [Extract Sales] ─┐
│                    │
├─ [Extract Mktg]  ─┼─→ [Clean] → [Validate] → [Enrich] → [Load] ─┬─→ [Sales Report]
│                    │                                               │
└─ [Extract Finance] ┘                                               ├─→ [Mktg Report]
                                                                     │
                                                                     └─→ [Finance Report]
```

---

### Real-Time Analytics Workflow

**Stream processing** for real-time business intelligence:

```python
class RealTimeAnalyticsWorkflow:
    """
    Real-time analytics with streaming data.
    """
    
    def __init__(self):
        self.event_buffer = []
        self.analytics_window = 60  # seconds
    
    async def process_event_stream(self):
        """
        Process events in real-time with windowing.
        
        Pattern:
        - Collect events in time windows
        - Analyze each window
        - Detect anomalies
        - Generate alerts
        """
        
        while True:
            # Wait for window to fill
            await asyncio.sleep(self.analytics_window)
            
            if not self.event_buffer:
                continue
            
            # Get events from current window
            events = self.event_buffer.copy()
            self.event_buffer.clear()
            
            print(f"📊 Processing {len(events)} events...")
            
            # Parallel analysis
            analysis_results = await asyncio.gather(
                self.pattern_detector.detect(events),
                self.anomaly_detector.detect(events),
                self.aggregator.aggregate(events)
            )
            
            patterns, anomalies, aggregates = analysis_results
            
            # Check for alerts
            if anomalies:
                print(f"🚨 {len(anomalies)} anomalies detected")
                await self.alert_handler.send_alerts(anomalies)
            
            # Update dashboards
            await self.dashboard.update(aggregates)
    
    async def ingest_event(self, event: Dict[str, Any]):
        """Add event to processing buffer"""
        self.event_buffer.append(event)
```

---

## Data Pipeline Workflows

### Multi-Stage Data Pipeline

```python
class DataPipelineWorkflow:
    """
    Complex data pipeline with multiple stages.
    """
    
    async def execute_pipeline(self, input_data: Any):
        """
        Execute multi-stage data pipeline.
        
        Stages:
        1. Ingestion - Receive and validate
        2. Preprocessing - Clean and normalize
        3. Feature extraction - Generate features
        4. Model inference - Apply ML models
        5. Post-processing - Format results
        6. Storage - Save to database
        7. Notification - Alert downstream systems
        """
        
        try:
            # Stage 1: Ingestion
            print("Stage 1: Ingesting data...")
            validated_data = await self.ingestor.validate(input_data)
            
            # Stage 2: Preprocessing
            print("Stage 2: Preprocessing...")
            clean_data = await self.preprocessor.clean(validated_data)
            normalized_data = await self.preprocessor.normalize(clean_data)
            
            # Stage 3: Feature Extraction (parallel)
            print("Stage 3: Extracting features...")
            features = await asyncio.gather(
                self.text_extractor.extract(normalized_data),
                self.numerical_extractor.extract(normalized_data),
                self.categorical_extractor.extract(normalized_data)
            )
            
            combined_features = self.feature_combiner.combine(features)
            
            # Stage 4: Model Inference (parallel models)
            print("Stage 4: Running models...")
            predictions = await asyncio.gather(
                self.classifier.predict(combined_features),
                self.regressor.predict(combined_features),
                self.clusterer.predict(combined_features)
            )
            
            # Stage 5: Post-processing
            print("Stage 5: Post-processing...")
            final_results = await self.postprocessor.format(predictions)
            
            # Stage 6: Storage
            print("Stage 6: Storing results...")
            await self.storage.save(final_results)
            
            # Stage 7: Notification
            print("Stage 7: Notifying systems...")
            await self.notifier.notify(final_results)
            
            return final_results
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            await self.error_handler.handle(e)
            raise
```

---

## Real-Time Processing Workflows

### Event-Driven Workflow

```python
class EventDrivenWorkflow:
    """
    React to events and trigger appropriate workflows.
    """
    
    def __init__(self):
        self.event_handlers = {
            "customer_signup": self.handle_signup,
            "purchase_completed": self.handle_purchase,
            "support_ticket": self.handle_support,
            "system_alert": self.handle_alert
        }
    
    async def process_event(self, event: Dict[str, Any]):
        """Route event to appropriate handler"""
        
        event_type = event.get("type")
        handler = self.event_handlers.get(event_type)
        
        if handler:
            await handler(event)
        else:
            print(f"⚠️ No handler for event type: {event_type}")
    
    async def handle_signup(self, event: Dict[str, Any]):
        """
        Customer signup workflow.
        
        Steps:
        1. Send welcome email
        2. Create onboarding tasks
        3. Assign customer success rep
        4. Schedule check-in
        """
        
        customer_data = event["data"]
        
        await asyncio.gather(
            self.email_agent.send_welcome(customer_data),
            self.onboarding_agent.create_tasks(customer_data),
            self.cs_agent.assign_rep(customer_data),
            self.scheduler.schedule_checkin(customer_data)
        )
    
    async def handle_purchase(self, event: Dict[str, Any]):
        """
        Purchase completion workflow.
        
        Steps:
        1. Send receipt
        2. Update CRM
        3. Trigger fulfillment
        4. Update analytics
        """
        
        purchase_data = event["data"]
        
        await asyncio.gather(
            self.email_agent.send_receipt(purchase_data),
            self.crm_agent.update_customer(purchase_data),
            self.fulfillment_agent.initiate(purchase_data),
            self.analytics_agent.record_purchase(purchase_data)
        )
```

---

## Decision Workflows

### Multi-Criteria Decision Workflow

```python
class DecisionWorkflow:
    """
    Make complex decisions based on multiple criteria.
    """
    
    async def investment_decision_workflow(self, opportunity: Dict[str, Any]):
        """
        Investment decision based on multiple analyses.
        
        Steps:
        1. Financial analysis
        2. Market analysis
        3. Risk assessment
        4. Competitive analysis
        5. Decision synthesis
        6. Recommendation generation
        """
        
        print("🎯 Evaluating investment opportunity...")
        
        # Parallel analysis
        analyses = await asyncio.gather(
            self.financial_agent.analyze(opportunity),
            self.market_agent.analyze(opportunity),
            self.risk_agent.assess(opportunity),
            self.competitive_agent.analyze(opportunity)
        )
        
        financial, market, risk, competitive = analyses
        
        # Synthesize decision
        print("⚖️ Synthesizing decision...")
        decision = await self.decision_agent.synthesize({
            "financial": financial,
            "market": market,
            "risk": risk,
            "competitive": competitive
        })
        
        # Generate recommendation
        print("📝 Generating recommendation...")
        recommendation = await self.recommender.generate(decision)
        
        return {
            "decision": decision,
            "recommendation": recommendation,
            "confidence": decision.confidence,
            "analyses": {
                "financial": financial,
                "market": market,
                "risk": risk,
                "competitive": competitive
            }
        }
```

---

## Monitoring & Error Handling

### Workflow Monitoring

```python
class WorkflowMonitor:
    """
    Monitor workflow execution and health.
    """
    
    def __init__(self):
        self.metrics = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "avg_duration": 0,
            "step_failures": {}
        }
    
    async def monitor_workflow(self, workflow_fn):
        """Decorator to monitor workflow execution"""
        
        async def wrapped(*args, **kwargs):
            start_time = datetime.now()
            self.metrics["executions"] += 1
            
            try:
                result = await workflow_fn(*args, **kwargs)
                
                self.metrics["successes"] += 1
                duration = (datetime.now() - start_time).total_seconds()
                
                # Update average duration
                current_avg = self.metrics["avg_duration"]
                count = self.metrics["executions"]
                self.metrics["avg_duration"] = (
                    (current_avg * (count - 1) + duration) / count
                )
                
                print(f"✅ Workflow completed in {duration:.2f}s")
                
                return result
                
            except Exception as e:
                self.metrics["failures"] += 1
                print(f"❌ Workflow failed: {e}")
                
                # Track step failures
                step_id = getattr(e, "step_id", "unknown")
                self.metrics["step_failures"][step_id] = (
                    self.metrics["step_failures"].get(step_id, 0) + 1
                )
                
                raise
        
        return wrapped
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate health report"""
        
        success_rate = (
            self.metrics["successes"] / self.metrics["executions"]
            if self.metrics["executions"] > 0 else 0
        )
        
        return {
            "total_executions": self.metrics["executions"],
            "success_rate": success_rate,
            "average_duration": self.metrics["avg_duration"],
            "most_common_failures": sorted(
                self.metrics["step_failures"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
```

### Error Recovery Strategies

```python
class ErrorRecoveryWorkflow:
    """
    Implement comprehensive error recovery.
    """
    
    async def execute_with_recovery(self, workflow_step):
        """Execute step with multiple recovery strategies"""
        
        try:
            # Try primary execution
            return await workflow_step.execute()
            
        except TimeoutError:
            # Strategy 1: Retry with longer timeout
            print("⏱️ Timeout - retrying with extended timeout...")
            return await self._retry_with_longer_timeout(workflow_step)
            
        except ConnectionError:
            # Strategy 2: Fallback to cached data
            print("🔌 Connection error - using cached data...")
            return await self._use_cached_data(workflow_step)
            
        except ValidationError as e:
            # Strategy 3: Request human intervention
            print("👤 Validation error - requesting human review...")
            return await self._request_human_review(workflow_step, e)
            
        except Exception as e:
            # Strategy 4: Graceful degradation
            print(f"⚠️ Unexpected error - graceful degradation...")
            return await self._graceful_degradation(workflow_step, e)
    
    async def _retry_with_longer_timeout(self, step):
        """Retry with increased timeout"""
        step.timeout_seconds *= 2
        return await step.execute()
    
    async def _use_cached_data(self, step):
        """Fall back to cached results"""
        cached = self.cache.get(step.step_id)
        if cached:
            return cached
        raise RuntimeError("No cached data available")
    
    async def _request_human_review(self, step, error):
        """Add to human review queue"""
        await self.human_queue.add({
            "step": step.step_id,
            "error": str(error),
            "timestamp": datetime.now()
        })
        return {"status": "pending_review"}
    
    async def _graceful_degradation(self, step, error):
        """Return partial results"""
        return {
            "status": "degraded",
            "error": str(error),
            "partial_results": step.get_partial_results()
        }
```

---

## Best Practices

### 1. Workflow Design Principles

```python
# ✅ GOOD: Clear, modular workflow
class GoodWorkflow:
    async def process_order(self, order):
        # Each step has a single responsibility
        validated = await self.validate_order(order)
        charged = await self.charge_payment(validated)
        shipped = await self.ship_order(charged)
        confirmed = await self.send_confirmation(shipped)
        return confirmed

# ❌ BAD: Monolithic, hard to maintain
class BadWorkflow:
    async def process_order(self, order):
        # Everything in one method
        # Hard to test, debug, or modify
        result = await self.do_everything(order)
        return result
```

### 2. Error Handling

```python
# ✅ GOOD: Specific error handling
try:
    result = await agent.process()
except ValidationError as e:
    await handle_validation_error(e)
except TimeoutError as e:
    await handle_timeout(e)
except Exception as e:
    await handle_unexpected_error(e)

# ❌ BAD: Generic error handling
try:
    result = await agent.process()
except Exception:
    pass  # Silently fails
```

### 3. State Management

```python
# ✅ GOOD: Immutable workflow state
class ImmutableWorkflowState:
    def __init__(self, data):
        self._data = data.copy()
    
    def update(self, key, value):
        new_data = self._data.copy()
        new_data[key] = value
        return ImmutableWorkflowState(new_data)

# ❌ BAD: Mutable shared state
class MutableWorkflowState:
    data = {}  # Shared across instances!
```

### 4. Testing Workflows

```python
import pytest

class TestWorkflow:
    @pytest.fixture
    async def workflow(self):
        return MyWorkflow()
    
    async def test_successful_execution(self, workflow):
        result = await workflow.execute({"input": "test"})
        assert result["status"] == "success"
    
    async def test_error_recovery(self, workflow):
        # Inject error
        workflow.agent.should_fail = True
        
        result = await workflow.execute({"input": "test"})
        
        # Should recover gracefully
        assert result["status"] == "recovered"
    
    async def test_timeout_handling(self, workflow):
        # Set short timeout
        workflow.timeout = 0.1
        
        with pytest.raises(TimeoutError):
            await workflow.execute({"slow_input": "test"})
```

### 5. Performance Optimization

```python
# ✅ GOOD: Parallel independent operations
async def optimized_workflow(self):
    # Execute in parallel
    results = await asyncio.gather(
        self.agent1.process(),
        self.agent2.process(),
        self.agent3.process()
    )
    return self.synthesize(results)

# ❌ BAD: Sequential independent operations
async def slow_workflow(self):
    # Unnecessarily sequential
    result1 = await self.agent1.process()
    result2 = await self.agent2.process()
    result3 = await self.agent3.process()
    return self.synthesize([result1, result2, result3])
```

---

## Workflow Templates

### Template 1: Analysis Workflow

```python
async def analysis_workflow_template(data):
    """
    Generic analysis workflow template.
    
    Phases:
    1. Data validation
    2. Parallel analysis
    3. Synthesis
    4. Report generation
    """
    
    # Phase 1: Validate
    validated = await validator.validate(data)
    
    # Phase 2: Analyze (parallel)
    analyses = await asyncio.gather(
        analyzer1.analyze(validated),
        analyzer2.analyze(validated),
        analyzer3.analyze(validated)
    )
    
    # Phase 3: Synthesize
    synthesis = await synthesizer.combine(analyses)
    
    # Phase 4: Report
    report = await reporter.generate(synthesis)
    
    return report
```

### Template 2: Decision Workflow

```python
async def decision_workflow_template(input_data):
    """
    Generic decision-making workflow template.
    
    Phases:
    1. Gather information
    2. Evaluate options
    3. Make decision
    4. Generate recommendations
    """
    
    # Phase 1: Information gathering
    info = await info_gatherer.collect(input_data)
    
    # Phase 2: Evaluate options
    options = await option_generator.generate(info)
    evaluations = await asyncio.gather(*[
        evaluator.evaluate(option) for option in options
    ])
    
    # Phase 3: Decision
    decision = await decision_maker.decide(evaluations)
    
    # Phase 4: Recommendations
    recommendations = await recommender.generate(decision)
    
    return {
        "decision": decision,
        "recommendations": recommendations,
        "confidence": decision.confidence
    }
```

---

## Related Documentation

- [Agent-to-Agent Communication (A2A)](./A2A.md) - Inter-agent communication
- [Orchestration Patterns](./ORCHESTRATION.md) - Multi-agent coordination
- [MLOps Guide](./MLOPS.md) - Production deployment

---

**Next Steps**: Learn about [MLOps practices](./MLOPS.md) for deploying agent workflows in production.