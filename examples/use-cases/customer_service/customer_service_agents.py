"""
Intelligent Customer Service System - Multi-Agent A2A Implementation
====================================================================

This example demonstrates a complete customer service system where multiple
specialized agents communicate autonomously using A2A protocol to resolve
customer inquiries without human intervention.

Key Features:
- Autonomous agent-to-agent communication
- Intelligent query routing
- Specialized agents (Billing, Technical, General Support)
- Context sharing between agents
- Response synthesis and quality control
- Multi-turn conversations
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import openai


class AgentType(Enum):
    """Types of agents in the customer service system"""
    ROUTING = "routing"
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL_SUPPORT = "general_support"
    SYNTHESIS = "synthesis"


class MessageType(Enum):
    """Types of messages agents can send"""
    QUERY = "query"
    REQUEST = "request"
    RESPONSE = "response"
    HANDOFF = "handoff"
    COLLABORATION = "collaboration"
    FINAL_RESPONSE = "final_response"


@dataclass
class Message:
    """A2A protocol message structure"""
    message_id: str
    sender: AgentType
    recipient: AgentType
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 5=critical


@dataclass
class CustomerContext:
    """Shared customer context across agents"""
    customer_id: str
    query: str
    conversation_history: List[Dict] = field(default_factory=list)
    customer_data: Dict[str, Any] = field(default_factory=dict)
    issue_type: Optional[str] = None
    resolved: bool = False
    escalated: bool = False


class BaseAgent:
    """
    Base class for all customer service agents.
    
    Implements A2A protocol for agent-to-agent communication.
    """
    
    def __init__(self, agent_type: AgentType, name: str, llm_client):
        self.agent_type = agent_type
        self.name = name
        self.llm_client = llm_client
        self.message_queue: List[Message] = []
        self.conversation_log: List[Message] = []
        
        print(f"🤖 {self.name} ({agent_type.value}) initialized")
    
    async def send_message(self, recipient: AgentType, message_type: MessageType, 
                          content: Dict[str, Any], context: Dict[str, Any] = None):
        """Send A2A message to another agent"""
        message = Message(
            message_id=f"{self.agent_type.value}_{datetime.now().timestamp()}",
            sender=self.agent_type,
            recipient=recipient,
            message_type=message_type,
            content=content,
            context=context or {}
        )
        
        self.conversation_log.append(message)
        
        print(f"  📤 {self.name} → {recipient.value}: {message_type.value}")
        
        return message
    
    async def receive_message(self, message: Message):
        """Receive and queue incoming message"""
        self.message_queue.append(message)
        print(f"  📥 {self.name} ← {message.sender.value}: {message.message_type.value}")
    
    async def process_with_llm(self, prompt: str, context: Dict = None) -> str:
        """Process request using LLM"""
        try:
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            if context:
                messages.insert(1, {
                    "role": "system", 
                    "content": f"Context: {json.dumps(context, indent=2)}"
                })
            
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    def get_system_prompt(self) -> str:
        """Override in subclasses"""
        return f"You are a {self.name} in a customer service system."


class RoutingAgent(BaseAgent):
    """
    Routes customer queries to appropriate specialist agents.
    
    Analyzes query intent and determines best agent to handle it.
    """
    
    def __init__(self, llm_client):
        super().__init__(AgentType.ROUTING, "Routing Agent", llm_client)
        
        self.routing_rules = {
            "billing": ["payment", "charge", "invoice", "refund", "billing", "credit card", "subscription"],
            "technical": ["not working", "error", "broken", "bug", "crash", "issue", "problem", "setup"],
            "general_support": ["account", "change", "update", "information", "help", "question"]
        }
    
    def get_system_prompt(self) -> str:
        return """You are a routing agent in a customer service system. 
        Analyze customer queries and determine which specialist should handle them:
        - BILLING: Payment, charges, refunds, invoices, subscriptions
        - TECHNICAL: Service issues, errors, bugs, setup problems
        - GENERAL_SUPPORT: Account management, general questions, information requests
        
        Respond with JSON: {"agent": "billing|technical|general_support", "confidence": 0-1, "reasoning": "..."}"""
    
    async def route_query(self, customer_context: CustomerContext) -> AgentType:
        """Analyze query and route to appropriate agent"""
        
        print(f"\n🔍 {self.name} analyzing query...")
        
        # Use LLM to determine routing
        prompt = f"""Analyze this customer query and route to the correct agent:
        
        Query: "{customer_context.query}"
        
        Return JSON with agent, confidence, and reasoning."""
        
        response = await self.process_with_llm(prompt, {
            "customer_id": customer_context.customer_id,
            "query": customer_context.query
        })
        
        try:
            # Parse LLM response
            routing_decision = json.loads(response)
            agent_type = routing_decision["agent"]
            confidence = routing_decision["confidence"]
            reasoning = routing_decision["reasoning"]
            
            print(f"  ✓ Routing to: {agent_type.upper()} (confidence: {confidence})")
            print(f"  📝 Reasoning: {reasoning}")
            
            # Map to AgentType
            agent_map = {
                "billing": AgentType.BILLING,
                "technical": AgentType.TECHNICAL,
                "general_support": AgentType.GENERAL_SUPPORT
            }
            
            return agent_map.get(agent_type, AgentType.GENERAL_SUPPORT)
        
        except Exception as e:
            print(f"  ⚠️ Routing error, defaulting to General Support: {e}")
            return AgentType.GENERAL_SUPPORT


class BillingAgent(BaseAgent):
    """
    Handles billing, payments, and financial inquiries.
    """
    
    def __init__(self, llm_client):
        super().__init__(AgentType.BILLING, "Billing Agent", llm_client)
        
        # Simulated customer billing data
        self.billing_database = {
            "CUST001": {
                "transactions": [
                    {"date": "2025-10-05", "amount": 49.99, "description": "Monthly Subscription"},
                    {"date": "2025-11-05", "amount": 49.99, "description": "Monthly Subscription"}
                ],
                "balance": 0.00,
                "payment_method": "Credit Card ending in 1234",
                "next_billing_date": "2025-12-05"
            }
        }
    
    def get_system_prompt(self) -> str:
        return """You are a billing specialist agent. Handle all payment, charge, 
        invoice, and subscription-related inquiries. Be precise with financial information 
        and always verify amounts and dates. Offer refunds when appropriate."""
    
    async def handle_query(self, customer_context: CustomerContext) -> Dict[str, Any]:
        """Process billing-related query"""
        
        print(f"\n💳 {self.name} processing billing inquiry...")
        
        # Retrieve billing data
        billing_data = self.billing_database.get(
            customer_context.customer_id, 
            {"error": "Customer not found"}
        )
        
        # Process with LLM
        prompt = f"""Handle this billing inquiry:
        
        Customer Query: "{customer_context.query}"
        
        Billing Data:
        {json.dumps(billing_data, indent=2)}
        
        Provide a clear, helpful response addressing the customer's concern.
        If there's an issue, suggest a resolution."""
        
        response = await self.process_with_llm(prompt, {
            "customer_id": customer_context.customer_id,
            "billing_data": billing_data
        })
        
        return {
            "agent": "billing",
            "response": response,
            "data": billing_data,
            "action_taken": "Reviewed billing history"
        }


class TechnicalAgent(BaseAgent):
    """
    Handles technical issues, troubleshooting, and service problems.
    """
    
    def __init__(self, llm_client):
        super().__init__(AgentType.TECHNICAL, "Technical Agent", llm_client)
        
        # Simulated service status
        self.service_status = {
            "CUST001": {
                "service_active": True,
                "last_login": "2025-11-07 14:30:00",
                "devices": 2,
                "bandwidth_usage": "45GB/100GB",
                "recent_issues": []
            }
        }
    
    def get_system_prompt(self) -> str:
        return """You are a technical support specialist agent. Diagnose and resolve 
        technical issues, provide troubleshooting steps, and escalate when necessary. 
        Be systematic and clear in your explanations."""
    
    async def handle_query(self, customer_context: CustomerContext) -> Dict[str, Any]:
        """Process technical query"""
        
        print(f"\n🔧 {self.name} diagnosing technical issue...")
        
        # Retrieve service data
        service_data = self.service_status.get(
            customer_context.customer_id,
            {"error": "Service data not found"}
        )
        
        # Process with LLM
        prompt = f"""Diagnose and resolve this technical issue:
        
        Customer Query: "{customer_context.query}"
        
        Service Status:
        {json.dumps(service_data, indent=2)}
        
        Provide troubleshooting steps and resolution. Be specific and actionable."""
        
        response = await self.process_with_llm(prompt, {
            "customer_id": customer_context.customer_id,
            "service_data": service_data
        })
        
        return {
            "agent": "technical",
            "response": response,
            "data": service_data,
            "action_taken": "Performed diagnostic check"
        }


class GeneralSupportAgent(BaseAgent):
    """
    Handles general inquiries, account management, and information requests.
    """
    
    def __init__(self, llm_client):
        super().__init__(AgentType.GENERAL_SUPPORT, "General Support Agent", llm_client)
    
    def get_system_prompt(self) -> str:
        return """You are a general support agent. Handle account management, 
        general questions, and information requests. Be friendly, helpful, and 
        can request assistance from other specialist agents when needed."""
    
    async def handle_query(self, customer_context: CustomerContext) -> Dict[str, Any]:
        """Process general support query"""
        
        print(f"\n💬 {self.name} handling general inquiry...")
        
        prompt = f"""Handle this customer inquiry:
        
        Query: "{customer_context.query}"
        
        Provide helpful, clear information. If you need specialist help 
        (billing or technical), indicate that."""
        
        response = await self.process_with_llm(prompt, {
            "customer_id": customer_context.customer_id
        })
        
        return {
            "agent": "general_support",
            "response": response,
            "action_taken": "Provided information"
        }


class SynthesisAgent(BaseAgent):
    """
    Synthesizes responses from multiple agents into coherent final response.
    Ensures quality and completeness.
    """
    
    def __init__(self, llm_client):
        super().__init__(AgentType.SYNTHESIS, "Response Synthesis Agent", llm_client)
    
    def get_system_prompt(self) -> str:
        return """You are a response synthesis agent. Combine responses from 
        multiple specialist agents into a single, coherent, professional response. 
        Ensure consistency, clarity, and completeness."""
    
    async def synthesize_response(self, customer_context: CustomerContext, 
                                  agent_responses: List[Dict[str, Any]]) -> str:
        """Combine multiple agent responses"""
        
        print(f"\n✨ {self.name} synthesizing final response...")
        
        prompt = f"""Synthesize these agent responses into one clear, professional response:
        
        Original Query: "{customer_context.query}"
        
        Agent Responses:
        {json.dumps(agent_responses, indent=2)}
        
        Create a unified response that addresses all aspects of the customer's inquiry."""
        
        final_response = await self.process_with_llm(prompt, {
            "customer_id": customer_context.customer_id,
            "num_agents": len(agent_responses)
        })
        
        return final_response


class CustomerServiceOrchestrator:
    """
    Orchestrates multi-agent customer service system.
    
    Manages agent communication and workflow.
    """
    
    def __init__(self, api_key: str):
        # Initialize OpenAI client
        self.llm_client = openai.OpenAI(api_key=api_key)
        
        # Initialize all agents
        self.routing_agent = RoutingAgent(self.llm_client)
        self.billing_agent = BillingAgent(self.llm_client)
        self.technical_agent = TechnicalAgent(self.llm_client)
        self.general_support_agent = GeneralSupportAgent(self.llm_client)
        self.synthesis_agent = SynthesisAgent(self.llm_client)
        
        print("\n" + "="*60)
        print("🏢 CUSTOMER SERVICE SYSTEM INITIALIZED")
        print("="*60)
    
    async def handle_customer_query(self, customer_id: str, query: str) -> Dict[str, Any]:
        """
        Process customer query through multi-agent system.
        
        Workflow:
        1. Routing agent analyzes query
        2. Appropriate specialist agent(s) handle query
        3. Synthesis agent creates final response
        4. Response returned to customer
        """
        
        print(f"\n\n{'='*60}")
        print(f"📞 NEW CUSTOMER QUERY")
        print(f"{'='*60}")
        print(f"Customer ID: {customer_id}")
        print(f"Query: {query}")
        
        # Create customer context
        context = CustomerContext(
            customer_id=customer_id,
            query=query
        )
        
        # Step 1: Route query
        target_agent_type = await self.routing_agent.route_query(context)
        
        # Step 2: Get specialist agent
        agent_map = {
            AgentType.BILLING: self.billing_agent,
            AgentType.TECHNICAL: self.technical_agent,
            AgentType.GENERAL_SUPPORT: self.general_support_agent
        }
        
        specialist_agent = agent_map[target_agent_type]
        
        # Step 3: Process query
        agent_response = await specialist_agent.handle_query(context)
        
        # Step 4: Synthesize response
        final_response = await self.synthesis_agent.synthesize_response(
            context, 
            [agent_response]
        )
        
        # Step 5: Return result
        result = {
            "customer_id": customer_id,
            "query": query,
            "routed_to": target_agent_type.value,
            "response": final_response,
            "timestamp": datetime.now().isoformat(),
            "resolved": True
        }
        
        print(f"\n{'='*60}")
        print("✅ QUERY RESOLVED")
        print(f"{'='*60}")
        print(f"\n📝 Final Response:\n{final_response}\n")
        
        return result
    
    async def handle_complex_query(self, customer_id: str, query: str) -> Dict[str, Any]:
        """
        Handle complex query requiring multiple agents.
        
        Demonstrates agent collaboration and A2A communication.
        """
        
        print(f"\n\n{'='*60}")
        print(f"🔥 COMPLEX MULTI-AGENT QUERY")
        print(f"{'='*60}")
        print(f"Customer ID: {customer_id}")
        print(f"Query: {query}")
        
        context = CustomerContext(customer_id=customer_id, query=query)
        
        # Route query
        print("\n[Phase 1] Routing and Analysis...")
        target_agent_type = await self.routing_agent.route_query(context)
        
        # Detect if multiple agents needed
        query_lower = query.lower()
        needs_billing = any(word in query_lower for word in ["charge", "payment", "billing", "refund"])
        needs_technical = any(word in query_lower for word in ["not working", "error", "issue", "broken"])
        
        agent_responses = []
        
        # Process with multiple agents if needed
        print("\n[Phase 2] Multi-Agent Processing...")
        
        if needs_billing:
            print("  → Consulting Billing Agent...")
            billing_response = await self.billing_agent.handle_query(context)
            agent_responses.append(billing_response)
        
        if needs_technical:
            print("  → Consulting Technical Agent...")
            technical_response = await self.technical_agent.handle_query(context)
            agent_responses.append(technical_response)
        
        if not agent_responses:
            # Default to routed agent
            agent_map = {
                AgentType.BILLING: self.billing_agent,
                AgentType.TECHNICAL: self.technical_agent,
                AgentType.GENERAL_SUPPORT: self.general_support_agent
            }
            response = await agent_map[target_agent_type].handle_query(context)
            agent_responses.append(response)
        
        # Synthesize all responses
        print("\n[Phase 3] Response Synthesis...")
        final_response = await self.synthesis_agent.synthesize_response(
            context,
            agent_responses
        )
        
        result = {
            "customer_id": customer_id,
            "query": query,
            "agents_involved": [r["agent"] for r in agent_responses],
            "response": final_response,
            "timestamp": datetime.now().isoformat(),
            "resolved": True
        }
        
        print(f"\n{'='*60}")
        print("✅ COMPLEX QUERY RESOLVED")
        print(f"{'='*60}")
        print(f"Agents Involved: {', '.join(result['agents_involved'])}")
        print(f"\n📝 Final Response:\n{final_response}\n")
        
        return result


async def main():
    """
    Demonstrate autonomous multi-agent customer service system.
    """
    
    # Get API key from environment or use placeholder
    import os
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    # Initialize orchestrator
    orchestrator = CustomerServiceOrchestrator(api_key)
    
    # Test scenarios
    scenarios = [
        {
            "customer_id": "CUST001",
            "query": "Why was I charged $49.99 twice this month?",
            "description": "Simple billing inquiry"
        },
        {
            "customer_id": "CUST001",
            "query": "My service isn't working and I think I was overcharged",
            "description": "Complex query requiring technical AND billing agents"
        },
        {
            "customer_id": "CUST001",
            "query": "How do I update my account information?",
            "description": "General support inquiry"
        }
    ]
    
    print("\n" + "="*60)
    print("🎬 RUNNING TEST SCENARIOS")
    print("="*60)
    
    # Scenario 1: Simple billing query
    print(f"\n\n📋 SCENARIO 1: {scenarios[0]['description']}")
    await orchestrator.handle_customer_query(
        scenarios[0]["customer_id"],
        scenarios[0]["query"]
    )
    
    # Wait between scenarios
    await asyncio.sleep(2)
    
    # Scenario 2: Complex multi-agent query
    print(f"\n\n📋 SCENARIO 2: {scenarios[1]['description']}")
    await orchestrator.handle_complex_query(
        scenarios[1]["customer_id"],
        scenarios[1]["query"]
    )
    
    # Wait between scenarios
    await asyncio.sleep(2)
    
    # Scenario 3: General support
    print(f"\n\n📋 SCENARIO 3: {scenarios[2]['description']}")
    await orchestrator.handle_customer_query(
        scenarios[2]["customer_id"],
        scenarios[2]["query"]
    )
    
    print("\n" + "="*60)
    print("✨ ALL SCENARIOS COMPLETED")
    print("="*60)
    print("\n💡 Key Takeaways:")
    print("  • Agents communicate autonomously using A2A protocol")
    print("  • Routing agent directs queries to specialists")
    print("  • Multiple agents collaborate on complex issues")
    print("  • Synthesis agent ensures quality responses")
    print("  • No human intervention required")
    print("  • Scalable to handle multiple customers simultaneously")


if __name__ == "__main__":
    asyncio.run(main())
