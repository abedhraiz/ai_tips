# A2P - Agent-to-Person Communication

## Overview

Agent-to-Person (A2P) communication defines the patterns and best practices for AI agents interacting with human users. This protocol focuses on creating natural, effective, and trustworthy human-AI interactions.

## Key Principles

1. **Clarity**: Clear, unambiguous communication
2. **Transparency**: Honest about capabilities and limitations
3. **Empathy**: Understanding user needs and emotions
4. **Adaptability**: Adjusting to user preferences and context
5. **Safety**: Protecting user privacy and security

## Communication Patterns

### 1. Conversational Interface

The most common A2P pattern - natural dialogue between human and AI.

```python
import openai
import os

class ConversationalAgent:
    def __init__(self, model="gpt-4"):
        self.client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.model = model
        self.conversation_history = []
        
    def chat(self, user_message):
        """Send message and get response"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Get AI response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def reset(self):
        """Clear conversation history"""
        self.conversation_history = []

# Usage
agent = ConversationalAgent()

user_input = "I need help planning a trip to Japan"
response = agent.chat(user_input)
print(f"AI: {response}")

user_input = "What's the best time to visit?"
response = agent.chat(user_input)
print(f"AI: {response}")
```

**Benefits**: Natural interaction, context preservation, easy to use

---

### 2. Clarification Loop

AI requests clarification when user intent is ambiguous.

```python
class ClarifyingAgent:
    def __init__(self):
        self.client = openai.OpenAI()
        self.pending_clarification = None
        
    def process_request(self, user_message, context=None):
        """Process request with clarification if needed"""
        
        # Check if message needs clarification
        clarification_check = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Determine if this request is ambiguous and needs clarification. Respond with JSON: {\"needs_clarification\": true/false, \"questions\": [\"question1\", ...]}"},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(clarification_check.choices[0].message.content)
        
        if result['needs_clarification']:
            self.pending_clarification = user_message
            return {
                "status": "needs_clarification",
                "questions": result['questions']
            }
        else:
            # Process the request
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide clear, accurate responses."},
                    {"role": "user", "content": user_message}
                ]
            )
            
            return {
                "status": "completed",
                "response": response.choices[0].message.content
            }
    
    def provide_clarification(self, clarification):
        """User provides clarification"""
        combined = f"{self.pending_clarification}\n\nClarification: {clarification}"
        self.pending_clarification = None
        return self.process_request(combined)

# Example usage
agent = ClarifyingAgent()

result = agent.process_request("Book a flight")

if result['status'] == 'needs_clarification':
    print("AI: I need more information:")
    for q in result['questions']:
        print(f"  - {q}")
    
    # User provides clarification
    clarification = "From New York to London, economy class, departing next Monday"
    final_result = agent.provide_clarification(clarification)
    print(f"\nAI: {final_result['response']}")
```

**Benefits**: Reduces errors, ensures accurate understanding, better outcomes

---

### 3. Progressive Disclosure

AI reveals information gradually based on user needs.

```python
class ProgressiveDisclosureAgent:
    def __init__(self):
        self.client = openai.OpenAI()
        self.detail_level = "summary"  # summary, moderate, detailed
        
    def set_detail_level(self, level):
        """User sets preferred detail level"""
        self.detail_level = level
        
    def respond(self, question):
        """Respond with appropriate detail level"""
        system_prompts = {
            "summary": "Provide a brief, high-level answer (2-3 sentences).",
            "moderate": "Provide a moderate explanation with key details (1-2 paragraphs).",
            "detailed": "Provide a comprehensive, detailed explanation with examples."
        }
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompts[self.detail_level]},
                {"role": "user", "content": question}
            ]
        )
        
        answer = response.choices[0].message.content
        
        # Offer to elaborate
        if self.detail_level != "detailed":
            answer += "\n\n[Would you like more details? Say 'tell me more']"
            
        return answer

# Usage
agent = ProgressiveDisclosureAgent()

# Start with summary
agent.set_detail_level("summary")
response = agent.respond("What is quantum computing?")
print(f"Summary:\n{response}\n")

# User wants more details
agent.set_detail_level("moderate")
response = agent.respond("Tell me more about quantum computing")
print(f"Moderate:\n{response}\n")

# User wants comprehensive explanation
agent.set_detail_level("detailed")
response = agent.respond("Explain quantum computing in detail")
print(f"Detailed:\n{response}")
```

**Benefits**: Reduces cognitive overload, user controls depth, improves engagement

---

### 4. Feedback and Learning

AI learns from user feedback to improve responses.

```python
class LearningAgent:
    def __init__(self):
        self.client = openai.OpenAI()
        self.feedback_history = []
        self.preferences = {}
        
    def respond(self, question):
        """Generate response considering past feedback"""
        # Build context from preferences
        preference_context = self._build_preference_context()
        
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. User preferences: {preference_context}"},
            {"role": "user", "content": question}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        answer = response.choices[0].message.content
        response_id = len(self.feedback_history)
        
        return {
            "response_id": response_id,
            "answer": answer,
            "feedback_prompt": "Was this response helpful? (yes/no/suggest improvement)"
        }
    
    def receive_feedback(self, response_id, feedback, suggestion=None):
        """Process user feedback"""
        self.feedback_history.append({
            "response_id": response_id,
            "feedback": feedback,
            "suggestion": suggestion
        })
        
        # Update preferences based on feedback
        if suggestion:
            self._update_preferences(suggestion)
            
        return "Thank you for your feedback! I'll use it to improve."
    
    def _build_preference_context(self):
        """Build context from user preferences"""
        if not self.preferences:
            return "None yet"
        
        return ", ".join([f"{k}: {v}" for k, v in self.preferences.items()])
    
    def _update_preferences(self, suggestion):
        """Update preferences from feedback"""
        # Use AI to extract preferences
        extraction = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract user preferences from feedback. Return JSON with preference categories."},
                {"role": "user", "content": suggestion}
            ],
            response_format={"type": "json_object"}
        )
        
        import json
        new_prefs = json.loads(extraction.choices[0].message.content)
        self.preferences.update(new_prefs)

# Usage
agent = LearningAgent()

result = agent.respond("Explain machine learning")
print(f"AI: {result['answer']}\n")
print(result['feedback_prompt'])

# User provides feedback
agent.receive_feedback(
    response_id=result['response_id'],
    feedback="no",
    suggestion="Please use more practical examples and less jargon"
)

# Next response will incorporate feedback
result2 = agent.respond("Explain neural networks")
print(f"\nAI (improved): {result2['answer']}")
```

**Benefits**: Personalization, continuous improvement, better user satisfaction

---

### 5. Multi-Turn Task Completion

AI guides users through complex multi-step tasks.

```python
class TaskCompletionAgent:
    def __init__(self):
        self.client = openai.OpenAI()
        self.current_task = None
        self.task_state = {}
        
    def start_task(self, task_description):
        """Initialize a new task"""
        # Break down task into steps
        planning = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Break down this task into clear, sequential steps. Return JSON with 'steps' array."},
                {"role": "user", "content": task_description}
            ],
            response_format={"type": "json_object"}
        )
        
        import json
        plan = json.loads(planning.choices[0].message.content)
        
        self.current_task = task_description
        self.task_state = {
            "steps": plan['steps'],
            "current_step": 0,
            "completed_steps": [],
            "step_data": {}
        }
        
        return {
            "task": task_description,
            "total_steps": len(plan['steps']),
            "first_step": plan['steps'][0],
            "prompt": self._format_step_prompt(plan['steps'][0], 0)
        }
    
    def complete_step(self, user_input):
        """User completes current step"""
        current_step_idx = self.task_state['current_step']
        current_step = self.task_state['steps'][current_step_idx]
        
        # Store step data
        self.task_state['step_data'][current_step_idx] = user_input
        self.task_state['completed_steps'].append(current_step)
        self.task_state['current_step'] += 1
        
        # Check if task is complete
        if self.task_state['current_step'] >= len(self.task_state['steps']):
            return self._finalize_task()
        
        # Move to next step
        next_step = self.task_state['steps'][self.task_state['current_step']]
        return {
            "status": "in_progress",
            "progress": f"{self.task_state['current_step']}/{len(self.task_state['steps'])}",
            "next_step": next_step,
            "prompt": self._format_step_prompt(next_step, self.task_state['current_step'])
        }
    
    def _format_step_prompt(self, step, step_num):
        """Format prompt for current step"""
        total = len(self.task_state['steps'])
        return f"Step {step_num + 1} of {total}: {step}\n\nPlease provide the required information."
    
    def _finalize_task(self):
        """Complete the task"""
        # Generate summary
        summary_data = "\n".join([
            f"Step {i+1}: {step}\nData: {self.task_state['step_data'].get(i, 'N/A')}"
            for i, step in enumerate(self.task_state['steps'])
        ])
        
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Generate a completion summary for this task"},
                {"role": "user", "content": f"Task: {self.current_task}\n\n{summary_data}"}
            ]
        )
        
        return {
            "status": "completed",
            "summary": completion.choices[0].message.content,
            "all_data": self.task_state['step_data']
        }

# Usage
agent = TaskCompletionAgent()

# Start complex task
result = agent.start_task("Help me create a business plan")
print(f"Task: {result['task']}")
print(f"Total steps: {result['total_steps']}")
print(f"\n{result['prompt']}")

# Complete step 1
user_response = "A mobile app for language learning"
result = agent.complete_step(user_response)
print(f"\nProgress: {result['progress']}")
print(result['prompt'])

# Complete step 2
user_response = "Young professionals aged 25-35 who want to learn languages for travel"
result = agent.complete_step(user_response)

# Continue until task is complete...
```

**Benefits**: Structured guidance, progress tracking, reduced user error

---

## A2P Best Practices

### 1. Set Clear Expectations
```python
initial_message = """
Hi! I'm your AI assistant. Here's what I can help with:
✓ Answer questions and provide information
✓ Help with tasks and planning
✓ Learn from your feedback

Please note:
✗ I don't have access to real-time data
✗ I can't perform actions outside this conversation
✗ I may occasionally make mistakes - please verify important information

How can I help you today?
"""
```

### 2. Acknowledge Limitations
```python
if "real-time stock price" in user_message:
    response = "I don't have access to real-time stock data. I recommend checking financial websites like Yahoo Finance or Bloomberg for current prices. I can help you understand how to interpret stock information once you have it."
```

### 3. Provide Actionable Responses
```python
# Bad: Vague response
"You should improve your website's SEO."

# Good: Actionable steps
"""
To improve your website's SEO, follow these steps:

1. Optimize title tags (50-60 characters)
2. Add meta descriptions (150-160 characters)
3. Use header tags (H1, H2, H3) properly
4. Improve page load speed (target <3 seconds)
5. Build quality backlinks

Would you like detailed guidance on any of these steps?
"""
```

### 4. Respect Privacy
```python
class PrivacyAwareAgent:
    def check_for_sensitive_data(self, message):
        """Detect and warn about sensitive information"""
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'  # Email
        ]
        
        import re
        for pattern in sensitive_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "⚠️ Warning: Your message may contain sensitive information. Please avoid sharing personal data like social security numbers, credit card numbers, or passwords."
        
        return None
```

### 5. Offer Multiple Options
```python
def provide_options(question):
    return """
    I can help you with this in several ways:
    
    A) Quick overview (2-3 minutes read)
    B) Detailed explanation with examples (10-15 minutes)
    C) Step-by-step tutorial (30+ minutes)
    D) Just answer specific questions you have
    
    Which would you prefer? (Reply with A, B, C, or D)
    """
```

## Accessibility Considerations

### 1. Screen Reader Friendly
- Use clear, descriptive text
- Avoid relying solely on formatting
- Provide alt-text for any visual elements

### 2. Simple Language Option
```python
def simplify_response(complex_text):
    """Provide simpler alternative"""
    simplified = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Rewrite this at a 6th grade reading level"},
            {"role": "user", "content": complex_text}
        ]
    )
    return simplified.choices[0].message.content
```

### 3. Multiple Format Support
- Text responses
- Bulleted lists
- Step-by-step instructions
- Visual diagrams (when possible)

## Error Handling

```python
class RobustA2PAgent:
    def safe_respond(self, user_message):
        """Handle errors gracefully"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": user_message}],
                timeout=30
            )
            return response.choices[0].message.content
            
        except openai.APITimeoutError:
            return "I'm taking longer than usual to respond. Let me try again with a simpler approach..."
            
        except openai.APIError as e:
            return f"I encountered a technical issue. Please try rephrasing your question or try again in a moment."
            
        except Exception as e:
            return "I apologize, but I'm having trouble processing your request. Could you please rephrase it?"
```

## Evaluation Metrics

1. **User Satisfaction**: Survey ratings, feedback scores
2. **Task Completion Rate**: % of tasks successfully completed
3. **Clarification Rate**: How often AI needs to ask for clarification
4. **Response Relevance**: User ratings of response quality
5. **Conversation Length**: Average turns needed to resolve queries

---

**Related Protocols**: [A2A](./A2A.md) | [A2S](./A2S.md) | [MCP](./MCP.md) | [Orchestration](./ORCHESTRATION.md)
