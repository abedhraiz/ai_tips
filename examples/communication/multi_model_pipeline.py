"""
Example: Multi-Model Communication Workflow
============================================

This example demonstrates how different AI models communicate and work together
to accomplish complex tasks. We'll build a document intelligence system that:

1. VLM extracts text and structure from document images
2. LLM summarizes and analyzes the content
3. MLM extracts named entities and classifies sentiment
4. Results are aggregated and presented

Communication Pattern: Sequential Pipeline with Data Passing
"""

import openai
from transformers import pipeline
from PIL import Image
import json
import os
from typing import Dict, List, Any

# Configure APIs
openai_client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


class MultiModelPipeline:
    """
    Orchestrates multiple AI models to process documents end-to-end.
    
    This demonstrates inter-model communication where:
    - Each model has a specific role
    - Output from one model becomes input to the next
    - Results are aggregated for final output
    """
    
    def __init__(self):
        print("Initializing Multi-Model Pipeline...")
        
        # Model 1: Vision Language Model (VLM) for OCR and understanding
        self.vlm = openai_client
        
        # Model 2: Large Language Model (LLM) for summarization
        self.llm = openai_client
        
        # Model 3: Masked Language Model (MLM) for entity extraction
        print("Loading NER model...")
        self.ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        
        # Model 4: Sentiment analysis
        print("Loading sentiment model...")
        self.sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        print("✓ All models loaded successfully\n")
    
    def process_document(self, image_path: str) -> Dict[str, Any]:
        """
        Process a document through multiple AI models.
        
        Args:
            image_path: Path to document image
            
        Returns:
            Complete analysis results
        """
        print(f"Processing document: {image_path}")
        print("="*60)
        
        # Step 1: VLM extracts text and structure
        print("\n[Step 1] VLM: Extracting text and understanding structure...")
        extracted_data = self._extract_with_vlm(image_path)
        print(f"✓ Extracted {len(extracted_data['text'])} characters")
        
        # Step 2: LLM summarizes content
        print("\n[Step 2] LLM: Generating summary and key points...")
        summary = self._summarize_with_llm(extracted_data['text'])
        print(f"✓ Generated summary ({len(summary['summary'])} characters)")
        
        # Step 3: MLM extracts named entities
        print("\n[Step 3] MLM: Extracting named entities...")
        entities = self._extract_entities(extracted_data['text'])
        print(f"✓ Found {len(entities)} entities")
        
        # Step 4: Sentiment analysis
        print("\n[Step 4] MLM: Analyzing sentiment...")
        sentiment = self._analyze_sentiment(extracted_data['text'])
        print(f"✓ Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
        
        # Step 5: Aggregate results
        print("\n[Step 5] Aggregating results...")
        results = {
            "document": image_path,
            "extracted_text": extracted_data['text'],
            "document_type": extracted_data['type'],
            "summary": summary['summary'],
            "key_points": summary['key_points'],
            "entities": entities,
            "sentiment": sentiment,
            "metadata": {
                "total_characters": len(extracted_data['text']),
                "total_entities": len(entities),
                "processing_steps": 5
            }
        }
        
        print("✓ Processing complete!")
        return results
    
    def _extract_with_vlm(self, image_path: str) -> Dict[str, Any]:
        """Use VLM to extract text and understand document structure."""
        # In production, load actual image
        # For this example, we'll simulate
        
        prompt = """Analyze this document image and:
1. Extract all text content
2. Identify the document type (invoice, contract, report, letter, etc.)
3. Describe the structure and layout

Return your response as JSON with keys: text, type, structure"""
        
        # Simulate VLM response (in production, pass actual image)
        response = self.vlm.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        # In production: {"type": "image_url", "image_url": {"url": image_path}}
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Parse response (simplified for example)
        return {
            "text": "Sample extracted text from document. This is a business report about Q4 2024 financial results. Revenue increased by 25% compared to last year. The company plans to expand operations in Europe and Asia. Key challenges include supply chain issues and increased competition.",
            "type": "Business Report",
            "structure": "Header, body paragraphs, financial data section"
        }
    
    def _summarize_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM to generate summary and extract key points."""
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a document analyst. Create concise summaries and extract key points. Return JSON with 'summary' and 'key_points' array."
                },
                {
                    "role": "user",
                    "content": f"Analyze this document:\n\n{text}"
                }
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Use MLM to extract named entities."""
        # Run NER model
        entities = self.ner_model(text)
        
        # Format results
        formatted = []
        for entity in entities:
            formatted.append({
                "text": entity['word'],
                "type": entity['entity_group'],
                "score": round(entity['score'], 3)
            })
        
        return formatted
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze document sentiment."""
        # Chunk text if too long
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        result = self.sentiment_model(text)[0]
        return {
            "label": result['label'],
            "score": round(result['score'], 3)
        }


def demonstrate_model_communication():
    """
    Demonstrate how models communicate in a real workflow.
    """
    print("\n" + "="*60)
    print("MULTI-MODEL COMMUNICATION DEMONSTRATION")
    print("="*60)
    print("\nThis example shows how different AI models work together:")
    print("• VLM: Extracts text from images")
    print("• LLM: Summarizes and analyzes content")
    print("• MLM: Extracts entities and sentiment")
    print("• All results are combined for complete analysis\n")
    
    # Initialize pipeline
    pipeline = MultiModelPipeline()
    
    # Process a document
    results = pipeline.process_document("sample_document.jpg")
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print(f"\n📄 Document Type: {results['document_type']}")
    
    print(f"\n📝 Summary:")
    print(f"  {results['summary']}")
    
    print(f"\n🔑 Key Points:")
    for i, point in enumerate(results['key_points'], 1):
        print(f"  {i}. {point}")
    
    print(f"\n👤 Named Entities:")
    for entity in results['entities'][:10]:  # Show first 10
        print(f"  • {entity['text']} ({entity['type']}) - confidence: {entity['score']}")
    
    print(f"\n😊 Sentiment: {results['sentiment']['label']} ({results['sentiment']['score']})")
    
    print(f"\n📊 Metadata:")
    print(f"  • Characters: {results['metadata']['total_characters']}")
    print(f"  • Entities Found: {results['metadata']['total_entities']}")
    print(f"  • Processing Steps: {results['metadata']['processing_steps']}")
    
    # Save results to JSON
    output_file = "document_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


def demonstrate_parallel_processing():
    """
    Demonstrate parallel processing where multiple models work simultaneously.
    """
    print("\n" + "="*60)
    print("PARALLEL PROCESSING DEMONSTRATION")
    print("="*60)
    
    import asyncio
    
    async def parallel_analysis(text: str):
        """Run multiple models in parallel."""
        # Simulate parallel processing
        tasks = [
            asyncio.create_task(asyncio.sleep(1)),  # Sentiment analysis
            asyncio.create_task(asyncio.sleep(1)),  # Entity extraction
            asyncio.create_task(asyncio.sleep(1)),  # Summarization
        ]
        
        await asyncio.gather(*tasks)
        
        return {
            "sentiment": "POSITIVE",
            "entities": ["Company A", "John Smith"],
            "summary": "Business performance summary"
        }
    
    print("\n⚡ Running models in parallel...")
    # results = asyncio.run(parallel_analysis("Sample text"))
    print("✓ All models completed simultaneously")


if __name__ == "__main__":
    print("\n" + "🤖 AI TIPS: Multi-Model Communication Examples" + "\n")
    
    # Demonstrate sequential pipeline
    demonstrate_model_communication()
    
    # Demonstrate parallel processing
    demonstrate_parallel_processing()
    
    print("\n" + "="*60)
    print("COMMUNICATION PATTERNS DEMONSTRATED:")
    print("="*60)
    print("✓ Sequential Pipeline: Models process in order")
    print("✓ Data Passing: Output → Input between models")
    print("✓ Result Aggregation: Combining multiple model outputs")
    print("✓ Parallel Processing: Running models simultaneously")
    print("\n💡 These patterns are fundamental to building complex AI systems!")
