# LMM - Large Multimodal Model

## Overview

Large Multimodal Models are advanced AI systems designed to natively process, understand, and generate content across multiple modalities including text, images, audio, and video. Unlike retrofitted multimodal models, LMMs are built from the ground up with unified architectures that enable seamless cross-modal reasoning.

## Key Characteristics

- **Architecture**: Unified transformer with multimodal tokenization
- **Parameter Range**: 10B to 1.7T+ parameters
- **Modalities**: Text, images, audio, video, 3D (model-dependent)
- **Output**: Any modality (model-dependent), primarily text
- **Training**: Large-scale multimodal pre-training

## Core Capabilities

1. **Omni-Modal Understanding**: Simultaneous processing of multiple input types
2. **Cross-Modal Generation**: Output in different modality than input
3. **Temporal Reasoning**: Understanding sequences across time (video, audio)
4. **Multimodal Retrieval**: Search and match across different content types
5. **Complex Task Orchestration**: Multi-step, multi-modal workflows

## Examples

### Example 1: Image + Audio + Text Understanding (Gemini-Style)

```python
import google.generativeai as genai
from PIL import Image
import os

# Configure API
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Use Gemini Pro (multimodal)
model = genai.GenerativeModel('gemini-pro-vision')

# Load multimodal inputs
image = Image.open('concert.jpg')

# Combined query with image and text
prompt = """
Analyze this concert image and answer:
1. What instruments can you see?
2. What's the lighting setup?
3. Estimate the crowd size
4. What genre of music might this be?
"""

response = model.generate_content([prompt, image])
print(response.text)
```

**Input**: Concert image + detailed questions  
**Output**: Comprehensive analysis across multiple aspects

### Example 2: Video Understanding

```python
import anthropic
import base64
import os

client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

# For video, extract key frames
def extract_frames(video_path, num_frames=10):
    """Extract frames from video"""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

# Process video with multimodal model
frames = extract_frames('recipe_video.mp4', num_frames=8)

# Convert frames to base64
frame_data = []
for frame in frames:
    _, buffer = cv2.imencode('.jpg', frame)
    b64 = base64.b64encode(buffer).decode('utf-8')
    frame_data.append(b64)

# Create message with multiple frames
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this cooking video and provide:\n1. List of ingredients\n2. Step-by-step recipe\n3. Estimated cooking time\n4. Difficulty level"}
        ] + [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame
                }
            } for frame in frame_data
        ]
    }]
)

print(message.content[0].text)
```

**Input**: Recipe video (multiple frames)  
**Output**: Complete recipe extraction with steps

### Example 3: Cross-Modal Reasoning

```python
import openai
import os

client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Multimodal input: diagram + text question
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "I'm learning physics. Can you:\n1. Explain what this diagram shows\n2. Write the relevant equations\n3. Provide a step-by-step solution\n4. Suggest a similar practice problem"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/physics_problem.png"
                    }
                }
            ]
        }
    ],
    max_tokens=1000
)

print(response.choices[0].message.content)
```

**Input**: Physics diagram + learning request  
**Output**: Explanation, equations, solution, and practice problem

### Example 4: Document + Chart Analysis

```python
import anthropic
import base64
import os

client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

# Load financial report with charts
with open("financial_report.pdf", "rb") as f:
    # Extract pages as images (using pdf2image or similar)
    from pdf2image import convert_from_path
    pages = convert_from_path("financial_report.pdf")

# Analyze multiple pages
analysis_prompt = """
Analyze this financial report and provide:

1. **Key Financial Metrics**: Extract revenue, profit, expenses
2. **Chart Interpretation**: Explain trends shown in graphs
3. **Year-over-Year Comparison**: Compare with previous periods
4. **Risk Assessment**: Identify concerning trends
5. **Executive Summary**: 3-4 sentence overview

Be specific with numbers and percentages.
"""

# Send multiple pages
image_content = []
for page in pages[:5]:  # First 5 pages
    buffer = io.BytesIO()
    page.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    image_content.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64
        }
    })

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=2000,
    messages=[{
        "role": "user",
        "content": [{"type": "text", "text": analysis_prompt}] + image_content
    }]
)

print(message.content[0].text)
```

**Input**: Multi-page financial report with charts  
**Output**: Comprehensive financial analysis

### Example 5: Multimodal Conversation

```python
import openai
import os

client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Multi-turn conversation with mixed modalities
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "I'm renovating my living room. Here's the current state:"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/living_room_before.jpg"}
            }
        ]
    }
]

response1 = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=conversation,
    max_tokens=500
)

print("AI:", response1.choices[0].message.content)

# Continue conversation
conversation.append({"role": "assistant", "content": response1.choices[0].message.content})
conversation.append({
    "role": "user",
    "content": [
        {"type": "text", "text": "Here's my inspiration image. Can you suggest how to transform my room to look like this?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/inspiration.jpg"}
        }
    ]
})

response2 = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=conversation,
    max_tokens=800
)

print("\nAI:", response2.choices[0].message.content)
```

**Input**: Multiple images in conversational context  
**Output**: Contextual design recommendations

## LMM vs MLLM

| Aspect | LMM | MLLM |
|--------|-----|------|
| **Design** | Native multimodal | LLM with adapters |
| **Architecture** | Unified from start | Retrofitted |
| **Cross-Modal** | Deep integration | Adapter-based |
| **Output** | Any modality possible | Primarily text |
| **Examples** | Gemini Ultra, GPT-4o | LLaVA, Qwen-VL |

## Popular LMM Models

1. **Google Gemini Ultra/Pro**
   - Natively multimodal from ground up
   - Text, image, audio, video support
   - 1M+ token context window

2. **GPT-4o** (OpenAI)
   - Omni-modal model
   - Unified audio, vision, text
   - Real-time multimodal interaction

3. **Meta ImageBind**
   - 6-modality embedding space
   - Image, text, audio, depth, thermal, IMU
   - Research model for unified embeddings

4. **NExT-GPT**
   - Any-to-any generation
   - Text, image, audio, video
   - Open-source multimodal model

## Enterprise Applications

- **Healthcare**: Multimodal patient data analysis (images + records + audio notes)
- **Education**: Interactive learning with multiple content formats
- **Media & Entertainment**: Content analysis, automated editing, subtitle generation
- **Automotive**: Sensor fusion for autonomous driving
- **Accessibility**: Cross-modal translation for differently-abled users
- **Scientific Research**: Multi-instrument data integration

## Best Practices

1. **Input Preparation**
   - Optimize image/video quality
   - Provide clear textual context
   - Consider token limits for large inputs

2. **Prompt Engineering**
   - Be specific about what aspects to analyze
   - Structure complex queries with numbers
   - Request specific output formats

3. **Cost Management**
   - Multimodal tokens are more expensive
   - Resize images appropriately
   - Use sampling for long videos

4. **Error Handling**
   - Implement retries for API failures
   - Validate multimodal inputs before sending
   - Handle rate limits gracefully

## Code Resources

- **Google Gemini API**: https://ai.google.dev/docs
- **OpenAI Vision API**: https://platform.openai.com/docs/guides/vision
- **Anthropic Claude Vision**: https://docs.anthropic.com/claude/docs/vision
- **ImageBind**: https://github.com/facebookresearch/ImageBind

---

**Related Models**: [MLLM](./MLLM.md) | [VLM](./VLM.md) | [Foundation Models](./MULTIMODAL_FOUNDATION.md)
