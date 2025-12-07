# CLIP - Contrastive Language-Image Pre-training

## Overview

**CLIP (Contrastive Language-Image Pre-training)** is a multimodal model that learns to connect images and text by training on 400 million image-text pairs from the internet. It can understand images in terms of natural language descriptions, enabling zero-shot image classification and powerful image-text matching.

## Key Characteristics

| Property | Value |
|----------|-------|
| **Modalities** | Image + Text |
| **Architecture** | Dual encoder (ViT + Transformer) |
| **Training** | Contrastive learning |
| **Key Ability** | Zero-shot classification |
| **Created By** | OpenAI (2021) |

## How It Works

```
CLIP: Contrastive Learning

1. Encode images and text separately
2. Project to shared embedding space
3. Maximize similarity of matching pairs

       Image                    Text
         │                       │
         ▼                       ▼
    ┌─────────┐            ┌─────────┐
    │ Vision  │            │  Text   │
    │ Encoder │            │ Encoder │
    │ (ViT)   │            │  (GPT)  │
    └────┬────┘            └────┬────┘
         │                       │
         ▼                       ▼
    ┌─────────┐            ┌─────────┐
    │ Project │            │ Project │
    │  Layer  │            │  Layer  │
    └────┬────┘            └────┬────┘
         │                       │
         └───────────┬───────────┘
                     │
              Cosine Similarity
                     │
         ┌───────────┴───────────┐
         │    Contrastive Loss   │
         │  (InfoNCE / NT-Xent)  │
         └───────────────────────┘

Training pairs in batch:
         T1    T2    T3    T4
   I1   ✓     ✗     ✗     ✗
   I2   ✗     ✓     ✗     ✗
   I3   ✗     ✗     ✓     ✗
   I4   ✗     ✗     ✗     ✓

Maximize diagonal, minimize off-diagonal
```

## CLIP Family Models

| Model | Image Encoder | Parameters | Resolution | Best For |
|-------|---------------|------------|------------|----------|
| **CLIP ViT-B/32** | ViT-B | 150M | 224 | Fast inference |
| **CLIP ViT-B/16** | ViT-B | 150M | 224 | Balanced |
| **CLIP ViT-L/14** | ViT-L | 430M | 224 | Better accuracy |
| **CLIP ViT-L/14@336** | ViT-L | 430M | 336 | High resolution |
| **OpenCLIP ViT-G/14** | ViT-G | 1.8B | 224 | State-of-the-art |
| **SigLIP** | ViT | Various | Various | Improved CLIP |
| **EVA-CLIP** | EVA-ViT | 1B+ | 224-336 | Vision tasks |

## Examples with Code

### Example 1: Zero-Shot Image Classification

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image
image = Image.open("cat.jpg")

# Define candidate labels
labels = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

# Process inputs
inputs = processor(
    text=labels,
    images=image,
    return_tensors="pt",
    padding=True
)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

# Print results
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.2%}")
# a photo of a cat: 95.32%
# a photo of a dog: 3.21%
# a photo of a bird: 1.47%
```

### Example 2: Image-Text Similarity

```python
from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Multiple images and texts
images = [Image.open(f"image_{i}.jpg") for i in range(3)]
texts = [
    "A sunset over the ocean",
    "A dog playing in the park",
    "A city skyline at night"
]

# Process all combinations
inputs = processor(
    text=texts,
    images=images,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    # Similarity matrix: images x texts
    similarity = outputs.logits_per_image.softmax(dim=1)

print("Similarity Matrix:")
print(similarity)
# Each row shows which text best matches each image
```

### Example 3: Image Embeddings for Search

```python
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from PIL import Image
import os

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Build image database
def get_image_embeddings(image_paths):
    embeddings = []
    for path in image_paths:
        image = Image.open(path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embeddings.append(embedding.squeeze().numpy())
    return np.array(embeddings)

# Search by text
def search_images(query_text, image_embeddings, image_paths, top_k=5):
    inputs = processor(text=query_text, return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    
    # Compute similarities
    similarities = np.dot(image_embeddings, text_embedding.squeeze().numpy())
    
    # Get top-k
    top_indices = similarities.argsort()[::-1][:top_k]
    
    return [(image_paths[i], similarities[i]) for i in top_indices]

# Usage
image_paths = [f"images/{f}" for f in os.listdir("images")]
image_embeddings = get_image_embeddings(image_paths)

results = search_images("a red sports car", image_embeddings, image_paths)
for path, score in results:
    print(f"{path}: {score:.3f}")
```

### Example 4: Custom Classification with Prompts

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def classify_with_prompts(image, categories, prompt_template="a photo of a {}"):
    """Classify image into categories using CLIP"""
    
    # Create text prompts for each category
    prompts = [prompt_template.format(cat) for cat in categories]
    
    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
    
    # Return sorted predictions
    results = list(zip(categories, probs[0].tolist()))
    return sorted(results, key=lambda x: x[1], reverse=True)

# Classify animal
image = Image.open("animal.jpg")
categories = ["cat", "dog", "bird", "fish", "rabbit", "hamster"]
predictions = classify_with_prompts(image, categories)

for category, prob in predictions[:3]:
    print(f"{category}: {prob:.2%}")
```

### Example 5: Prompt Engineering for Better Results

```python
# Different prompt templates can improve accuracy

def ensemble_classify(image, categories, prompt_templates):
    """Ensemble over multiple prompt templates"""
    
    all_probs = []
    
    for template in prompt_templates:
        prompts = [template.format(cat) for cat in categories]
        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        all_probs.append(probs[0])
    
    # Average probabilities
    avg_probs = torch.stack(all_probs).mean(dim=0)
    
    return list(zip(categories, avg_probs.tolist()))

# Use multiple templates
templates = [
    "a photo of a {}",
    "a picture of a {}",
    "an image of a {}",
    "a {} in the scene",
    "a professional photo of a {}",
]

# This often improves accuracy by 2-5%
predictions = ensemble_classify(image, categories, templates)
```

### Example 6: CLIP for Content Moderation

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def moderate_image(image, threshold=0.5):
    """Check if image contains inappropriate content"""
    
    categories = {
        "safe": [
            "a safe family-friendly image",
            "appropriate content for all ages",
            "a normal photograph"
        ],
        "unsafe": [
            "inappropriate content",
            "violent or graphic imagery",
            "explicit material"
        ]
    }
    
    all_texts = categories["safe"] + categories["unsafe"]
    
    inputs = processor(text=all_texts, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    
    safe_score = probs[:len(categories["safe"])].mean().item()
    unsafe_score = probs[len(categories["safe"]):].mean().item()
    
    is_safe = safe_score > unsafe_score and unsafe_score < threshold
    
    return {
        "is_safe": is_safe,
        "safe_score": safe_score,
        "unsafe_score": unsafe_score
    }

# Check image
result = moderate_image(Image.open("image.jpg"))
print(f"Safe: {result['is_safe']}, Score: {result['safe_score']:.2%}")
```

### Example 7: OpenCLIP (Open-Source Alternative)

```python
import open_clip
import torch
from PIL import Image

# Load OpenCLIP model (many more variants available)
model, preprocess, _ = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='laion2b_s32b_b82k'
)
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# Process image
image = preprocess(Image.open("image.jpg")).unsqueeze(0)

# Process text
texts = ["a dog", "a cat", "a bird"]
text_tokens = tokenizer(texts)

# Get embeddings
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)
    
    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

for text, prob in zip(texts, similarity[0]):
    print(f"{text}: {prob:.2%}")
```

## Use Cases

| Application | Description | Example |
|-------------|-------------|---------|
| **Zero-shot Classification** | Classify without training | Any image category |
| **Image Search** | Find images by text query | "sunset at beach" |
| **Content Moderation** | Detect inappropriate content | Filter uploads |
| **Image Captioning** | Score caption candidates | Rank descriptions |
| **Face Similarity** | Match faces to descriptions | "person with glasses" |
| **Fashion Search** | Find similar clothing | "red summer dress" |
| **Medical Imaging** | Zero-shot diagnosis | X-ray analysis |
| **Document Understanding** | Image-text matching | Form processing |

## CLIP vs Other Models

| Feature | CLIP | CNN + Labels | VLM |
|---------|------|--------------|-----|
| **Zero-shot** | ✅ Yes | ❌ No | ✅ Yes |
| **Training data** | 400M pairs | Labeled images | Multimodal |
| **Flexibility** | ✅ Any category | Fixed classes | ✅ Conversational |
| **Speed** | Fast | Fast | Slow |
| **Understanding** | Matching | Classification | Reasoning |

## Benchmarks

### Zero-Shot ImageNet Classification

| Model | Top-1 Accuracy | Parameters |
|-------|----------------|------------|
| CLIP ViT-B/32 | 63.4% | 150M |
| CLIP ViT-B/16 | 68.3% | 150M |
| CLIP ViT-L/14 | 75.5% | 430M |
| OpenCLIP ViT-G/14 | 80.1% | 1.8B |
| EVA-CLIP | 82.0% | 1B |
| SigLIP | 83.1% | 878M |

### Linear Probe (Fine-tuned)

| Model | ImageNet | Food101 | Pets |
|-------|----------|---------|------|
| CLIP ViT-B/32 | 72.8% | 90.5% | 91.3% |
| CLIP ViT-L/14 | 85.4% | 94.8% | 95.2% |

## Limitations

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Abstract concepts** | Struggles with abstract ideas | Better prompts |
| **Counting** | Poor at counting objects | Specialized models |
| **Spatial relations** | "left of" difficult | Use VLMs |
| **Fine-grained** | Similar categories | Domain fine-tuning |
| **OCR** | Limited text reading | OCR models |
| **Bias** | Inherits training biases | Careful evaluation |

## Related Models

| Model | Relation | Key Difference |
|-------|----------|----------------|
| **ALIGN** | Google's CLIP | Larger dataset |
| **BLIP/BLIP-2** | Extends CLIP | Generation capability |
| **SigLIP** | Improved CLIP | Sigmoid loss |
| **CoCa** | Unified model | Decoder added |
| **LLaVA** | VLM with CLIP | LLM integration |
| **Stable Diffusion** | Uses CLIP | Text-to-image |

## CLIP in Production

```python
# Efficient batch processing
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import DataLoader
import numpy as np

class ImageDataset:
    def __init__(self, image_paths, processor):
        self.paths = image_paths
        self.processor = processor
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        return self.processor(images=image, return_tensors="pt")['pixel_values'][0]

# Batch embedding
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

dataset = ImageDataset(image_paths, processor)
loader = DataLoader(dataset, batch_size=32, num_workers=4)

all_embeddings = []
with torch.no_grad():
    for batch in loader:
        embeddings = model.get_image_features(pixel_values=batch)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        all_embeddings.append(embeddings.cpu().numpy())

embeddings = np.concatenate(all_embeddings, axis=0)
```

## Resources

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP GitHub](https://github.com/openai/CLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Hugging Face CLIP](https://huggingface.co/docs/transformers/model_doc/clip)
- [CLIP Interrogator](https://github.com/pharmapsychotic/clip-interrogator)

## Related Docs

- **[VLM](./VLM.md)** - Vision-language models built on CLIP
- **[VIT](./VIT.md)** - Vision Transformer (CLIP's image encoder)
- **[DIFFUSION](./DIFFUSION.md)** - CLIP guides image generation
- **[LMM](./LMM.md)** - Multimodal models
