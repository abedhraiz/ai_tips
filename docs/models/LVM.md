# LVM - Large Vision Model

## Overview

Large Vision Models (LVMs) are specialized computer vision architectures focused exclusively on visual data processing without inherent language generation capabilities. These models excel at feature extraction, image classification, object detection, and visual representation learning.

## Key Characteristics

- **Architecture**: Vision Transformers (ViT), CNNs, or hybrid architectures
- **Parameter Range**: 300M to 22B parameters
- **Training**: Self-supervised learning (DINO, MAE) and supervised classification
- **Output**: Feature vectors, classification logits, visual embeddings

## Core Capabilities

1. **Image Classification**: Multi-class categorization at scale
2. **Feature Extraction**: Dense visual embeddings for downstream tasks
3. **Object Detection**: Bounding box prediction and localization
4. **Visual Similarity**: Embedding-based image retrieval
5. **Transfer Learning**: Pre-trained representations for specialized tasks

## Examples

### Example 1: Image Classification with DINOv2

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# Load DINOv2 model
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

# Load and process image
image = Image.open('cat.jpg')
inputs = processor(images=image, return_tensors="pt")

# Get visual features
with torch.no_grad():
    outputs = model(**inputs)
    features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling

print(f"Feature vector shape: {features.shape}")  # [1, 768]
```

**Input**: Cat image  
**Output**: 768-dimensional feature vector representing the image

### Example 2: Visual Similarity Search

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

def get_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :]
    
    return embedding.numpy()

# Get embeddings for multiple images
query_embedding = get_image_embedding('query.jpg')
image1_embedding = get_image_embedding('image1.jpg')
image2_embedding = get_image_embedding('image2.jpg')

# Compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity1 = cosine_similarity(query_embedding, image1_embedding)
similarity2 = cosine_similarity(query_embedding, image2_embedding)

print(f"Similarity to image1: {similarity1[0][0]:.4f}")
print(f"Similarity to image2: {similarity2[0][0]:.4f}")
```

**Input**: Query image + database of images  
**Output**: Similarity scores for finding visually similar images

### Example 3: Transfer Learning for Custom Classification

```python
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load pre-trained LVM as feature extractor
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
backbone = AutoModel.from_pretrained('facebook/dinov2-base')

# Freeze backbone
for param in backbone.parameters():
    param.requires_grad = False

# Add custom classification head
class CustomClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, pixel_values):
        with torch.no_grad():
            features = self.backbone(pixel_values=pixel_values).last_hidden_state
            pooled = features.mean(dim=1)  # Global average pooling
        
        logits = self.classifier(pooled)
        return logits

# Create model for 10-class classification
model = CustomClassifier(backbone, num_classes=10)

# Training loop (simplified)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Example training step
# for batch in train_loader:
#     optimizer.zero_grad()
#     outputs = model(batch['pixel_values'])
#     loss = criterion(outputs, batch['labels'])
#     loss.backward()
#     optimizer.step()

print("Model ready for fine-tuning on custom dataset")
```

**Input**: Custom image dataset  
**Output**: Fine-tuned model for specific classification task

### Example 4: Object Detection with Features

```python
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw

# Load DETR model (detection transformer)
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Load image
image = Image.open("street.jpg")
inputs = processor(images=image, return_tensors="pt")

# Detect objects
with torch.no_grad():
    outputs = model(**inputs)

# Post-process
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]

# Draw bounding boxes
draw = ImageDraw.Draw(image)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    label_name = model.config.id2label[label.item()]
    
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), f"{label_name}: {round(score.item(), 3)}", fill="red")
    
    print(f"Detected {label_name} with confidence {round(score.item(), 3)} at {box}")

image.save("detected.jpg")
```

**Input**: Street scene image  
**Output**: Detected objects with bounding boxes and confidence scores

### Example 5: Image Retrieval System

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import faiss
import numpy as np
import os

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

def extract_features(image_path):
    """Extract features from an image"""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return features.flatten()

# Build index from image database
image_dir = 'image_database/'
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

print("Extracting features from database...")
features_list = []
for img_path in image_paths:
    features = extract_features(img_path)
    features_list.append(features)

# Create FAISS index
features_array = np.array(features_list).astype('float32')
dimension = features_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(features_array)

print(f"Indexed {len(image_paths)} images")

# Search for similar images
query_image = 'query.jpg'
query_features = extract_features(query_image).astype('float32').reshape(1, -1)

k = 5  # Top 5 similar images
distances, indices = index.search(query_features, k)

print(f"\nTop {k} similar images to {query_image}:")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    print(f"{i+1}. {image_paths[idx]} (distance: {dist:.4f})")
```

**Input**: Query image + image database  
**Output**: Top-K most similar images with distances

## LVM vs VLM

| Aspect | LVM | VLM |
|--------|-----|-----|
| **Output** | Feature vectors, classifications | Natural language descriptions |
| **Language Understanding** | None | Yes |
| **Use Case** | Feature extraction, classification | Image captioning, VQA |
| **Architecture** | Vision-only | Vision + Language |
| **Examples** | DINOv2, ResNet, ViT | GPT-4V, LLaVA |

## Popular LVM Models

1. **DINOv2** (Meta)
   - Self-supervised vision transformer
   - 300M to 1B parameters
   - Excellent for feature extraction

2. **CLIP Vision Encoder** (OpenAI)
   - Part of CLIP but can be used standalone
   - 400M parameters (ViT-L/14)
   - Trained on image-text pairs

3. **Vision Transformer (ViT)** (Google)
   - Pure transformer for images
   - 86M to 632M parameters
   - Strong classification performance

4. **SigLIP** (Google)
   - Improved CLIP-style model
   - Better image-text alignment
   - Efficient training

5. **EfficientNet** (Google)
   - Efficient CNN architecture
   - 5M to 66M parameters
   - Optimized for mobile/edge

## Enterprise Applications

- **E-commerce**: Product image search and recommendation
- **Medical Imaging**: Disease detection, organ segmentation
- **Manufacturing**: Quality control, defect detection
- **Security**: Face recognition, anomaly detection
- **Agriculture**: Crop health monitoring, pest identification
- **Retail**: Inventory management, shelf monitoring

## Best Practices

1. **Choose the Right Model**
   - DINOv2 for general-purpose features
   - CLIP for multimodal retrieval
   - EfficientNet for edge deployment

2. **Feature Extraction**
   - Use CLS token or global pooling
   - Normalize features for similarity search
   - Consider dimensionality reduction for large-scale

3. **Transfer Learning**
   - Freeze backbone, train only head
   - Fine-tune last layers if needed
   - Use appropriate learning rates

4. **Deployment**
   - ONNX export for production
   - Quantization for edge devices
   - Batch processing for throughput

## Code Resources

- **Hugging Face Transformers**: https://huggingface.co/models?pipeline_tag=image-classification
- **TorchVision**: https://pytorch.org/vision/stable/models.html
- **Timm (PyTorch Image Models)**: https://github.com/huggingface/pytorch-image-models

---

**Related Models**: [VLM](./VLM.md) | [SAM](./SAM.md) | [Vision Transformers](./VIT.md)
