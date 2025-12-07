# ViT - Vision Transformer

## Overview

**Vision Transformers (ViT)** apply the transformer architecture, originally designed for NLP, directly to images. By treating an image as a sequence of patches (like words in a sentence), ViTs have achieved state-of-the-art results on image classification and serve as the backbone for many modern vision models.

## Key Characteristics

| Property | Value |
|----------|-------|
| **Type** | Image classification/feature extraction |
| **Architecture** | Pure Transformer encoder |
| **Input** | Image patches (16x16 or 14x14) |
| **Training** | Supervised or self-supervised |
| **Pretraining** | ImageNet, JFT-300M, etc. |
| **Output** | Class logits or embeddings |

## How It Works

```
Image (224×224×3)
       ↓
Split into patches (14×14 patches of 16×16 pixels)
       ↓
Flatten + Linear projection → Patch embeddings (196 × 768)
       ↓
Add position embeddings + [CLS] token
       ↓
┌──────────────────────────────────────┐
│     Transformer Encoder Blocks       │
│  ┌─────────────────────────────────┐ │
│  │  Multi-Head Self-Attention      │ │
│  ├─────────────────────────────────┤ │ × 12 (or more)
│  │  Feed-Forward Network           │ │
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
       ↓
[CLS] token embedding → Classification head → Predictions
```

## Patch Embedding Visualization

```
Original Image (224×224)          Patches (14×14 grid)
┌─────────────────────┐          ┌──┬──┬──┬──┬──┬──┐
│                     │          │1 │2 │3 │4 │..│14│
│      🖼️ Image       │    →     ├──┼──┼──┼──┼──┼──┤
│                     │          │15│16│17│18│..│28│
│                     │          ├──┼──┼──┼──┼──┼──┤
│                     │          │..│..│..│..│..│..│
└─────────────────────┘          └──┴──┴──┴──┴──┴──┘
                                 
Each patch (16×16×3) → Flattened (768) → Embedding (768)
Total: 196 patch tokens + 1 [CLS] token = 197 tokens
```

## Popular Models

### Original ViT Family

| Model | Params | Layers | Hidden | Heads | ImageNet Acc |
|-------|--------|--------|--------|-------|--------------|
| **ViT-Ti** | 5.7M | 12 | 192 | 3 | 72.2% |
| **ViT-S** | 22M | 12 | 384 | 6 | 79.8% |
| **ViT-B/16** | 86M | 12 | 768 | 12 | 84.5% |
| **ViT-L/16** | 307M | 24 | 1024 | 16 | 87.1% |
| **ViT-H/14** | 632M | 32 | 1280 | 16 | 88.6% |

### ViT Variants

| Model | Key Innovation | Use Case |
|-------|---------------|----------|
| **DeiT** | Knowledge distillation | Efficient training |
| **Swin Transformer** | Shifted windows | Hierarchical features |
| **BEiT** | BERT-style pretraining | Self-supervised |
| **MAE** | Masked autoencoding | Self-supervised |
| **DINO/DINOv2** | Self-distillation | Strong features |
| **EVA** | Scaling + CLIP | Large-scale |
| **SigLIP** | Sigmoid loss | Efficient CLIP |

### Model Comparison

| Model | Params | ImageNet-1K | Notes |
|-------|--------|-------------|-------|
| ViT-B/16 | 86M | 84.5% | Baseline |
| DeiT-B | 86M | 83.4% | No extra data |
| Swin-B | 88M | 85.2% | Hierarchical |
| BEiT-B | 86M | 85.2% | Self-supervised |
| MAE-B | 86M | 83.6% | Masked training |
| DINOv2-B | 86M | 86.3% | Excellent features |
| EVA-02-B | 86M | 88.5% | CLIP+MAE |

## Examples with Code

### Example 1: Image Classification

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load model and processor
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Load and process image
image = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print(f"Predicted: {model.config.id2label[predicted_class]}")
```

### Example 2: Feature Extraction

```python
from transformers import ViTModel, ViTImageProcessor
import torch

# Load for feature extraction (no classification head)
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    
    # [CLS] token embedding - global image representation
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)
    
    # All patch embeddings
    patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Shape: (1, 196, 768)
    
    # Pooled output
    pooled = outputs.pooler_output  # Shape: (1, 768)
```

### Example 3: DINOv2 (Best Feature Extractor)

```python
import torch
from transformers import AutoModel, AutoImageProcessor

# DINOv2 produces excellent general-purpose features
model = AutoModel.from_pretrained('facebook/dinov2-base')
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0]  # CLS token

# Use for:
# - Image similarity
# - Clustering
# - Few-shot classification
# - Retrieval
```

### Example 4: Swin Transformer

```python
from transformers import SwinForImageClassification, AutoImageProcessor

# Swin has hierarchical features (better for detection/segmentation)
model = SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224')
processor = AutoImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224')

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()
```

### Example 5: MAE (Masked Autoencoder)

```python
from transformers import ViTMAEForPreTraining, ViTImageProcessor
import torch

# MAE is trained by masking 75% of patches and reconstructing
model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
processor = ViTImageProcessor.from_pretrained('facebook/vit-mae-base')

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    
    # Reconstruction loss
    loss = outputs.loss
    
    # Reconstructed patches
    reconstructed = outputs.logits
```

### Example 6: Fine-Tuning ViT

```python
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("food101", split="train[:1000]")

# Load model for fine-tuning
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=101,
    ignore_mismatched_sizes=True
)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Preprocess
def transform(examples):
    examples["pixel_values"] = [
        processor(images=img, return_tensors="pt")["pixel_values"][0]
        for img in examples["image"]
    ]
    return examples

dataset = dataset.with_transform(transform)

# Train
training_args = TrainingArguments(
    output_dir="./vit-food101",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

### Example 7: Image Similarity with CLIP ViT

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Compare images
image1 = Image.open("cat1.jpg")
image2 = Image.open("cat2.jpg")
image3 = Image.open("dog.jpg")

inputs = processor(images=[image1, image2, image3], return_tensors="pt")

with torch.no_grad():
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Similarity matrix
    similarity = image_features @ image_features.T
    print(similarity)
    # [[1.0, 0.85, 0.32],   # cat1 similar to cat2
    #  [0.85, 1.0, 0.35],   # cat2 similar to cat1
    #  [0.32, 0.35, 1.0]]   # dog different from cats
```

## ViT vs CNN Comparison

| Aspect | ViT | CNN (ResNet) |
|--------|-----|--------------|
| **Inductive Bias** | Minimal (learns from data) | Strong (locality, translation) |
| **Data Efficiency** | Needs more data | Works with less data |
| **Scalability** | ✅ Excellent | Limited |
| **Global Context** | ✅ From first layer | Only in deep layers |
| **Speed** | Slower (attention O(n²)) | ✅ Faster |
| **Transfer Learning** | ✅ Excellent | Good |
| **Interpretability** | ✅ Attention maps | Harder |

## Attention Visualization

```python
from transformers import ViTModel, ViTImageProcessor
import matplotlib.pyplot as plt

model = ViTModel.from_pretrained('google/vit-base-patch16-224', output_attentions=True)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    
# Get attention from last layer, first head
attention = outputs.attentions[-1][0, 0]  # (197, 197)

# CLS token attention to patches (what CLS looks at)
cls_attention = attention[0, 1:]  # (196,)
cls_attention = cls_attention.reshape(14, 14)  # Reshape to patch grid

# Visualize
plt.imshow(cls_attention, cmap='hot')
plt.title("ViT Attention Map")
plt.show()
```

## Use Cases

| Use Case | Best Model | Notes |
|----------|-----------|-------|
| **Classification** | EVA, SigLIP | Highest accuracy |
| **Feature Extraction** | DINOv2 | Best general features |
| **Object Detection** | Swin, ViT-Det | Hierarchical features |
| **Segmentation** | Swin, SAM (ViT) | Dense prediction |
| **Image Retrieval** | CLIP, DINOv2 | Semantic search |
| **Medical Imaging** | BEiT, MAE | Self-supervised helps |
| **Multimodal** | CLIP, SigLIP | Vision-language |

## Memory Requirements

| Model | Parameters | VRAM (FP32) | VRAM (FP16) |
|-------|-----------|-------------|-------------|
| ViT-Ti | 5.7M | 0.2 GB | 0.1 GB |
| ViT-S | 22M | 0.5 GB | 0.25 GB |
| ViT-B | 86M | 1.5 GB | 0.75 GB |
| ViT-L | 307M | 4 GB | 2 GB |
| ViT-H | 632M | 8 GB | 4 GB |
| ViT-G | 1.8B | 24 GB | 12 GB |

## Choosing the Right ViT

```
What's your task?
├── Image Classification
│   ├── Best accuracy → EVA-02, SigLIP
│   ├── Good balance → ViT-L, DeiT
│   └── Fast/small → ViT-S, DeiT-S
│
├── Feature Extraction
│   ├── General purpose → DINOv2
│   ├── With text → CLIP, SigLIP
│   └── Self-supervised → MAE, BEiT
│
├── Detection/Segmentation
│   ├── General → Swin Transformer
│   └── Segmentation → SAM (ViT backbone)
│
└── Limited Data
    ├── Use self-supervised → MAE, DINO
    └── Use knowledge distillation → DeiT
```

## Training Tips

1. **Data augmentation is crucial** - RandAugment, Mixup, CutMix
2. **Use pretrained weights** - Training from scratch needs huge data
3. **Warmup learning rate** - ViTs benefit from longer warmup
4. **Layer decay** - Different LR for different layers when fine-tuning
5. **Regularization** - Stochastic depth, dropout, weight decay

## ViT in Modern Architectures

```
SAM (Segmentation):     Image → [ViT-H Encoder] → Features → Mask Decoder

CLIP (Vision-Language): Image → [ViT] → Image Embed ↔ Text Embed ← [Text Encoder] ← Text

Stable Diffusion 3:     Image → [ViT/DiT] → Denoising

LLaVA (VLM):           Image → [ViT/CLIP] → [Projection] → [LLM] → Response
```

## Related Models

- **[VLM](./VLM.md)** - Uses ViT as vision encoder
- **[SAM](./SAM.md)** - Uses ViT for segmentation
- **[LVM](./LVM.md)** - ViT variants for pure vision
- **[Diffusion](./DIFFUSION.md)** - DiT uses transformer for diffusion

## Resources

- [Original ViT Paper](https://arxiv.org/abs/2010.11929)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [Hugging Face ViT](https://huggingface.co/docs/transformers/model_doc/vit)
- [Timm Library](https://github.com/huggingface/pytorch-image-models)
