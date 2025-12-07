# SAM - Segment Anything Model

## Overview

**Segment Anything Model (SAM)** is a foundational model for image segmentation that can segment any object in any image with zero-shot generalization. Given a prompt (point, box, or text), SAM produces high-quality segmentation masks.

## Key Characteristics

| Property | Value |
|----------|-------|
| **Developer** | Meta AI |
| **Type** | Vision Foundation Model |
| **Task** | Universal image segmentation |
| **Architecture** | ViT encoder + prompt decoder |
| **Training Data** | SA-1B (1 billion masks) |
| **Zero-shot** | Yes - works on any image |

## How It Works

```
                    SAM Architecture
    
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Image ─────→ [Image Encoder] ─────→ Image Embedding
    │               (ViT-H/L/B)              (256×64×64)  
    │                                             │       
    │                                             ↓       
    │  Prompt ────→ [Prompt Encoder] ──→  ┌─────────────┐
    │  (point/box/mask/text)               │   Mask      │
    │                                      │   Decoder   │
    │                                      └──────┬──────┘
    │                                             │       
    │                                             ↓       
    │                                    Segmentation Masks
    │                                    + Confidence Scores
    └─────────────────────────────────────────────────────┘
```

## SAM Versions

| Version | Release | Key Improvements |
|---------|---------|------------------|
| **SAM** | Apr 2023 | Original model |
| **SAM-HQ** | Aug 2023 | Higher quality masks |
| **FastSAM** | Jun 2023 | 50x faster (YOLO-based) |
| **MobileSAM** | Jun 2023 | Mobile-optimized |
| **EfficientSAM** | Dec 2023 | Efficient distillation |
| **SAM 2** | Jul 2024 | Video + improved quality |

## Model Sizes

| Model | Encoder | Parameters | Speed | Quality |
|-------|---------|-----------|-------|---------|
| **SAM ViT-B** | ViT-Base | 91M | Fast | Good |
| **SAM ViT-L** | ViT-Large | 308M | Medium | Better |
| **SAM ViT-H** | ViT-Huge | 636M | Slow | Best |
| **MobileSAM** | TinyViT | 10M | Very Fast | Good |
| **FastSAM** | YOLOv8 | 68M | Very Fast | Good |
| **SAM 2 Large** | Hiera-L | 224M | Medium | Excellent |

## Prompt Types

```
1. Point Prompts:        2. Box Prompts:         3. Mask Prompts:
   Click on object          Draw bounding box        Provide rough mask
   
   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
   │      •      │          │ ┌─────────┐ │          │  ░░░░░░░░░  │
   │             │          │ │         │ │          │  ░░░░░░░░░  │
   │             │          │ │         │ │          │  ░░░░░░░░░  │
   │             │          │ └─────────┘ │          │             │
   └─────────────┘          └─────────────┘          └─────────────┘

4. Text Prompts (with extensions):
   "cat", "person", "building" → Segment matching objects
```

## Examples with Code

### Example 1: Basic Segmentation with Point Prompt

```python
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

# Load model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
sam.to("cuda")
predictor = SamPredictor(sam)

# Load image
image = cv2.imread("photo.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# Point prompt (x, y coordinates)
input_point = np.array([[500, 375]])  # Click location
input_label = np.array([1])  # 1 = foreground, 0 = background

# Predict mask
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True  # Returns 3 masks
)

# Best mask
best_mask = masks[np.argmax(scores)]
```

### Example 2: Box Prompt

```python
# Bounding box prompt [x1, y1, x2, y2]
input_box = np.array([100, 100, 400, 400])

masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False
)
```

### Example 3: Multiple Points

```python
# Multiple points for better precision
input_points = np.array([
    [500, 375],   # Foreground point 1
    [550, 400],   # Foreground point 2
    [200, 100],   # Background point (things to exclude)
])
input_labels = np.array([1, 1, 0])  # 1=foreground, 0=background

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True
)
```

### Example 4: Automatic Mask Generation

```python
from segment_anything import SamAutomaticMaskGenerator

# Generate all masks in image
mask_generator = SamAutomaticMaskGenerator(sam)

# This finds and segments EVERYTHING
masks = mask_generator.generate(image)

# Each mask contains:
# - 'segmentation': binary mask
# - 'area': mask area in pixels
# - 'bbox': bounding box
# - 'predicted_iou': quality score
# - 'stability_score': mask stability

# Sort by area (largest first)
masks = sorted(masks, key=lambda x: x['area'], reverse=True)

# Visualize
for mask in masks[:10]:  # Top 10 objects
    segmentation = mask['segmentation']
    # Overlay on image...
```

### Example 5: SAM with Grounding DINO (Text Prompts)

```python
from groundingdino.util.inference import load_model, predict
from segment_anything import sam_model_registry, SamPredictor
import cv2

# Load Grounding DINO for text-to-box
grounding_model = load_model("groundingdino_config.py", "groundingdino_weights.pth")

# Detect objects from text
text_prompt = "cat . dog . person"
boxes, logits, phrases = predict(
    model=grounding_model,
    image=image,
    caption=text_prompt,
    box_threshold=0.35,
    text_threshold=0.25
)

# Use boxes as SAM prompts
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)

for box, phrase in zip(boxes, phrases):
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    print(f"Segmented: {phrase}")
```

### Example 6: SAM 2 for Video

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

# Load SAM 2
predictor = build_sam2_video_predictor("sam2_hiera_l.yaml", "sam2_hiera_large.pt")

# Initialize with video
with torch.inference_mode():
    state = predictor.init_state(video_path="video.mp4")
    
    # Add prompt on first frame
    _, _, masks = predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=1,
        points=[[500, 375]],
        labels=[1]
    )
    
    # Propagate through video
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        # masks contains segmentation for each frame
        print(f"Frame {frame_idx}: {len(obj_ids)} objects tracked")
```

### Example 7: FastSAM (Quick Alternative)

```python
from fastsam import FastSAM, FastSAMPrompt

# Much faster than original SAM
model = FastSAM("FastSAM-x.pt")

# Run on image
results = model(
    image,
    device="cuda",
    retina_masks=True,
    conf=0.4,
    iou=0.9
)

# Process with prompts
prompt_process = FastSAMPrompt(image, results, device="cuda")

# Point prompt
masks = prompt_process.point_prompt(points=[[500, 375]], pointlabel=[1])

# Box prompt
masks = prompt_process.box_prompt(bbox=[100, 100, 400, 400])

# Text prompt
masks = prompt_process.text_prompt(text="cat")
```

## Use Cases

| Use Case | Method | Notes |
|----------|--------|-------|
| **Object Removal** | SAM + Inpainting | Segment, then fill |
| **Background Removal** | Point/box prompt | Product photos |
| **Image Editing** | SAM + Diffusion | Edit specific regions |
| **Video Object Tracking** | SAM 2 | Track across frames |
| **Medical Imaging** | Fine-tuned SAM | Tumor segmentation |
| **Satellite Imagery** | SAM + CLIP | Building/road detection |
| **Annotation Tool** | Interactive SAM | Faster labeling |
| **AR/VR** | Real-time SAM | Object isolation |

## Performance Benchmarks

### Segmentation Quality (mIoU)

| Model | COCO | ADE20K | LVIS |
|-------|------|--------|------|
| SAM ViT-H | 79.2 | 47.1 | 44.2 |
| SAM 2 Large | 81.5 | 49.3 | 47.8 |
| SAM-HQ | 80.1 | 48.5 | 46.1 |

### Speed Comparison

| Model | Image Encode | Mask Decode | Total (512x512) |
|-------|-------------|-------------|-----------------|
| SAM ViT-H | 450ms | 20ms | 470ms |
| SAM ViT-B | 120ms | 20ms | 140ms |
| FastSAM | 40ms | - | 40ms |
| MobileSAM | 10ms | 20ms | 30ms |
| SAM 2 L | 200ms | 15ms | 215ms |

## Memory Requirements

| Model | VRAM (FP32) | VRAM (FP16) |
|-------|------------|-------------|
| SAM ViT-B | 4 GB | 2 GB |
| SAM ViT-L | 8 GB | 4 GB |
| SAM ViT-H | 12 GB | 6 GB |
| MobileSAM | 1 GB | 0.5 GB |
| SAM 2 Large | 10 GB | 5 GB |

## Tips for Best Results

### Point Prompts
```python
# Use multiple points for complex objects
points = [
    [center_x, center_y],      # Main point
    [edge_x1, edge_y1],        # Edge points help
    [edge_x2, edge_y2],
]
labels = [1, 1, 1]  # All foreground

# Add background points to exclude areas
points.append([background_x, background_y])
labels.append(0)  # Background
```

### Box Prompts
```python
# Tight boxes work better than loose ones
# Add small padding (~5-10%) for best results
padding = 0.05
x1 = x1 - width * padding
y1 = y1 - height * padding
x2 = x2 + width * padding
y2 = y2 + height * padding
```

### Automatic Generation Tuning
```python
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,        # Density of point grid
    pred_iou_thresh=0.88,      # Quality threshold
    stability_score_thresh=0.95,  # Stability threshold
    crop_n_layers=1,           # Multi-scale crops
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Filter tiny masks
)
```

## Integration Examples

### SAM + Stable Diffusion (Inpainting)

```python
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamPredictor
import numpy as np
from PIL import Image

# 1. Segment object with SAM
predictor.set_image(image)
masks, _, _ = predictor.predict(point_coords=[[x, y]], point_labels=[1])
mask = masks[0]

# 2. Convert to inpainting mask
mask_image = Image.fromarray((mask * 255).astype(np.uint8))

# 3. Inpaint with Stable Diffusion
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
result = pipe(
    prompt="a golden retriever",
    image=Image.fromarray(image),
    mask_image=mask_image
).images[0]
```

### SAM for Data Annotation

```python
# Interactive annotation workflow
class SAMAnnotator:
    def __init__(self, model_path):
        self.sam = sam_model_registry["vit_h"](checkpoint=model_path)
        self.predictor = SamPredictor(self.sam)
    
    def annotate(self, image, clicks):
        """
        clicks: list of (x, y, is_foreground) tuples
        """
        self.predictor.set_image(image)
        
        points = np.array([[c[0], c[1]] for c in clicks])
        labels = np.array([1 if c[2] else 0 for c in clicks])
        
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
        
        return masks[np.argmax(scores)]
```

## Choosing the Right SAM

```
What's your priority?
├── Speed (real-time/mobile)
│   ├── Very fast → FastSAM, MobileSAM
│   └── Fast → SAM ViT-B
│
├── Quality (highest accuracy)
│   ├── Static images → SAM ViT-H, SAM-HQ
│   └── Video → SAM 2
│
├── Video segmentation
│   └── SAM 2 (built for video)
│
└── Edge/Mobile deployment
    └── MobileSAM, EfficientSAM
```

## Related Models

- **[VLM](./VLM.md)** - For understanding image content
- **[Diffusion](./DIFFUSION.md)** - Use SAM masks for inpainting
- **[ViT](./VIT.md)** - SAM's encoder architecture

## Resources

- [SAM GitHub](https://github.com/facebookresearch/segment-anything)
- [SAM 2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [SAM Demo](https://segment-anything.com/)
- [Hugging Face SAM](https://huggingface.co/facebook/sam-vit-huge)
