# Video Models - Video AI

## Overview

**Video Models** process and generate video content, understanding temporal dynamics, motion, and the relationship between frames. This includes video generation from text, video understanding, action recognition, video captioning, and video editing.

## Categories

| Category | Input | Output | Examples |
|----------|-------|--------|----------|
| **Text-to-Video** | Text | Video | Sora, Runway Gen-3 |
| **Image-to-Video** | Image | Video | Stable Video Diffusion |
| **Video Understanding** | Video | Text/Labels | VideoLLaMA, Video-LLaVA |
| **Action Recognition** | Video | Actions | TimeSformer, VideoMAE |
| **Video Captioning** | Video | Text | VideoBLIP |
| **Video Editing** | Video + Text | Video | Runway, Pika |
| **Video Interpolation** | Frames | Frames | FILM, RIFE |

---

## Text-to-Video Generation

### Key Models

| Model | Provider | Max Duration | Resolution | Best For |
|-------|----------|--------------|------------|----------|
| **Sora** | OpenAI | 60s | Up to 1080p | Highest quality |
| **Runway Gen-3 Alpha** | Runway | 10s | 720p-4K | Professional |
| **Pika 1.0** | Pika Labs | 4s | 1080p | Creative |
| **Kling** | Kuaishou | 5s | 1080p | Motion quality |
| **Dream Machine** | Luma AI | 5s | 720p | Fast iteration |
| **Stable Video Diffusion** | Stability | 4s | 576p | Open-source |
| **AnimateDiff** | Community | 16 frames | 512p | Animation |
| **CogVideo** | THUDM | 6s | 720p | Open-source |

### Example: Runway Gen-3

```python
import runwayml

client = runwayml.RunwayML()

# Generate video from text
task = client.image_to_video.create(
    model='gen3a_turbo',
    prompt_image="https://example.com/image.jpg",  # or base64
    prompt_text="A serene lake with gentle ripples, birds flying overhead, golden hour lighting",
    duration=10,  # seconds
    watermark=False,
)

# Wait for completion
result = client.tasks.retrieve(task.id)
while result.status != "SUCCEEDED":
    time.sleep(5)
    result = client.tasks.retrieve(task.id)

video_url = result.output[0]
print(f"Video: {video_url}")
```

### Example: Stable Video Diffusion

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

# Load model
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Load conditioning image
image = load_image("input_image.png")
image = image.resize((1024, 576))

# Generate video frames
generator = torch.manual_seed(42)
frames = pipe(
    image,
    decode_chunk_size=8,
    generator=generator,
    num_frames=25,  # ~3s at 8fps
    motion_bucket_id=127,  # Higher = more motion (0-255)
    noise_aug_strength=0.02
).frames[0]

# Export to video
export_to_video(frames, "output_video.mp4", fps=8)
```

### Example: AnimateDiff

```python
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif

# Load motion adapter
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16
)

# Load pipeline with motion
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16
)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# Generate animated sequence
output = pipe(
    prompt="A beautiful sunset over ocean waves, cinematic",
    negative_prompt="blurry, low quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25
)

# Save as GIF
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

### Example: CogVideoX

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Load model
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Generate video
prompt = "A serene lake surrounded by mountains at sunrise, time-lapse"

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).frames[0]

export_to_video(video, "cogvideo_output.mp4", fps=8)
```

---

## Video Understanding

### Key Models

| Model | Architecture | Tasks | Best For |
|-------|--------------|-------|----------|
| **GPT-4o** | Multimodal | Understanding + Chat | General analysis |
| **Gemini 1.5 Pro** | Multimodal | Long video | Long context |
| **Video-LLaVA** | LLaVA + Video | QA, Captioning | Open-source |
| **VideoChat2** | Video-LLM | Conversation | Dialog |
| **InternVideo2** | Foundation | Multiple | Research |
| **VideoBLIP** | BLIP + Temporal | Captioning | Descriptions |

### Example: Video Analysis with GPT-4o

```python
from openai import OpenAI
import base64

client = OpenAI()

def extract_frames(video_path, num_frames=10):
    """Extract frames from video using OpenCV"""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            frames.append(base64.b64encode(buffer).decode('utf-8'))
    
    cap.release()
    return frames

# Extract frames
frames = extract_frames("video.mp4", num_frames=10)

# Create message with frames
content = [{"type": "text", "text": "Describe what happens in this video:"}]
for frame in frames:
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
    })

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": content}],
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Example: Video-LLaVA

```python
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import numpy as np
import av

def read_video(path, num_frames=8):
    """Read video frames"""
    container = av.open(path)
    total_frames = container.streams.video[0].frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    
    return np.stack(frames)

# Load model
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Read video
video = read_video("example.mp4", num_frames=8)

# Process
prompt = "USER: <video>\nWhat is happening in this video? ASSISTANT:"
inputs = processor(text=prompt, videos=video, return_tensors="pt").to("cuda")

# Generate
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## Action Recognition

### Key Models

| Model | Architecture | Accuracy (K400) | Best For |
|-------|--------------|-----------------|----------|
| **VideoMAE V2** | ViT + MAE | 87.4% | General |
| **InternVideo2** | ViT-L | 89.1% | State-of-the-art |
| **TimeSformer** | Transformer | 80.7% | Efficiency |
| **Video Swin** | Swin + Temporal | 84.9% | Accuracy |
| **SlowFast** | Two-stream | 79.8% | Speed-accuracy |
| **X3D** | Efficient 3D | 79.1% | Efficiency |

### Example: Action Recognition with VideoMAE

```python
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch
import av
import numpy as np

def sample_frames(video_path, num_frames=16):
    container = av.open(video_path)
    frames = []
    
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    
    # Sample evenly
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return [frames[i] for i in indices]

# Load model
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

# Process video
video = sample_frames("action_video.mp4", num_frames=16)
inputs = processor(video, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get top predictions
probs = torch.softmax(logits, dim=-1)
top5 = torch.topk(probs, 5)

for idx, prob in zip(top5.indices[0], top5.values[0]):
    label = model.config.id2label[idx.item()]
    print(f"{label}: {prob:.2%}")
```

### Example: Real-time Action Recognition

```python
import cv2
import torch
from collections import deque
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

class RealTimeActionRecognizer:
    def __init__(self, model_name="MCG-NJU/videomae-base-finetuned-kinetics", buffer_size=16):
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.model.eval()
        
        self.frame_buffer = deque(maxlen=buffer_size)
    
    def predict(self):
        if len(self.frame_buffer) < 16:
            return None
        
        frames = list(self.frame_buffer)
        inputs = self.processor(frames, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        top_idx = probs.argmax().item()
        return self.model.config.id2label[top_idx], probs[0, top_idx].item()
    
    def add_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_buffer.append(frame_rgb)

# Usage with webcam
recognizer = RealTimeActionRecognizer()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    recognizer.add_frame(frame)
    
    result = recognizer.predict()
    if result:
        action, confidence = result
        cv2.putText(frame, f"{action}: {confidence:.2%}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Action Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Video Editing & Effects

### Key Models

| Model | Task | Input | Output |
|-------|------|-------|--------|
| **Runway Gen-3** | Edit | Video + Text | Edited video |
| **Pika** | Edit + Effects | Video + Text | Edited video |
| **FILM** | Interpolation | 2 frames | N frames |
| **RIFE** | Interpolation | 2 frames | N frames |
| **CoDeF** | Editing | Video + Text | Edited video |
| **ProPainter** | Inpainting | Video + Mask | Filled video |

### Example: Frame Interpolation with RIFE

```python
import cv2
import torch
from rife_model import RIFE

# Load model
model = RIFE()
model.load_model("train_log", -1)
model.eval()

def interpolate_frames(frame1, frame2, num_intermediate=1):
    """Generate intermediate frames between two frames"""
    # Normalize frames
    f1 = torch.from_numpy(frame1).permute(2, 0, 1).float() / 255.0
    f2 = torch.from_numpy(frame2).permute(2, 0, 1).float() / 255.0
    
    f1 = f1.unsqueeze(0)
    f2 = f2.unsqueeze(0)
    
    intermediate = []
    for i in range(1, num_intermediate + 1):
        ratio = i / (num_intermediate + 1)
        with torch.no_grad():
            mid = model.inference(f1, f2, ratio)
        
        mid_np = (mid[0].permute(1, 2, 0).numpy() * 255).astype('uint8')
        intermediate.append(mid_np)
    
    return intermediate

# Double video frame rate
cap = cv2.VideoCapture("input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# Output at 2x fps
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_2x.mp4", fourcc, fps * 2, (width, height))

ret, prev_frame = cap.read()
while True:
    ret, curr_frame = cap.read()
    if not ret:
        break
    
    # Write original frame
    out.write(prev_frame)
    
    # Generate and write interpolated frame
    mid_frames = interpolate_frames(prev_frame, curr_frame, 1)
    for mid in mid_frames:
        out.write(mid)
    
    prev_frame = curr_frame

out.release()
cap.release()
```

### Example: Video Object Removal

```python
import cv2
import torch
from propainter import ProPainter

# Load model
model = ProPainter()
model.load_model("propainter.pth")
model.to("cuda")

def remove_object_from_video(video_path, masks_path, output_path):
    """Remove object using mask video"""
    cap = cv2.VideoCapture(video_path)
    mask_cap = cv2.VideoCapture(masks_path)
    
    frames = []
    masks = []
    
    while True:
        ret, frame = cap.read()
        ret_m, mask = mask_cap.read()
        if not ret:
            break
        
        frames.append(frame)
        masks.append(mask[:,:,0])  # Grayscale mask
    
    # Process in batches
    frames_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    masks_tensor = torch.from_numpy(np.stack(masks)).unsqueeze(1).float() / 255.0
    
    # Inpaint
    with torch.no_grad():
        result = model(frames_tensor.cuda(), masks_tensor.cuda())
    
    # Save output
    result_np = (result.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = result_np[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    for frame in result_np:
        out.write(frame)
    
    out.release()
```

---

## Benchmarks

### Text-to-Video Quality

| Model | FVD ↓ | CLIPSIM ↑ | Motion | Consistency |
|-------|-------|-----------|--------|-------------|
| Sora | ~200* | ~0.30* | Excellent | Excellent |
| Gen-3 Alpha | ~250 | ~0.28 | Very Good | Very Good |
| Kling | ~280 | ~0.27 | Excellent | Good |
| Pika | ~320 | ~0.26 | Good | Good |
| SVD | ~400 | ~0.24 | Good | Good |

*Estimated based on demo videos

### Action Recognition (Kinetics-400)

| Model | Top-1 Acc | Top-5 Acc | FLOPs |
|-------|-----------|-----------|-------|
| InternVideo2-L | 89.1% | 97.8% | 612G |
| VideoMAE V2-g | 87.4% | 97.6% | 582G |
| Video Swin-L | 84.9% | 96.7% | 604G |
| TimeSformer-L | 80.7% | 94.7% | 2380G |

---

## Choosing Video Models

```
What's your task?
│
├── Generate Video
│   ├── Highest quality → Sora (when available)
│   ├── Production use → Runway Gen-3
│   ├── Open-source → Stable Video Diffusion
│   └── Animation → AnimateDiff
│
├── Understand Video
│   ├── General QA → GPT-4o or Gemini 1.5
│   ├── Long videos → Gemini 1.5 Pro
│   ├── Open-source → Video-LLaVA
│   └── Detailed captioning → VideoBLIP
│
├── Classify Actions
│   ├── Best accuracy → InternVideo2
│   ├── General → VideoMAE V2
│   └── Efficient → SlowFast, X3D
│
└── Edit Video
    ├── Text-guided editing → Runway
    ├── Frame interpolation → RIFE
    └── Object removal → ProPainter
```

## Limitations

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Temporal coherence** | Objects change between frames | Longer training, consistency losses |
| **Physics** | Unrealistic physics | Physics-based priors |
| **Long duration** | Quality degrades | Chunk-based generation |
| **Compute cost** | Very expensive | Efficient architectures |
| **Motion blur** | Artifacts in fast motion | Motion conditioning |

## Related Models

- **[DIFFUSION](./DIFFUSION.md)** - Underlying generation architecture
- **[VLM](./VLM.md)** - Video understanding capabilities
- **[LMM](./LMM.md)** - Multimodal video + language
- **[VIT](./VIT.md)** - Vision backbone for video models
- **[AUDIO](./AUDIO.md)** - Audio generation for videos

## Resources

- [Runway ML](https://runwayml.com/)
- [Pika Labs](https://pika.art/)
- [Luma Dream Machine](https://lumalabs.ai/dream-machine)
- [Stable Video Diffusion](https://stability.ai/stable-video)
- [AnimateDiff](https://animatediff.github.io/)
- [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE)
