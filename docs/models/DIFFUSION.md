# Diffusion Models

## Overview

**Diffusion Models** are generative models that learn to create data by reversing a gradual noising process. They have become the state-of-the-art for image generation, producing higher quality outputs than GANs with more stable training.

## Key Characteristics

| Property | Value |
|----------|-------|
| **Type** | Generative model |
| **Training** | Denoising score matching |
| **Architecture** | U-Net, DiT (Diffusion Transformer) |
| **Output** | Images, video, audio, 3D |
| **Quality** | State-of-the-art |
| **Inference** | 20-50 steps typically |

## How It Works

### Forward Process (Adding Noise)
```
Clean Image → Add noise → More noise → ... → Pure Noise
    x₀     →    x₁     →    x₂     → ... →    xₜ
```

### Reverse Process (Denoising)
```
Pure Noise → Denoise → Denoise → ... → Clean Image
    xₜ     →  x_{t-1} →  x_{t-2} → ... →    x₀
```

### Training
```
1. Take clean image x₀
2. Add noise at random timestep t to get xₜ
3. Train network to predict the noise ε
4. Loss = ||ε - ε_predicted||²
```

### Inference
```
1. Start with random noise xₜ
2. Predict noise at each step
3. Subtract predicted noise
4. Repeat for T steps
5. Get clean image x₀
```

## Types of Diffusion Models

### 1. Latent Diffusion (Stable Diffusion)
```
Text Prompt → Text Encoder → Cross-Attention
                                   ↓
Latent Noise → U-Net (denoising) → Latent → VAE Decoder → Image
```

### 2. Pixel Diffusion (DALL-E 2, Imagen)
```
Text → Diffusion directly in pixel space → Image
(Higher quality, but much slower and more expensive)
```

### 3. Diffusion Transformers (DiT)
```
Noise + Condition → Transformer blocks → Denoised output
(Used in SORA, Flux, SD3)
```

## Popular Models

### Text-to-Image

| Model | Resolution | Speed | Quality | Open Source |
|-------|-----------|-------|---------|-------------|
| **Stable Diffusion 1.5** | 512x512 | Fast | Good | ✅ Yes |
| **Stable Diffusion 2.1** | 768x768 | Fast | Better | ✅ Yes |
| **SDXL** | 1024x1024 | Medium | Excellent | ✅ Yes |
| **SD 3.5** | 1024x1024 | Medium | Excellent | ✅ Yes |
| **Flux.1** | 1024x1024 | Fast | Excellent | ✅ Yes |
| **DALL-E 3** | 1024x1024 | Medium | Excellent | ❌ API only |
| **Midjourney v6** | 1024x1024 | Medium | Excellent | ❌ Closed |
| **Imagen 3** | Variable | Medium | Excellent | ❌ API only |

### Video Generation

| Model | Length | Quality | Open Source |
|-------|--------|---------|-------------|
| **Stable Video Diffusion** | 4s | Good | ✅ Yes |
| **SORA** | 60s | Excellent | ❌ Closed |
| **Runway Gen-3** | 10s | Excellent | ❌ API only |
| **Kling** | 5s | Excellent | ❌ API only |

### Other Modalities

| Model | Type | Use Case |
|-------|------|----------|
| **AudioLDM** | Audio | Sound effects, music |
| **Stable Audio** | Music | Music generation |
| **Point-E** | 3D | Point cloud generation |
| **Shap-E** | 3D | 3D mesh generation |

## Examples with Code

### Example 1: Basic Text-to-Image

```python
from diffusers import StableDiffusionPipeline
import torch

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate image
prompt = "A majestic lion in a field of flowers, digital art, highly detailed"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("lion.png")
```

### Example 2: SDXL for High Quality

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to("cuda")

prompt = "Astronaut riding a horse on Mars, cinematic lighting, 8k"
negative_prompt = "blurry, low quality, distorted"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=1024,
    width=1024
).images[0]
```

### Example 3: Image-to-Image

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Load and transform existing image
init_image = Image.open("sketch.png").resize((512, 512))

prompt = "A detailed oil painting of a landscape"
image = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,  # 0 = no change, 1 = complete change
    guidance_scale=7.5
).images[0]
```

### Example 4: Inpainting

```python
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

image = Image.open("photo.png").resize((512, 512))
mask = Image.open("mask.png").resize((512, 512))  # White = area to change

prompt = "A cat sitting on the couch"
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50
).images[0]
```

### Example 5: ControlNet (Guided Generation)

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from PIL import Image
import torch

# Load ControlNet for edge-guided generation
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Detect edges in source image
canny = CannyDetector()
image = Image.open("source.png")
canny_image = canny(image)

# Generate with edge guidance
result = pipe(
    prompt="A beautiful house, photorealistic",
    image=canny_image,
    num_inference_steps=30
).images[0]
```

### Example 6: Flux (Latest State-of-the-Art)

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

# Flux excels at following complex prompts
prompt = """
A photorealistic image of a steampunk robot sitting at a 
Victorian desk, writing with a quill pen. Brass gears visible, 
warm candlelight, intricate details, 8k resolution.
"""

image = pipe(
    prompt=prompt,
    num_inference_steps=4,  # Schnell is fast!
    guidance_scale=0,
).images[0]
```

## Key Techniques

### 1. Classifier-Free Guidance (CFG)
```python
# guidance_scale controls prompt adherence
# Higher = follows prompt more strictly (but can be overcooked)
guidance_scale = 7.5  # Typical range: 5-15
```

### 2. Negative Prompts
```python
# What NOT to include
negative_prompt = "blurry, low quality, deformed, ugly, bad anatomy"
```

### 3. Schedulers (Samplers)
```python
from diffusers import DPMSolverMultistepScheduler

# Faster sampling
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# Now can use fewer steps (20-25 instead of 50)
```

### 4. LoRA (Lightweight Fine-tuning)
```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("base_model")
pipe.load_lora_weights("path/to/lora")  # Add custom style/character
```

## Use Cases

| Use Case | Best Model | Notes |
|----------|-----------|-------|
| **General Art** | SDXL, Flux | Versatile |
| **Photorealism** | SDXL, Midjourney | Realistic images |
| **Anime/Illustration** | SD 1.5 + LoRA | Many community models |
| **Concept Art** | SDXL, SD3 | Great composition |
| **Product Design** | DALL-E 3 | Clean, commercial |
| **Architecture** | ControlNet | Precise structure |
| **Video** | SVD, Runway | Animation |
| **3D Assets** | Shap-E, Point-E | 3D generation |

## Memory Requirements

| Model | VRAM (FP16) | VRAM (FP32) | Notes |
|-------|------------|-------------|-------|
| SD 1.5 | 4 GB | 8 GB | Most compatible |
| SD 2.1 | 5 GB | 10 GB | - |
| SDXL | 8 GB | 16 GB | Needs good GPU |
| SDXL + Refiner | 12 GB | 24 GB | Two models |
| Flux.1 | 12-24 GB | N/A | BFloat16 only |
| SD3 | 10 GB | N/A | Efficient |

### Memory Optimization

```python
# Enable attention slicing (reduces memory)
pipe.enable_attention_slicing()

# Use CPU offloading
pipe.enable_model_cpu_offload()

# Use VAE slicing for large images
pipe.enable_vae_slicing()

# Use xformers for memory efficiency
pipe.enable_xformers_memory_efficient_attention()
```

## Speed Comparison

| Model | Steps | Time (RTX 3090) | Quality |
|-------|-------|-----------------|---------|
| SD 1.5 | 50 | 3s | Good |
| SD 1.5 + DPM | 20 | 1.5s | Good |
| SDXL | 30 | 8s | Excellent |
| LCM-SDXL | 4 | 1s | Good |
| Flux Schnell | 4 | 2s | Excellent |
| SDXL Turbo | 1-4 | 0.5s | Good |

## LCM (Latent Consistency Models)

Fast distilled versions:
```python
from diffusers import LatentConsistencyModelPipeline

pipe = LatentConsistencyModelPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Only 4 steps needed!
image = pipe("Beautiful sunset over ocean", num_inference_steps=4).images[0]
```

## Diffusion vs Other Generative Models

| Aspect | Diffusion | GAN | VAE |
|--------|-----------|-----|-----|
| **Quality** | ✅ Excellent | Good | Medium |
| **Diversity** | ✅ High | Medium | High |
| **Training** | ✅ Stable | Unstable | Stable |
| **Speed** | Slow | ✅ Fast | ✅ Fast |
| **Mode Collapse** | ✅ No | Yes | ✅ No |
| **Controllability** | ✅ Excellent | Limited | Limited |

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Blurry images | Increase steps, use better scheduler |
| Not following prompt | Increase guidance_scale (7-12) |
| Artifacts/distortions | Add to negative prompt, reduce guidance |
| OOM (Out of Memory) | Enable attention slicing, reduce size |
| Slow generation | Use LCM, Turbo, or fewer steps |
| Bad hands/anatomy | Use specific negative prompts, inpaint |

## Best Practices

1. **Start with SDXL or Flux** for general use
2. **Use 25-30 steps** with DPM++ scheduler
3. **Guidance scale 7-9** for balanced results
4. **Always use negative prompts** for quality
5. **Use ControlNet** for precise control
6. **Fine-tune with LoRA** for specific styles

## Related Models

- **[GAN](./GAN.md)** - Alternative generative approach
- **[VAE](./VAE.md)** - Used as encoder/decoder in latent diffusion
- **[LCM](./LCM.md)** - Fast diffusion variants
- **[VLM](./VLM.md)** - For understanding generated images

## Resources

- [Diffusers Library](https://huggingface.co/docs/diffusers)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [Civitai](https://civitai.com/) - Community models
- [Hugging Face Models](https://huggingface.co/models?library=diffusers)
