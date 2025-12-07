# GAN - Generative Adversarial Networks

## Overview

**Generative Adversarial Networks** consist of two neural networks—a Generator and a Discriminator—competing against each other. The Generator creates fake data while the Discriminator tries to distinguish real from fake, leading to increasingly realistic outputs.

## Key Characteristics

| Property | Value |
|----------|-------|
| **Type** | Generative model |
| **Architecture** | Generator + Discriminator |
| **Training** | Adversarial (minimax game) |
| **Output** | Images, video, audio |
| **Inference** | Single forward pass (fast) |
| **Quality** | Excellent (but being surpassed by diffusion) |

## How It Works

```
         ┌─────────────────────────────────────────────────┐
         │                   GAN Training                   │
         └─────────────────────────────────────────────────┘
         
Random Noise (z)                          Real Images
      │                                        │
      ▼                                        ▼
┌─────────────┐                         ┌─────────────┐
│  Generator  │                         │    Real     │
│     (G)     │                         │   Dataset   │
└──────┬──────┘                         └──────┬──────┘
       │                                       │
       ▼                                       ▼
 Fake Images ─────────────────────────► ┌─────────────┐
                                        │Discriminator│
                                        │     (D)     │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        Real or Fake?
                                        
Training:
- D tries to correctly classify real vs fake
- G tries to fool D into thinking fake is real
- They improve each other through competition
```

## Types of GANs

### By Architecture

| Type | Description | Use Case |
|------|-------------|----------|
| **Vanilla GAN** | Original architecture | Basic generation |
| **DCGAN** | Convolutional layers | Image generation |
| **StyleGAN** | Style-based synthesis | High-quality faces |
| **ProGAN** | Progressive growing | High-resolution |
| **BigGAN** | Large-scale training | ImageNet generation |

### By Conditioning

| Type | Description | Use Case |
|------|-------------|----------|
| **cGAN** | Conditional on labels | Controlled generation |
| **Pix2Pix** | Image-to-image | Style transfer |
| **CycleGAN** | Unpaired translation | Domain transfer |
| **SPADE** | Semantic layout | Scene generation |

### By Application

| Type | Description | Use Case |
|------|-------------|----------|
| **StyleGAN2/3** | Face generation | Portraits, avatars |
| **GauGAN** | Landscape generation | Scene creation |
| **StarGAN** | Multi-domain transfer | Face attributes |
| **AnimeGAN** | Photo to anime | Style conversion |

## Popular Models

### Face Generation

| Model | Resolution | FID Score | Notes |
|-------|-----------|-----------|-------|
| **StyleGAN3** | 1024x1024 | 2.79 | Alias-free, smooth video |
| **StyleGAN2** | 1024x1024 | 2.84 | Previous SOTA |
| **StyleGAN-XL** | 1024x1024 | 2.02 | ImageNet trained |
| **EG3D** | 512x512 | - | 3D-aware faces |

### General Image Generation

| Model | Dataset | Quality |
|-------|---------|---------|
| **BigGAN** | ImageNet | Excellent |
| **BigGAN-deep** | ImageNet | Best GAN quality |
| **GigaGAN** | Various | Scalable |

### Image-to-Image

| Model | Task | Paired Data? |
|-------|------|--------------|
| **Pix2Pix** | Translation | Yes |
| **CycleGAN** | Translation | No |
| **UNIT** | Translation | No |
| **MUNIT** | Multi-modal | No |

## Examples with Code

### Example 1: Generate Faces with StyleGAN2

```python
import torch
from PIL import Image

# Load pretrained StyleGAN2
# Using NVIDIA's official implementation
import dnnlib
import legacy

network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'

with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].cuda()

# Generate random face
z = torch.randn([1, G.z_dim]).cuda()
c = None  # No class conditioning
img = G(z, c)

# Convert to PIL Image
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
Image.fromarray(img[0].cpu().numpy(), 'RGB').save('generated_face.png')
```

### Example 2: Image-to-Image with Pix2Pix

```python
from transformers import pipeline
from PIL import Image

# Using Hugging Face's implementation
pipe = pipeline("image-to-image", model="lllyasviel/sd-controlnet-canny")

# Or using original Pix2Pix
import torch
from torchvision import transforms

# Load edge image (input)
input_image = Image.open("edges.png")

# For facades: edges → building
# For maps: satellite → map
# For shoes: edges → shoe
```

### Example 3: CycleGAN (Unpaired Translation)

```python
# Horse ↔ Zebra, Photo ↔ Monet, Summer ↔ Winter
import torch
from PIL import Image
from torchvision import transforms

# Load pretrained CycleGAN
# Models available: horse2zebra, monet2photo, etc.

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Convert horse to zebra
horse_image = Image.open("horse.jpg")
input_tensor = transform(horse_image).unsqueeze(0)

# G_AB transforms A → B (horse → zebra)
with torch.no_grad():
    zebra_tensor = G_AB(input_tensor)
```

### Example 4: Conditional Generation

```python
import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_embedding), -1)
        return self.model(gen_input)

# Generate specific digit (e.g., "7")
noise = torch.randn(1, 100)
label = torch.tensor([7])  # Generate digit 7
generated = generator(noise, label)
```

### Example 5: StyleGAN Latent Space Manipulation

```python
# Interpolation between two faces
z1 = torch.randn([1, 512]).cuda()  # Face 1
z2 = torch.randn([1, 512]).cuda()  # Face 2

# Linear interpolation
for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
    z_interp = (1 - alpha) * z1 + alpha * z2
    img = G(z_interp, None)
    # Save intermediate face

# Style mixing
w1 = G.mapping(z1, None)  # Get W latent
w2 = G.mapping(z2, None)

# Mix styles: coarse from w1, fine from w2
w_mixed = w1.clone()
w_mixed[:, 4:] = w2[:, 4:]  # Mix at layer 4
img_mixed = G.synthesis(w_mixed)
```

## Training a GAN

### Basic GAN Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize
generator = Generator(latent_dim=100)
discriminator = Discriminator()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Real images
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, 100)
        fake_images = generator(noise)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # Fool discriminator
        
        g_loss.backward()
        optimizer_G.step()
```

## Use Cases

| Use Case | Best Model | Notes |
|----------|-----------|-------|
| **Face Generation** | StyleGAN3 | SOTA for faces |
| **Face Editing** | StyleGAN + encoder | Latent manipulation |
| **Super Resolution** | ESRGAN, Real-ESRGAN | Image upscaling |
| **Inpainting** | DeepFill, LaMa | Fill missing regions |
| **Style Transfer** | CycleGAN, Neural Style | Artistic styles |
| **Data Augmentation** | cGAN, StyleGAN | Synthetic data |
| **Video Generation** | StyleGAN-V, VideoGPT | Video synthesis |
| **Image Colorization** | Pix2Pix, DeOldify | B&W to color |

## GAN vs Diffusion Models

| Aspect | GANs | Diffusion |
|--------|------|-----------|
| **Speed** | ✅ Fast (single pass) | Slow (many steps) |
| **Quality** | Excellent | ✅ Slightly better |
| **Training** | Unstable | ✅ Stable |
| **Mode Coverage** | May miss modes | ✅ Full coverage |
| **Controllability** | Limited | ✅ Excellent |
| **Memory** | ✅ Lower | Higher |
| **Current Trend** | Declining | ✅ Rising |

## Common Problems & Solutions

### Mode Collapse
Generator produces limited variety
```python
# Solutions:
# 1. Minibatch discrimination
# 2. Feature matching loss
# 3. Unrolled GAN training
# 4. Wasserstein loss (WGAN)
```

### Training Instability
```python
# Solutions:
# 1. Two-Timescale Update Rule (TTUR)
# 2. Spectral normalization
# 3. Gradient penalty (WGAN-GP)
# 4. Progressive growing
```

### Checkerboard Artifacts
```python
# Solution: Use resize-convolution instead of deconvolution
nn.Upsample(scale_factor=2, mode='nearest'),
nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
```

## Memory Requirements

| Model | Parameters | VRAM Training | VRAM Inference |
|-------|-----------|---------------|----------------|
| DCGAN | 5M | 2 GB | 0.5 GB |
| StyleGAN2 (256) | 30M | 8 GB | 2 GB |
| StyleGAN2 (1024) | 30M | 16 GB | 4 GB |
| BigGAN | 160M | 32 GB | 8 GB |
| StyleGAN-XL | 170M | 32 GB | 8 GB |

## Evaluation Metrics

| Metric | Measures | Lower/Higher Better |
|--------|----------|---------------------|
| **FID** | Quality + diversity | Lower |
| **IS (Inception Score)** | Quality + diversity | Higher |
| **LPIPS** | Perceptual similarity | Lower |
| **KID** | Distribution similarity | Lower |

## StyleGAN Architecture

```
Mapping Network:
z ∈ Z (512-dim) → 8 FC layers → w ∈ W (512-dim)

Synthesis Network:
Constant 4×4 → Style modulation → Upsample → ... → 1024×1024

Style Injection at each layer:
- Coarse (4-8): Pose, face shape
- Middle (16-32): Features, hair style  
- Fine (64-1024): Colors, micro details
```

## Applications in Industry

| Industry | Application | Example |
|----------|-------------|---------|
| **Gaming** | Character creation | MetaHuman-like |
| **Fashion** | Virtual try-on | Clothing on model |
| **Real Estate** | Virtual staging | Furniture in rooms |
| **Entertainment** | De-aging, face swap | Movie VFX |
| **Healthcare** | Medical imaging | Synthetic training data |
| **E-commerce** | Product images | Background removal |

## Related Models

- **[Diffusion](./DIFFUSION.md)** - Modern alternative with better quality
- **[VAE](./VAE.md)** - Related generative model
- **[VLM](./VLM.md)** - For understanding generated images

## Resources

- [GAN Lab (Interactive)](https://poloclub.github.io/ganlab/)
- [Original GAN Paper](https://arxiv.org/abs/1406.2661)
- [StyleGAN3 Paper](https://arxiv.org/abs/2106.12423)
- [PyTorch GAN Zoo](https://github.com/facebookresearch/pytorch_GAN_zoo)
- [Hugging Face Models](https://huggingface.co/models?other=gan)
