# VAE - Variational Autoencoder

## Overview

**Variational Autoencoders** are generative models that learn to encode data into a structured latent space and decode it back. Unlike regular autoencoders, VAEs learn a probabilistic latent space, enabling smooth interpolation and generation of new samples.

## Key Characteristics

| Property | Value |
|----------|-------|
| **Type** | Generative model |
| **Architecture** | Encoder + Decoder |
| **Training** | ELBO maximization (reconstruction + KL divergence) |
| **Latent Space** | Continuous, regularized |
| **Output** | Images, text, molecules, audio |
| **Speed** | Fast (single pass) |

## How It Works

```
                    VAE Architecture
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │  Input x → [Encoder] → μ, σ → z ~ N(μ,σ) → [Decoder] → x̂
    │                                                 │
    └─────────────────────────────────────────────────┘

Encoder: Maps input to distribution parameters (mean μ, std σ)
Sampling: z = μ + σ * ε, where ε ~ N(0,1)  (reparameterization trick)
Decoder: Reconstructs input from latent z

Loss = Reconstruction Loss + KL Divergence
     = ||x - x̂||² + KL(q(z|x) || p(z))
```

## VAE vs Regular Autoencoder

```
Autoencoder:
Input → Encoder → z (point) → Decoder → Output
Problem: Latent space is unstructured, gaps between points

VAE:
Input → Encoder → μ, σ → z ~ N(μ,σ) → Decoder → Output
Solution: Latent space is continuous and regularized
```

| Aspect | Autoencoder | VAE |
|--------|-------------|-----|
| **Latent** | Deterministic point | Probability distribution |
| **Generation** | Cannot generate new | Can sample new |
| **Interpolation** | May have gaps | Smooth interpolation |
| **Regularization** | None | KL divergence |

## Types of VAEs

| Type | Description | Use Case |
|------|-------------|----------|
| **Vanilla VAE** | Original architecture | Basic generation |
| **β-VAE** | Stronger disentanglement | Interpretable features |
| **VQ-VAE** | Discrete latents | High-quality images |
| **VQ-VAE-2** | Hierarchical discrete | Even better images |
| **CVAE** | Conditional generation | Controlled output |
| **VAE-GAN** | GAN discriminator | Sharper images |
| **NVAE** | Deep hierarchical | SOTA image VAE |

## Popular Models Using VAE

| Model | Use | Notes |
|-------|-----|-------|
| **Stable Diffusion VAE** | Image encoding | 8x compression |
| **VQ-VAE (DALL-E 1)** | Image tokens | Discrete codes |
| **VQGAN** | Image compression | High quality |
| **AudioLDM VAE** | Audio encoding | Mel spectrograms |
| **Latent Video VAE** | Video encoding | Temporal compression |

## Examples with Code

### Example 1: Simple VAE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD
```

### Example 2: Convolutional VAE

```python
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 → 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16 → 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8 → 4
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4 → 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 32 → 64
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z):
        h = self.fc_decode(z).view(-1, 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

### Example 3: Using Stable Diffusion VAE

```python
from diffusers import AutoencoderKL
import torch
from PIL import Image
from torchvision import transforms

# Load pretrained VAE from Stable Diffusion
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16
)
vae = vae.to("cuda")

# Encode image to latent
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

image = Image.open("photo.jpg")
image_tensor = transform(image).unsqueeze(0).to("cuda", dtype=torch.float16)

with torch.no_grad():
    # Encode: 512x512 → 64x64 latent (8x compression)
    latent = vae.encode(image_tensor).latent_dist.sample()
    latent = latent * 0.18215  # Scaling factor
    
    # Decode back
    decoded = vae.decode(latent / 0.18215).sample
```

### Example 4: VQ-VAE (Vector Quantized)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z):
        # z: (B, C, H, W)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, z.size(1))
        
        # Find nearest embedding
        distances = torch.cdist(z_flat, self.embedding.weight)
        indices = distances.argmin(dim=1)
        
        # Quantize
        z_q = self.embedding(indices).view(z.permute(0, 2, 3, 1).shape)
        z_q = z_q.permute(0, 3, 1, 2)
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Losses
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        
        return z_q, commitment_loss + 0.25 * codebook_loss, indices
```

### Example 5: Latent Space Interpolation

```python
# Interpolate between two images in latent space
def interpolate(vae, image1, image2, num_steps=10):
    # Encode both images
    mu1, logvar1 = vae.encode(image1)
    mu2, logvar2 = vae.encode(image2)
    
    z1 = vae.reparameterize(mu1, logvar1)
    z2 = vae.reparameterize(mu2, logvar2)
    
    # Linear interpolation in latent space
    interpolations = []
    for alpha in torch.linspace(0, 1, num_steps):
        z_interp = (1 - alpha) * z1 + alpha * z2
        decoded = vae.decode(z_interp)
        interpolations.append(decoded)
    
    return interpolations

# Smooth morphing between faces/objects
frames = interpolate(vae, face1, face2, num_steps=30)
```

### Example 6: Conditional VAE

```python
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super().__init__()
        
        # Encoder with condition
        self.fc1 = nn.Linear(input_dim + num_classes, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder with condition
        self.fc3 = nn.Linear(latent_dim + num_classes, 400)
        self.fc4 = nn.Linear(400, input_dim)
    
    def encode(self, x, c):
        # c is one-hot class label
        xc = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1(xc))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        h = F.relu(self.fc3(zc))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# Generate specific digit
label = F.one_hot(torch.tensor([7]), num_classes=10).float()
z = torch.randn(1, latent_dim)
generated_7 = cvae.decode(z, label)
```

## Use Cases

| Use Case | Best Model | Notes |
|----------|-----------|-------|
| **Image Compression** | SD-VAE, VQGAN | High compression ratio |
| **Latent Diffusion** | SD-VAE | Enables efficient diffusion |
| **Anomaly Detection** | VAE | High reconstruction error = anomaly |
| **Data Augmentation** | CVAE | Generate variations |
| **Drug Discovery** | Molecular VAE | Generate molecules |
| **Music Generation** | MusicVAE | Interpolate melodies |
| **Face Editing** | β-VAE | Disentangled attributes |

## VAE vs GAN vs Diffusion

| Aspect | VAE | GAN | Diffusion |
|--------|-----|-----|-----------|
| **Training** | ✅ Stable | Unstable | ✅ Stable |
| **Quality** | Blurry | Sharp | ✅ Best |
| **Speed** | ✅ Fast | ✅ Fast | Slow |
| **Latent Space** | ✅ Structured | Unstructured | N/A |
| **Mode Coverage** | ✅ Full | May miss | ✅ Full |
| **Likelihood** | ✅ Tractable | Intractable | Tractable |

## The Blurriness Problem

VAEs tend to produce blurry outputs because:
1. MSE loss averages over uncertainty
2. Gaussian decoder assumption
3. KL term regularizes too strongly

**Solutions:**
```python
# 1. Perceptual loss
from torchvision.models import vgg16
vgg = vgg16(pretrained=True).features[:16]
perceptual_loss = F.mse_loss(vgg(recon), vgg(original))

# 2. Adversarial loss (VAE-GAN)
discriminator_loss = discriminator(recon)

# 3. Vector quantization (VQ-VAE)
# Discrete latents → sharper outputs

# 4. β-VAE with lower β
# Reduces KL weight: Loss = Recon + β * KL, β < 1
```

## Memory Requirements

| Model | Parameters | VRAM | Notes |
|-------|-----------|------|-------|
| Simple VAE | 1M | 0.5 GB | MNIST-scale |
| Conv VAE | 10M | 2 GB | 64x64 images |
| NVAE | 100M | 8 GB | SOTA quality |
| SD-VAE | 84M | 2 GB | 512x512 |
| VQ-VAE-2 | 50M | 4 GB | Hierarchical |

## β-VAE: Disentangled Representations

```python
# β > 1 encourages more disentanglement
# Each latent dimension captures independent factor

def beta_vae_loss(recon_x, x, mu, logvar, beta=4.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

# Result: z[0] might control size, z[1] rotation, z[2] color, etc.
```

## Applications

| Industry | Application |
|----------|-------------|
| **AI Art** | Latent space for diffusion models |
| **Healthcare** | Anomaly detection in medical images |
| **Chemistry** | Molecule generation/optimization |
| **Gaming** | Procedural content generation |
| **Finance** | Fraud detection (reconstruction error) |
| **Robotics** | World models, state representation |

## VAE in Modern AI Pipelines

```
Text-to-Image Pipeline (Stable Diffusion):

Text ──────────────────────────────────────────┐
                                               ↓
                                        [Text Encoder]
                                               │
Random Noise → [Diffusion U-Net] ← Conditioning
       ↓              ↓
  Latent Space    Denoised Latent
       │              │
       └──────────────┘
              ↓
         [VAE Decoder] ← THIS IS THE VAE
              ↓
         Final Image
```

## Related Models

- **[GAN](./GAN.md)** - Alternative with sharper outputs
- **[Diffusion](./DIFFUSION.md)** - Uses VAE for latent space
- **[Embedding Models](./EMBEDDING.md)** - Related latent representations

## Resources

- [Original VAE Paper](https://arxiv.org/abs/1312.6114)
- [β-VAE Paper](https://openreview.net/forum?id=Sy2fzU9gl)
- [VQ-VAE Paper](https://arxiv.org/abs/1711.00937)
- [NVAE Paper](https://arxiv.org/abs/2007.03898)
- [PyTorch VAE Collection](https://github.com/AntixK/PyTorch-VAE)
