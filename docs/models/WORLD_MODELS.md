# World Models

## Overview

**World Models** are AI systems that learn to understand, predict, and simulate environments or "worlds." They build internal representations that can predict future states, enabling planning, imagination, and sample-efficient learning. World models are fundamental to autonomous vehicles, robotics, video game AI, and increasingly to foundation models.

## Key Characteristics

| Property | Value |
|----------|-------|
| **Core Ability** | Predict future states |
| **Input** | Observations (images, states) |
| **Output** | Predicted states, rewards |
| **Training** | Self-supervised, RL |
| **Key Use** | Planning, simulation |

## How It Works

```
World Model Architecture:

Observation → Encoder → Latent State → Dynamics Model → Future States
                                           ↓
                                    Reward Predictor
                                           ↓
                                    Planning/Action

Classic World Model (Ha & Schmidhuber):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Observation    ┌───────────┐      ┌──────────────┐    │
│      (x_t)  ───►│  Vision   │──z──►│   Memory     │    │
│                 │  Model V  │      │   Model M    │    │
│                 │   (VAE)   │      │   (RNN/SSM)  │    │
│                 └───────────┘      └──────┬───────┘    │
│                                           │            │
│                                           ▼            │
│                                    ┌──────────────┐    │
│                                    │  Controller  │    │
│                                    │      C       │───►│ Action
│                                    └──────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘

Modern World Models (Dreamer, JEPA):
   
   Real World ────► Encoder ────► Latent Space ────► Predictor
        │                              │                  │
        │                              │                  ▼
        │                              │           Future Latent
        │                              │                  │
        │                              └─────┬────────────┘
        │                                    │
        └────────────────────────────────────┘
                    Training Loop
```

## Types of World Models

### By Architecture

| Type | Description | Example |
|------|-------------|---------|
| **Pixel-Space** | Predict future frames directly | SimVP, FutureGAN |
| **Latent-Space** | Predict in compressed space | Dreamer, JEPA |
| **State-Space** | Model dynamics in state space | MuZero, TD-MPC |
| **Hybrid** | Combine multiple approaches | Genie, GAIA-1 |

### By Domain

| Domain | Models | Application |
|--------|--------|-------------|
| **Games** | MuZero, EfficientZero | Game playing |
| **Robotics** | Dreamer, TD-MPC | Robot control |
| **Driving** | GAIA-1, NuPlan | Autonomous vehicles |
| **Video** | Genie, SORA | Video generation |
| **Physics** | Neural Physics Engine | Physical simulation |

## Key Models

| Model | Year | Key Innovation | Domain |
|-------|------|----------------|--------|
| **Ha & Schmidhuber** | 2018 | VAE + RNN | Games |
| **Dreamer V1/V2/V3** | 2020-23 | Latent imagination | General RL |
| **MuZero** | 2020 | Learned dynamics | Games |
| **JEPA** | 2022 | Joint embedding predictive | General |
| **Genie** | 2024 | Generative interactive | Video games |
| **SORA** | 2024 | World simulator | Video |
| **GAIA-1** | 2023 | Driving simulator | Autonomous driving |
| **UniSim** | 2023 | Universal simulator | General |

## Examples with Code

### Example 1: Simple World Model (VAE + RNN)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Vision model - encode observations to latent space"""
    def __init__(self, img_channels=3, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class MDN_RNN(nn.Module):
    """Memory model - predict next latent state"""
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256):
        super().__init__()
        self.rnn = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_reward = nn.Linear(hidden_dim, 1)
        
    def forward(self, z, action, hidden=None):
        x = torch.cat([z, action], dim=-1)
        output, hidden = self.rnn(x.unsqueeze(1), hidden)
        output = output.squeeze(1)
        
        mu = self.fc_mu(output)
        logvar = self.fc_logvar(output)
        reward = self.fc_reward(output)
        
        return mu, logvar, reward, hidden

class WorldModel(nn.Module):
    """Complete world model"""
    def __init__(self, img_channels=3, latent_dim=32, action_dim=3):
        super().__init__()
        self.vae = VAE(img_channels, latent_dim)
        self.rnn = MDN_RNN(latent_dim, action_dim)
        
    def imagine(self, initial_z, action_sequence, hidden=None):
        """Imagine future states given actions"""
        imagined_states = []
        z = initial_z
        
        for action in action_sequence:
            mu, logvar, reward, hidden = self.rnn(z, action, hidden)
            z = self.vae.reparameterize(mu, logvar)
            imagined_states.append((z, reward))
        
        return imagined_states

# Training
model = WorldModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for observation, action, next_obs in dataloader:
    # Encode observation
    mu, logvar = model.vae.encode(observation)
    z = model.vae.reparameterize(mu, logvar)
    
    # Predict next state
    pred_mu, pred_logvar, pred_reward, _ = model.rnn(z, action)
    
    # Encode next observation (target)
    target_mu, target_logvar = model.vae.encode(next_obs)
    
    # Losses
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    prediction_loss = F.mse_loss(pred_mu, target_mu)
    
    loss = kl_loss + prediction_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Example 2: Dreamer-Style World Model

```python
import torch
import torch.nn as nn
import torch.distributions as td

class RSSM(nn.Module):
    """Recurrent State Space Model"""
    def __init__(self, stoch_size=32, deter_size=256, hidden_size=200, action_size=4):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        
        # Recurrent model
        self.rnn = nn.GRUCell(hidden_size, deter_size)
        
        # Prior (predict stochastic state from deterministic)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size)  # mean and std
        )
        
        # Posterior (encode observation + deterministic)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_size + 1024, hidden_size),  # 1024 = encoded obs
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size)
        )
        
        # Pre-RNN
        self.pre_rnn = nn.Linear(stoch_size + action_size, hidden_size)
        
    def prior(self, deter):
        stats = self.prior_net(deter)
        mean, std = stats.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        return td.Normal(mean, std)
    
    def posterior(self, deter, embed):
        x = torch.cat([deter, embed], dim=-1)
        stats = self.posterior_net(x)
        mean, std = stats.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        return td.Normal(mean, std)
    
    def imagine(self, prev_state, action):
        """Imagination step (no observation)"""
        deter, stoch = prev_state['deter'], prev_state['stoch']
        
        # RNN step
        x = self.pre_rnn(torch.cat([stoch, action], dim=-1))
        deter = self.rnn(x, deter)
        
        # Sample from prior
        prior = self.prior(deter)
        stoch = prior.rsample()
        
        return {'deter': deter, 'stoch': stoch}
    
    def observe(self, prev_state, action, embed):
        """Observation step (with observation)"""
        deter, stoch = prev_state['deter'], prev_state['stoch']
        
        # RNN step
        x = self.pre_rnn(torch.cat([stoch, action], dim=-1))
        deter = self.rnn(x, deter)
        
        # Posterior from observation
        posterior = self.posterior(deter, embed)
        stoch = posterior.rsample()
        
        # Prior for KL
        prior = self.prior(deter)
        
        return {'deter': deter, 'stoch': stoch}, prior, posterior

class DreamerWorldModel(nn.Module):
    def __init__(self, obs_shape, action_size):
        super().__init__()
        # Encoder: observation -> embedding
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 1024)
        )
        
        # RSSM
        self.rssm = RSSM(action_size=action_size)
        
        # Decoder: state -> observation
        self.decoder = nn.Sequential(
            nn.Linear(32 + 256, 1024),
            nn.Unflatten(-1, (256, 2, 2)),
            nn.ConvTranspose2d(256, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2)
        )
        
        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(32 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def imagine_ahead(self, start_state, policy, horizon):
        """Imagine future trajectory using policy"""
        states = [start_state]
        rewards = []
        
        state = start_state
        for _ in range(horizon):
            # Get action from policy
            features = self.get_features(state)
            action = policy(features)
            
            # Imagine next state
            state = self.rssm.imagine(state, action)
            states.append(state)
            
            # Predict reward
            reward = self.reward_head(features)
            rewards.append(reward)
        
        return states, rewards
    
    def get_features(self, state):
        return torch.cat([state['stoch'], state['deter']], dim=-1)
```

### Example 3: MuZero-Style Model

```python
import torch
import torch.nn as nn

class MuZeroWorldModel(nn.Module):
    """MuZero: learns state dynamics without knowing game rules"""
    
    def __init__(self, obs_shape, action_space, hidden_size=256):
        super().__init__()
        
        # Representation: observation -> hidden state
        self.representation = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden_size)
        )
        
        # Dynamics: hidden state + action -> next hidden state + reward
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_size + action_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.reward_predictor = nn.Linear(hidden_size, 1)
        
        # Prediction: hidden state -> policy + value
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def initial_inference(self, observation):
        """h_0 = representation(o_0)"""
        hidden = self.representation(observation)
        policy = self.policy_head(hidden)
        value = self.value_head(hidden)
        return hidden, policy, value
    
    def recurrent_inference(self, hidden, action):
        """h_k, r_k = dynamics(h_{k-1}, a_{k-1})"""
        action_onehot = F.one_hot(action, num_classes=self.action_space).float()
        x = torch.cat([hidden, action_onehot], dim=-1)
        
        next_hidden = self.dynamics(x)
        reward = self.reward_predictor(next_hidden)
        policy = self.policy_head(next_hidden)
        value = self.value_head(next_hidden)
        
        return next_hidden, reward, policy, value
    
    def simulate(self, observation, action_sequence):
        """Simulate trajectory in latent space"""
        hidden, policy, value = self.initial_inference(observation)
        
        trajectory = [(hidden, None, policy, value)]
        
        for action in action_sequence:
            hidden, reward, policy, value = self.recurrent_inference(hidden, action)
            trajectory.append((hidden, reward, policy, value))
        
        return trajectory
```

### Example 4: JEPA (Joint Embedding Predictive Architecture)

```python
import torch
import torch.nn as nn

class JEPA(nn.Module):
    """
    JEPA: Predict in embedding space, not pixel space.
    Key insight: predict latent representations, not raw data.
    """
    
    def __init__(self, encoder, predictor):
        super().__init__()
        self.context_encoder = encoder  # Encode context (past)
        self.target_encoder = encoder    # Encode target (future) - EMA updated
        self.predictor = predictor       # Predict target embedding from context
        
    def forward(self, context_frames, target_frames):
        # Encode context
        context_embedding = self.context_encoder(context_frames)
        
        # Encode target (with stop gradient for target encoder)
        with torch.no_grad():
            target_embedding = self.target_encoder(target_frames)
        
        # Predict target from context
        predicted_embedding = self.predictor(context_embedding)
        
        # Loss: match predicted to actual target embedding
        loss = F.mse_loss(predicted_embedding, target_embedding)
        
        return loss
    
    @torch.no_grad()
    def update_target_encoder(self, momentum=0.99):
        """EMA update of target encoder"""
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

class VideoJEPA(nn.Module):
    """Video prediction with JEPA"""
    
    def __init__(self, embed_dim=768, num_frames=16):
        super().__init__()
        # Vision Transformer encoder
        self.encoder = ViT(embed_dim=embed_dim)
        
        # Predictor: lightweight transformer
        self.predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),
            num_layers=4
        )
        
        # Position embeddings for future frames
        self.future_pos = nn.Parameter(torch.randn(num_frames, embed_dim))
        
    def forward(self, context_video, target_video):
        # Encode all frames
        B, T, C, H, W = context_video.shape
        
        # Context: encode visible frames
        context_flat = context_video.view(B * T, C, H, W)
        context_embed = self.encoder(context_flat).view(B, T, -1)
        
        # Target: encode future frames (stop gradient)
        T_target = target_video.shape[1]
        target_flat = target_video.view(B * T_target, C, H, W)
        with torch.no_grad():
            target_embed = self.encoder(target_flat).view(B, T_target, -1)
        
        # Predict future embeddings
        pred_embed = self.predictor(
            context_embed,
            self.future_pos[:T_target].unsqueeze(0).expand(B, -1, -1)
        )
        
        # L2 loss in embedding space
        loss = F.mse_loss(pred_embed, target_embed)
        
        return loss
```

## Use Cases

| Domain | Application | Example Model |
|--------|-------------|---------------|
| **Games** | Game-playing AI | MuZero, EfficientZero |
| **Robotics** | Manipulation, locomotion | Dreamer, TD-MPC |
| **Autonomous Driving** | Scenario simulation | GAIA-1 |
| **Video Generation** | Consistent video synthesis | SORA, Genie |
| **Planning** | Long-horizon planning | Any world model |
| **Counterfactual** | "What if" simulation | UniSim |

## Benefits of World Models

| Benefit | Description |
|---------|-------------|
| **Sample Efficiency** | Learn from imagined experience |
| **Planning** | Look ahead before acting |
| **Generalization** | Transfer to new scenarios |
| **Safety** | Test actions in simulation |
| **Explainability** | Visualize predicted futures |

## Comparison

| Model | Architecture | Prediction | Domain |
|-------|--------------|------------|--------|
| **Dreamer** | RSSM + Actor-Critic | Latent | General RL |
| **MuZero** | Dynamics + MCTS | Latent | Games |
| **JEPA** | Joint embedding | Embedding | Self-supervised |
| **Genie** | Latent action | Video | Interactive |
| **SORA** | DiT | Video | Generation |

## Related Models

- **[RL](./RL.md)** - Uses world models for planning
- **[VIT](./VIT.md)** - Vision encoder component
- **[DIFFUSION](./DIFFUSION.md)** - Generation component
- **[VIDEO](./VIDEO.md)** - Video prediction

## Resources

- [World Models Paper (Ha & Schmidhuber)](https://arxiv.org/abs/1803.10122)
- [Dreamer V3](https://arxiv.org/abs/2301.04104)
- [MuZero Paper](https://arxiv.org/abs/1911.08265)
- [JEPA (Yann LeCun)](https://arxiv.org/abs/2212.03132)
- [Genie Paper](https://arxiv.org/abs/2402.15391)
- [SORA Technical Report](https://openai.com/sora)
