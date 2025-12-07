# RL - Reinforcement Learning Models

## Overview

**Reinforcement Learning Models** learn through trial and error by interacting with an environment. The agent takes actions to maximize cumulative rewards, making RL ideal for sequential decision-making, game playing, robotics, and increasingly for LLM alignment (RLHF).

## Key Characteristics

| Property | Value |
|----------|-------|
| **Learning Paradigm** | Trial and error |
| **Input** | State observations |
| **Output** | Actions |
| **Feedback** | Rewards (scalar) |
| **Key Concept** | Maximize cumulative reward |

## How It Works

```
Reinforcement Learning Loop:

    ┌─────────────────────────────────────────┐
    │                                         │
    ▼                                         │
┌───────┐         ┌─────────────┐            │
│ Agent │ ──────► │ Environment │            │
│       │ Action  │             │            │
│   π   │         │     s,r     │            │
└───────┘ ◄────── └─────────────┘            │
    │       State,                           │
    │       Reward                           │
    │                                         │
    └─────────────────────────────────────────┘

Goal: Learn policy π(a|s) that maximizes:
      E[Σ γᵗ rₜ]  (expected discounted return)
```

## Types of RL Algorithms

### Value-Based Methods

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **Q-Learning** | Learn action-value function | Discrete actions |
| **DQN** | Deep Q-Network | Atari games |
| **Double DQN** | Reduce overestimation | Improved DQN |
| **Dueling DQN** | Separate value/advantage | Better stability |
| **Rainbow** | Combined improvements | State-of-the-art DQN |

### Policy Gradient Methods

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **REINFORCE** | Vanilla policy gradient | Simple, educational |
| **A2C/A3C** | Advantage actor-critic | Parallel training |
| **PPO** | Proximal policy optimization | Most popular, stable |
| **TRPO** | Trust region policy | Theoretical guarantees |
| **SAC** | Soft actor-critic | Continuous control |

### Model-Based Methods

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **Dyna-Q** | Learn world model | Sample efficiency |
| **MuZero** | Learned dynamics | Games, planning |
| **Dreamer** | World models | Visual RL |
| **MBPO** | Model-based policy opt | Continuous control |

## Popular Models/Frameworks

| Name | Type | Notable Achievement |
|------|------|---------------------|
| **DQN** | Value-based | Atari superhuman |
| **AlphaGo/Zero** | MCTS + RL | Beat Go champion |
| **MuZero** | Model-based | Games without rules |
| **PPO** | Policy gradient | RLHF for LLMs |
| **SAC** | Actor-critic | Robot locomotion |
| **IMPALA** | Distributed | Large-scale training |
| **TD3** | Actor-critic | Continuous control |
| **Dreamer** | World model | Visual RL |

## Examples with Code

### Example 1: Q-Learning (Tabular)

```python
import numpy as np
import gymnasium as gym

# Create environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Initialize Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.8    # Learning rate
gamma = 0.95   # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 2000

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state

print("Learned Q-table:")
print(Q)
```

### Example 2: Deep Q-Network (DQN)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gymnasium as gym

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

# Training
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_network = DQN(state_dim, action_dim)
target_network = DQN(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
buffer = ReplayBuffer(10000)

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
gamma = 0.99
batch_size = 64

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0
    
    while True:
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(torch.FloatTensor(state))
                action = q_values.argmax().item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        # Train
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            
            # Compute Q values
            q_values = q_network(states).gather(1, actions.unsqueeze(1))
            
            # Compute target Q values
            with torch.no_grad():
                next_q_values = target_network(next_states).max(1)[0]
                target_q = rewards + gamma * next_q_values * (1 - dones)
            
            # Loss and update
            loss = nn.MSELoss()(q_values.squeeze(), target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if done:
            break
    
    # Update target network
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode}, Reward: {total_reward}")
```

### Example 3: PPO (Policy Gradient)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)
    
    def get_action(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 clip_ratio=0.2, epochs=10):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
    
    def compute_gae(self, rewards, values, dones, next_value, gae_lambda=0.95):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages)
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.stack(old_log_probs)
        
        for _ in range(self.epochs):
            logits, values = self.model(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # PPO clipped objective
            ratio = (log_probs - old_log_probs).exp()
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            entropy_loss = -entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Training loop
env = gym.make('CartPole-v1')
ppo = PPO(env.observation_space.shape[0], env.action_space.n)

for episode in range(500):
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    state, _ = env.reset()
    
    for _ in range(1000):  # Collect trajectory
        state_tensor = torch.FloatTensor(state)
        action, log_prob, value = ppo.model.get_action(state_tensor)
        
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value.item())
        dones.append(done)
        
        state = next_state
        if done:
            break
    
    # Compute returns and advantages
    with torch.no_grad():
        _, next_value = ppo.model(torch.FloatTensor(state))
    
    advantages = ppo.compute_gae(rewards, values, dones, next_value.item())
    returns = advantages + torch.tensor(values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    ppo.update(states, actions, log_probs, returns, advantages)
    print(f"Episode {episode}, Reward: {sum(rewards)}")
```

### Example 4: Stable Baselines3 (Production-Ready)

```python
from stable_baselines3 import PPO, DQN, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym

# Create vectorized environment
env = make_vec_env('CartPole-v1', n_envs=4)

# Train PPO agent
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)

# Callback for evaluation
eval_callback = EvalCallback(
    gym.make('CartPole-v1'),
    best_model_save_path='./logs/',
    log_path='./logs/',
    eval_freq=500,
    deterministic=True
)

# Train
model.learn(total_timesteps=100000, callback=eval_callback)

# Save and load
model.save("ppo_cartpole")
model = PPO.load("ppo_cartpole")

# Evaluate
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```

### Example 5: RLHF for LLMs

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import torch

# Load model with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# PPO config for RLHF
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
)

# Initialize trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

# Training loop (simplified)
def reward_function(response):
    """Your reward model or human feedback"""
    # Example: prefer shorter responses
    return -len(response) / 100

prompts = ["Write a poem about", "Explain quantum physics", "Tell me a joke"]

for epoch in range(10):
    for prompt in prompts:
        # Tokenize
        query_tensor = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate response
        response_tensor = model.generate(query_tensor, max_new_tokens=50)
        response_text = tokenizer.decode(response_tensor[0])
        
        # Compute reward
        reward = torch.tensor([reward_function(response_text)])
        
        # PPO step
        stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], [reward])
```

### Example 6: Multi-Agent RL

```python
from pettingzoo.mpe import simple_spread_v2
from stable_baselines3 import PPO
from supersuit import pad_observations_v0, pad_action_space_v0
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor

# Multi-agent environment
env = simple_spread_v2.parallel_env()
env = pad_observations_v0(env)
env = pad_action_space_v0(env)

# Train each agent
agents = {}
for agent_name in env.possible_agents:
    agents[agent_name] = PPO("MlpPolicy", env, verbose=1)

# Self-play training loop
for iteration in range(100):
    observations, _ = env.reset()
    
    while True:
        actions = {}
        for agent_name, obs in observations.items():
            action, _ = agents[agent_name].predict(obs)
            actions[agent_name] = action
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        if all(terminations.values()) or all(truncations.values()):
            break
```

## RLHF: RL for AI Alignment

| Component | Purpose | Example |
|-----------|---------|---------|
| **Reward Model** | Score responses | Bradley-Terry model |
| **PPO** | Optimize policy | Generate better text |
| **KL Penalty** | Stay near base model | Prevent mode collapse |
| **Reference Model** | Frozen baseline | Original pretrained LLM |

```
RLHF Pipeline:

1. Supervised Fine-tuning (SFT)
   Base Model → Instruction-tuned Model

2. Reward Modeling
   Human preferences → Reward Model

3. PPO Training
   RL Model → Aligned Model
   
   Loss = -reward + β * KL(policy || reference)
```

## Benchmarks

### Atari Games (DQN variants)

| Algorithm | Games > Human | Mean HNS |
|-----------|---------------|----------|
| DQN | 22/57 | 79% |
| Double DQN | 26/57 | 117% |
| Dueling DQN | 28/57 | 121% |
| Rainbow | 42/57 | 223% |

### Continuous Control (MuJoCo)

| Algorithm | HalfCheetah | Ant | Humanoid |
|-----------|-------------|-----|----------|
| PPO | 1800 | 3000 | 600 |
| SAC | 11000 | 5500 | 5200 |
| TD3 | 9500 | 4700 | 5100 |

## When to Use RL

| Scenario | Use RL | Alternative |
|----------|--------|-------------|
| Sequential decisions | ✅ Yes | - |
| Delayed rewards | ✅ Yes | - |
| Sparse rewards | ✅ Yes (with shaping) | Imitation learning |
| Clear reward function | ✅ Yes | - |
| No reward function | Inverse RL | Imitation learning |
| Lots of data available | Consider | Supervised learning |
| Real-time systems | Offline RL | Model predictive control |
| LLM alignment | ✅ RLHF | DPO, Constitutional AI |

## RL vs Other Paradigms

| Aspect | RL | Supervised | Unsupervised |
|--------|-----|-----------|--------------|
| **Feedback** | Rewards | Labels | None |
| **Goal** | Maximize reward | Minimize error | Find patterns |
| **Data** | Experience | Dataset | Dataset |
| **Sequential** | ✅ Yes | No | No |
| **Exploration** | ✅ Required | No | No |

## Common Libraries

| Library | Features | Best For |
|---------|----------|----------|
| **Stable Baselines3** | Production-ready | Standard RL |
| **RLlib** | Distributed | Large-scale |
| **CleanRL** | Single-file | Learning |
| **TRL** | RLHF | LLM alignment |
| **Tianshou** | Modular | Research |
| **RL Games** | GPU-accelerated | Robotics |

## Related Models

- **[LLM](./LLM.md)** - RLHF for alignment
- **[World Models](./WORLD_MODELS.md)** - Model-based RL
- **[GNN](./GNN.md)** - Graph-based RL environments

## Resources

- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [RL Course by David Silver](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)
- [TRL for RLHF](https://huggingface.co/docs/trl/)
- [Gymnasium](https://gymnasium.farama.org/)
