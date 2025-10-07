# Reinforcement Learning (RL) Algorithms

Deep reinforcement learning algorithms for training agents in interactive environments.

## Algorithms

### **`dqn.py`**
Deep Q-Network with experience replay and target networks.

**Configuration (`DQNConfig`):**
- `batch_size` - Training batch size
- `parallel_envs` - Number of parallel environments
- `replay_buffer_size` - Maximum replay buffer capacity
- `discount` - Discount factor (gamma)
- `epsilon` - Exploration rate (annealed over time)
- `target_update_frequency` - Steps between target network updates

**Best for:** Discrete action spaces, sample-efficient learning from replay

### **`vpg.py`**
Vanilla Policy Gradient for direct policy optimization.

**Configuration (`VPGConfig`):**
- `parallel_envs` - Number of parallel environments
- `rollout_steps` - Steps per rollout before training
- `training_epochs` - Epochs to train on each batch of data
- `discount` - Discount factor (gamma)
- `batch_size` - Training batch size

**Best for:** Continuous or discrete action spaces, learning from on-policy data
