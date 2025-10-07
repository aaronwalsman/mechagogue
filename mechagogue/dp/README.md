# Decision Process (DP) Framework

This directory contains environment builders for various decision process formulations, from simple bandits to complex multi-agent games.

## Single-Agent Environments

### **`bandit.py`**
Basic multi-armed bandit environment with stochastic rewards.

### **`contextual_bandit.py`**
Contextual bandit where actions depend on observable context/state.

### **`pomdp.py`**
Partially Observable Markov Decision Process (POMDP) framework. Builds environments from component functions:
- State transitions
- Observations
- Rewards
- Terminal conditions

### **`mdp.py`**
Markov Decision Process (MDP) - a fully observable special case of POMDP where observations equal state.

## Multi-Agent Environments

### **`turn_game.py`**
Turn-based multi-player game environment with configurable player order and observations.

### **`population_game.py`**
Population game builder for evolutionary simulations with:
- Active player tracking
- Family relationships between generations
- Population dynamics

### **`poeg.py`**
Partially Observable Ecological Game - extends population games with ecological dynamics and trait inheritance.

### **`others.py`**
Additional game formulations (POSG, Dec-POMDP) that share the POMDP functional structure.

## Usage

All environments follow a consistent builder pattern:

```python
# Single-agent example (MDP)
from mechagogue.dp.mdp import mdp

reset_env, step_env = mdp(
    init_state=...,
    transition=...,
    reward=...,
    terminal=...
)

# Multi-agent example (Population Game)
from mechagogue.dp.population_game import population_game

reset_env, step_env = population_game(
    init_state=...,
    transition=...,
    observe=...,
    active_players=...,
    family_info=...
)
```

The builders standardize function signatures and handle random key management automatically.
