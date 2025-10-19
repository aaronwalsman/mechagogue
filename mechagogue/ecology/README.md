# Ecology Framework

This directory contains components for simulating evolving populations in ecological environments with natural selection, reproduction, and trait inheritance.

## Core Components

### **`natural_selection.py`**
Natural selection algorithm for population evolution simulations. Key features:
- Simulates populations of agents forward through time
- No explicit optimization objective - relies on environmental dynamics
- Tracks family relationships (parents/children) across generations
- Supports state saving/loading for large populations
- Agents compete, reproduce, and evolve based on fitness in the environment

### **`policy.py`**
Policy and population abstractions for ecological agents. Provides:
- **Individual policy interface**: Single agent behavior with actions, adaptation, and traits
- **Vectorized population interface**: Manages collections of agents efficiently
- **Breeding mechanics**: Trait inheritance from parents to children
- **Standardization utilities**: Ensures consistent interfaces across implementations

## Key Concepts

- **Natural Selection**: Population dynamics driven by environment, not gradient descent
- **Breeding**: Mechanism for trait inheritance (mutation, crossover, etc.)
- **Active Players**: Dynamic population size as agents die and reproduce
- **Traits**: Observable characteristics that affect environment interactions
- **Adaptation**: Within-lifetime learning or behavioral adjustments
