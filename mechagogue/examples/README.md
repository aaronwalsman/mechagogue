# Mechagogue Examples

This directory contains example scripts demonstrating various training paradigms and algorithms available in Mechagogue.

Examples use JAX and are designed to be JIT-compiled for performance.

## Supervised Learning

- **`sup_example.py`** - Basic supervised learning with SUP framework using SGD on synthetic classification data

## Genetic Algorithms

- **`ga_example.py`** - Genetic algorithm training on synthetic classification data with elite selection

## Reinforcement Learning

### Policy Gradient Methods
- **`vpg_test.py`** - VPG test on simple 2D navigation task with uniform random policy
- **`nom_vpg.py`** - VPG training on Nom foraging environment

### Deep Q-Networks
- **`count_up.py`** - DQN and VPG training examples on count-up task

## Evolutionary Algorithms

- **`population_test.py`** - Multi-site population evolution simulation with natural selection dynamics

## Specialized Examples

### MaxNIST
Training scripts for multi-digit classification on the MaxNIST dataset.

See [`maxnist/README.md`](maxnist/README.md) for details.

### MaxAtar (Breakout)
Example scripts for DQN agent training in the Breakout environment.

See [`maxatar/README.md`](maxatar/README.md) for details.
