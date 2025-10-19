# Supervised Learning (SUP) Framework

Training algorithms for supervised learning tasks using both gradient-based and evolutionary approaches.

## Training Algorithms

### **`supervised_backprop.py`**
Standard supervised learning with backpropagation and gradient descent.

### **`ga.py`**
Genetic Algorithm for supervised learning without gradients.

**Configuration (`GAParams`):**
- `population_size` - Number of individuals in population
- `batch_size` - Training batch size
- `batches_per_step` - Batches per generation
- `elites` - Number of top performers to preserve
- `dunces` - Number of worst performers to remove
- `share_keys` - Share random keys across population

## Task Utilities

### **`tasks/classify.py`**
Classification-specific utilities.

## Algorithm Comparison

| Feature | Backpropagation | Genetic Algorithm |
|---------|----------------|-------------------|
| Optimization | Gradient-based | Fitness-based |
| Differentiability | Required | Not required |
| Memory | Single model | Full population |
| Exploration | Local (gradient) | Global (breeding) |
| Convergence | Faster | Slower |
| Architecture | Fixed | Can evolve |
