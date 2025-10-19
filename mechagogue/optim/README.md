# Optimizers

Gradient-based optimization algorithms for neural network training.

## Available Optimizers

### **`sgd.py`**
Stochastic Gradient Descent with optional momentum, Nesterov acceleration, and weight decay.

**Parameters:**
- `learning_rate` - Step size for parameter updates
- `momentum` - Momentum coefficient (0 = no momentum)
- `damping` - Damping factor for momentum
- `nesterov` - Enable Nesterov accelerated gradient
- `weight_decay` - L2 regularization strength

### **`adam.py`**
Adam optimizer with adaptive learning rates and bias correction.

**Parameters:**
- `learning_rate` - Step size (default: 3e-4)
- `beta1` - Exponential decay for first moment (default: 0.9)
- `beta2` - Exponential decay for second moment (default: 0.999)
- `epsilon` - Numerical stability term (default: 1e-8)

### **`adamw.py`**
AdamW with decoupled weight decay regularization (more effective than L2 for Adam).

**Parameters:**
- Same as Adam, plus:
- `weight_decay` - Weight decay coefficient (default: 1e-2)

### **`rmsprop.py`**
RMSProp with adaptive per-parameter learning rates based on gradient magnitude.

**Parameters:**
- `learning_rate` - Step size
- `alpha` - Smoothing constant for squared gradient (default: 0.95)
- `eps` - Numerical stability term (default: 1e-2)
- `centered` - Use centered variant (default: True)

## Interface

All optimizers return an object with two static methods:

- **`init(model_state)`** - Initialize optimizer state from model parameters
- **`optimize(grad, model_state, optim_state)`** - Apply gradient update and return updated parameters and optimizer state
