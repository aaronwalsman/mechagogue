'''
Adam optimizer with bias correction for momentum and velocity estimates.
'''

from typing import Any

import jax
import jax.numpy as jnp

from mechagogue.static_dataclass import static_data, static_functions

@static_data
class AdamState:
    momentum : Any = None
    velocity : Any = None
    t : int = 0

def adam(
    learning_rate=3e-4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    @static_functions
    class Adam:
        def init(model_state):
            momentum = jax.tree.map(jnp.zeros_like, model_state)
            velocity = jax.tree.map(jnp.zeros_like, model_state)
            return AdamState(momentum, velocity, 0)
        
        def optimize(grad, model_state, optim_state):
            t = optim_state.t + 1
            momentum = jax.tree.map(
                lambda m, g: beta1 * m + (1 - beta1) * g,
                optim_state.momentum,
                grad,
            )
            velocity = jax.tree.map(
                lambda v, g: beta2 * v + (1 - beta2) * (g ** 2),
                optim_state.velocity,
                grad,
            )
            
            momentum_hat = jax.tree.map(
                lambda m: m / (1 - beta1 ** t), momentum)
            velocity_hat = jax.tree.map(
                lambda v: v / (1 - beta2 ** t), velocity)
            
            updates = jax.tree.map(
                lambda m_hat, v_hat: m_hat / (jnp.sqrt(v_hat) + epsilon),
                momentum_hat,
                velocity_hat,
            )
            model_state = jax.tree.map(
                lambda p, u: p - learning_rate * u,
                model_state,
                updates,
            )
            
            optim_state = optim_state.replace(
                momentum=momentum,
                velocity=velocity,
                t=t,
            )
            return model_state, optim_state
        
    return Adam
