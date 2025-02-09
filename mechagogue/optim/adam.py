from typing import Any

import jax
import jax.numpy as jnp

from mechagogue.static_dataclass import static_dataclass

@static_dataclass
class AdamState:
    momentum : Any = None
    velocity : Any = None
    learning_rate : float = 3e-4
    beta1 : float = 0.9
    beta2 : float = 0.999
    epsilon : float = 1e-8
    t : int = 0

def adam(
    learning_rate=3e-4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    def init(model_state):
        momentum = jax.tree.map(jnp.zeros_like, model_state)
        velocity = jax.tree.map(jnp.zeros_like, model_state)
        return AdamState(
            momentum, velocity, learning_rate, beta1, beta2, epsilon, 0)
    
    def optim(grad, model_state, optim_state):
        t = optim_state.t + 1
        momentum = jax.tree.map(
            lambda m, g: optim_state.beta1 * m + (1 - optim_state.beta1) * g,
            optim_state.momentum,
            grad,
        )
        velocity = jax.tree.map(
            lambda v, g: (
                optim_state.beta2 * v + (1 - optim_state.beta2) * (g ** 2)),
            optim_state.velocity,
            grad,
        )
        
        momentum_hat = jax.tree.map(
            lambda m: m / (1 - optim_state.beta1 ** t), momentum)
        velocity_hat = jax.tree.map(
            lambda v: v / (1 - optim_state.beta2 ** t), velocity)
        
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
    
    return init, optim
