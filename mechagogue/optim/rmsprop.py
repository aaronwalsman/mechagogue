import jax
import jax.numpy as jnp
from typing import Any, Tuple

def rmsprop(
    learning_rate: float,
    alpha: float = 0.95,
    eps: float = 1e-2,
    centered: bool = True
):
    """
        RMSProp optimizer.

        Args:
            learning_rate: step size (alpha in MinAtar code).
            alpha: smoothing constant for squared gradient (MinAtar's SQUARED_GRAD_MOMENTUM).
            eps: term added to denominator for numerical stability (MinAtar's MIN_SQUARED_GRAD).
            centered: if True, uses centered RMSProp.

        Returns:
            init_fn: function(key, model_state) -> optim_state
            update_fn: function(key, grad, model_state, optim_state) -> (new_model_state, new_optim_state)
    """
    def init_fn(key: Any, model_state: Any) -> Any:
        # Initialize accumulators with the same structure as parameters
        sq_avg = jax.tree.map(lambda w: jnp.zeros_like(w), model_state)
        if centered:
            grad_avg = jax.tree.map(lambda w: jnp.zeros_like(w), model_state)
            return (sq_avg, grad_avg)
        else:
            return (sq_avg,)

    def update_fn(
        key: Any,
        grad: Any,
        model_state: Any,
        optim_state: Any
    ) -> Tuple[Any, Any]:
        # Unpack optimizer state
        sq_avg = optim_state[0]
        sq_avg_new = jax.tree.map(
            lambda v, g: alpha * v + (1.0 - alpha) * (g * g),
            sq_avg,
            grad,
        )

        if centered:
            grad_avg = optim_state[1]
            grad_avg_new = jax.tree.map(
                lambda m, g: alpha * m + (1.0 - alpha) * g,
                grad_avg,
                grad,
            )
            # variance = E[g^2] - (E[g])^2
            var = jax.tree.map(lambda v_n, m_n: v_n - m_n * m_n, sq_avg_new, grad_avg_new)
            # denom = sqrt(variance + eps)
            denom = jax.tree.map(lambda v: jnp.sqrt(v + eps), var)
            updates = jax.tree.map(lambda g, d: g / d, grad, denom)
            # parameter update
            new_params = jax.tree.map(lambda p, u: p - learning_rate * u, model_state, updates)
            return new_params, (sq_avg_new, grad_avg_new)
        else:
            # denom = sqrt(E[g^2] + eps)
            denom = jax.tree.map(lambda v: jnp.sqrt(v + eps), sq_avg_new)
            updates = jax.tree.map(lambda g, d: g / d, grad, denom)
            new_params = jax.tree.map(lambda p, u: p - learning_rate * u, model_state, updates)
            return new_params, (sq_avg_new,)

    return init_fn, update_fn
