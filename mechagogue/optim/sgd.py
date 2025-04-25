import jax
import jax.numpy as jnp

def sgd(
    learning_rate=3e-4,
    momentum=0,
    damping=0.,
    nesterov=False,
    weight_decay=0,
):
    def init(key, model_state):
        def init_leaf(leaf):
            if momentum:
                return jnp.zeros_like(leaf)
            else:
                return None
        return jax.tree.map(init_leaf, model_state)
    
    def optim(grad, model_state, optim_state):
        if weight_decay:
            def leaf_weight_decay(leaf_grad, leaf_model_param):
                return leaf_grad + leaf_model_param * weight_decay
            grad = jax.tree.map(leaf_weight_decay, grad, model_state)
        
        # TODO: is this right?
        if momentum:
            def leaf_momentum(leaf_grad, leaf_param):
                return (leaf_param * momentum + leaf_grad * (1.-damping))
            optim_state = jax.tree.map(leaf_momentum, grad, optim_state)
        
            velocity = optim_state
        else:
            velocity = grad
        
        if nesterov:
            def leaf_nesterov(leaf_grad, leaf_velocity, leaf_param):
                return leaf_grad + leaf_velocity * momentum
            model_update = jax.tree.map(
                leaf_nesterov, grad, velocity, optim_state)
        else:
            model_update = velocity
        
        def apply_leaf_update(leaf_model_param, leaf_update):
            return leaf_model_param - leaf_update * learning_rate
        
        model_state = jax.tree.map(
            apply_leaf_update, model_state, model_update)
        
        return model_state, optim_state
    
    return init, optim
