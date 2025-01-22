import jax
import jax.numpy as jnp

def sgd(
    learning_rate=3e-4,
    momentum=0,
    damping=0.,
    nesterov=False,
    weight_decay=0,
):
    def init_sgd(model_params):
        def init_leaf(leaf):
            if momentum:
                return jnp.zeros_like(leaf)
            else:
                return None
        return jax.tree.map(init_leaf, model_params)
    
    def optimize_sgd(grad, model_params, optimizer_params):
        def optimize_leaf(leaf_grad, leaf_model_param, leaf_optimizer_param):
            if weight_decay:
                leaf_grad = leaf_grad + leaf_model_params * weight_decay
            
            if momentum:
                leaf_optimizer_param = (
                    leaf_optimizer_param * momentum + leaf_grad * (1.-damping))
                leaf_velocity = leaf_optimizer_param
            else:
                leaf_velocity = leaf_grad
            
            if nesterov:
                leaf_update = leaf_grad + leaf_velocity * momentum
            else:
                leaf_update = leaf_velocity
            
            leaf_model_param = leaf_model_param - leaf_update * learning_rate
            
            return leaf_model_param, leaf_optimizer_param
        
        joined_tree = jax.tree.map(
            optimize_leaf, grad, model_params, optimizer_params)
        model_params, optimizer_params = jax.tree.transpose(
            jax.tree.structure(grad),
            jax.tree.structure(('*', '*')),
            joined_tree,
        )
        
        return model_params, optimizer_params
    
    return init_sgd, optimize_sgd
