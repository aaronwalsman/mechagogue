import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static_dataclass import static_dataclass
from mechagogue.tree import (
    tree_len, shuffle_tree, pad_tree_batch_size, batch_tree)
from mechagogue.arg_wrappers import ignore_unused_args
from mechagogue.eval.batch_eval import batch_evaluator

@static_dataclass
class SupParams:
    batch_size: int = 64
    shuffle: bool = True

def sup(
    params,
    init_model,
    model,
    init_optim,
    optim,
    loss_function,
    test_function,
):
    
    # wrap the provided functions
    init_model = ignore_unused_args(init_model,
        ('key',))
    model = ignore_unused_args(model,
        ('key', 'x', 'state'))
    init_optim = ignore_unused_args(init_optim,
        ('key', 'model_state'))
    optim = ignore_unused_args(optim,
        ('key', 'grad', 'model_state', 'optim_state'))
    loss_function = ignore_unused_args(loss_function,
        ('pred', 'y', 'mask'))
    test_function = ignore_unused_args(test_function,
        ('pred', 'y', 'mask'))
    
    def init(key):
        model_key, optim_key = jrng.split(key)
        model_state = init_model(model_key)
        optim_state = init_optim(optim_key, model_state)
        return model_state, optim_state
    
    def train(key, x, y, model_state, optim_state):
        if params.shuffle:
            key, shuffle_key = jrng.split(key)
            x, y = shuffle_tree(shuffle_key, (x, y))
        (x, y), mask = pad_tree_batch_size((x, y), params.batch_size)
        x, y, mask = batch_tree((x, y, mask), params.batch_size)
        
        def train_batch(model_optim_state, key_x_y_mask):
            model_state, optim_state = model_optim_state
            key, x, y, mask = key_x_y_mask
            
            def forward(key, x, y, mask, model_state):
                pred = model(key, x, model_state)
                loss = loss_function(pred, y, mask)
                return loss
            
            forward_key, optim_key = jrng.split(key)
            loss, grad = jax.value_and_grad(forward, argnums=4)(
                forward_key, x, y, mask, model_state)
            model_state, optim_state = optim(
                optim_key, grad, model_state, optim_state)
            
            return (model_state, optim_state), loss
        
        num_batches = tree_len(x)
        batch_keys = jrng.split(key, num_batches)
        (model_state, optim_state), losses = jax.lax.scan(
            train_batch,
            (model_state, optim_state),
            (batch_keys, x, y, mask),
        )
        return model_state, optim_state, losses
    
    test = batch_evaluator(model, test_function, params.batch_size)
    
    return init, train, test
