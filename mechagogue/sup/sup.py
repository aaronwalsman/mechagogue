import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.tree import (
    tree_len, shuffle_tree, pad_tree_batch_size, batch_tree)
from mechagogue.arg_wrappers import ignore_unused_args
from mechagogue.eval.batch_eval import batch_eval

def sup(
    init_model_params,
    model,
    init_optimizer_params,
    optimize,
    loss_function,
    test_function,
    batch_size,
    shuffle=True,
):
    
    # wrap the provided functions
    init_model_params = ignore_unused_args(init_model_params,
        ('key',))
    model = ignore_unused_args(model,
        ('key', 'x', 'params'))
    init_optimizer_params = ignore_unused_args(init_optimizer_params,
        ('key', 'model_params'))
    optimize = ignore_unused_args(optimize,
        ('key', 'grad', 'model_params', 'optimizer_params'))
    loss_function = ignore_unused_args(loss_function,
        ('pred', 'y', 'mask'))
    test_function = ignore_unused_args(test_function,
        ('pred', 'y', 'mask'))
    
    def init(key):
        model_key, optimizer_key = jrng.split(key)
        model_params = init_model_params(model_key)
        optimizer_params = init_optimizer_params(optimizer_key, model_params)
        return model_params, optimizer_params
    
    def train(key, x, y, model_params, optimizer_params):
        if shuffle:
            key, shuffle_key = jrng.split(key)
            x, y = shuffle_tree(shuffle_key, (x, y))
        (x, y), mask = pad_tree_batch_size((x, y), batch_size)
        x, y, mask = batch_tree((x, y, mask), batch_size)
        
        def train_batch(params, key_x_y_mask):
            model_params, optimizer_params = params
            key, x, y, mask = key_x_y_mask
            
            def forward(key, x, y, mask, model_params):
                pred = model(key, x, model_params)
                loss = loss_function(pred, y, mask)
                return loss
            
            forward_key, optimizer_key = jrng.split(key)
            loss, grad = jax.value_and_grad(forward, argnums=4)(
                forward_key, x, y, mask, model_params)
            model_params, optimizer_params = optimize(
                optimizer_key, grad, model_params, optimizer_params)
            
            return (model_params, optimizer_params), loss
        
        num_batches = tree_len(x)
        batch_keys = jrng.split(key, num_batches)
        (model_params, optimizer_params), losses = jax.lax.scan(
            train_batch,
            (model_params, optimizer_params),
            (batch_keys, x, y, mask),
        )
        return model_params, optimizer_params, losses
    
    def test(key, x, y, model_params):
        return batch_eval(
            key, model, model_params, test_function, batch_size, x, y)
    
    return init, train, test
