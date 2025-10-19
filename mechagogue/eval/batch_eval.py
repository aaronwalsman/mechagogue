'''
Batched model eval utilities.

Evaluates models on datasets by splitting into batches,
padding as needed, and computing weighted mean metrics.
'''

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.tree import pad_tree_batch_size, batch_tree
from mechagogue.standardize import standardize_args
from mechagogue.nn.layer import standardize_layer

def batch_evaluator(
    model,
    evaluate,
    batch_size,
):
    
    model = standardize_layer(model)
    evaluate = standardize_args(evaluate, ('pred', 'y', 'mask'))
    
    def batch_eval(key, x, y, model_state):
        (x, y), valid = pad_tree_batch_size((x,y), batch_size)
        x, y, valid = batch_tree((x,y,valid), batch_size)
        
        def eval_step(mean_total, key_x_y_valid):
            mean, total = mean_total
            key, x, y, valid = key_x_y_valid
            pred = model.forward(key, x, model_state)
            
            step_mean = evaluate(pred, y, valid)
            new_evals = jnp.sum(valid)
            
            new_total = total + new_evals
            a = total / new_total
            b = new_evals / new_total
            mean = mean * a + step_mean * b
            return (mean, new_total), None
        
        num_batches = valid.shape[0]
        step_keys = jrng.split(key, num_batches)
        z = jnp.array(0)
        (mean_eval, _), _ = jax.lax.scan(
            eval_step, (z,z), (step_keys, x, y, valid))
        
        return mean_eval
    
    return batch_eval
