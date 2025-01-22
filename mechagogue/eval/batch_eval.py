import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.tree import pad_tree_batch_size, batch_tree

def batch_eval(
    key,
    model,
    params,
    evaluate,
    batch_size,
    x,
    y,
):
    (x, y), valid = pad_tree_batch_size((x,y), batch_size)
    x, y, valid = batch_tree((x,y,valid), batch_size)
    
    def eval_step(mean_total, key_x_y_valid):
        mean, total = mean_total
        key, x, y, valid = key_x_y_valid
        #model_key, eval_key = jrng.split(key)
        pred = model(key, x, params)
        step_mean = evaluate(pred, y, valid)
        
        #jax.debug.print('m {m} v {v}', m=step_mean, v=valid)
        
        #step_mean = jnp.mean(batch_eval * valid, axis=-1)
        new_evals = jnp.sum(valid)
        
        new_total = total + new_evals
        a = total / new_total
        b = new_evals / new_total
        mean = mean * a + step_mean * b
        return (mean, new_total), None
    
    num_batches = valid.shape[0]
    step_keys = jrng.split(key, num_batches)
    z = jnp.array(0)
    (mean_eval, _), _ = jax.lax.scan(eval_step, (z,z), (step_keys, x, y, valid))
    
    return mean_eval
