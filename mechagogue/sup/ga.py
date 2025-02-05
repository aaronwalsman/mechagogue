import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static_dataclass import static_dataclass
from mechagogue.tree import (
    tree_len, shuffle_tree, pad_tree_batch_size, batch_tree, tree_getitem)
from mechagogue.arg_wrappers import ignore_unused_args
from mechagogue.population import population_step
from mechagogue.eval.batch_eval import batch_eval

@static_dataclass
class GAConfig:
    population_size: int = 256
    batch_size: int = 256
    shuffle=True,
    batches_per_step: int = 1
    elites: int = 1
    dunces: int = 0

def ga(
    config,
    init_model_params,
    model,
    init_mutate_params,
    mutate,
    loss_function,
    test_function,
):
    
    # wrap the provided functions
    init_model_params = ignore_unused_args(init_model_params,
        ('key',))
    model = ignore_unused_args(model,
        ('key', 'x', 'params'))
    init_mutate_params = ignore_unused_args(init_mutate_params,
        ('key', 'model_params'))
    mutate = ignore_unused_args(mutate,
        ('key', 'model_params', 'mutate_params'))
    loss_function = ignore_unused_args(loss_function,
        ('pred', 'y', 'mask'))
    test_function = ignore_unused_args(test_function,
        ('pred', 'y', 'mask'))
    
    def init(key):
        model_key, mutate_key = jrng.split(key)
        model_keys = jrng.split(model_key, config.population_size)
        model_params = jax.vmap(init_model_params)(model_keys)
        mutate_params = init_mutate_params(mutate_key, model_params)
        return model_params, mutate_params
    
    def train(key, x, y, model_params, mutate_params):
        population_size = tree_len(model_params)
        
        # shuffle and batch the data
        if config.shuffle:
            key, shuffle_key = jrng.split(key)
            x, y = shuffle_tree(shuffle_key, (x, y))
        (x, y), mask = pad_tree_batch_size(
            (x, y), config.batch_size * config.batches_per_step)
        x, y, mask = batch_tree(
            (x, y, mask), config.batch_size * config.batches_per_step)
        x, y, mask = batch_tree(
            (x, y, mask), config.batch_size, axis=1)
        
        def train_batch(params, key_x_y_mask):
            model_params, mutate_params = params
            key, x, y, mask = key_x_y_mask
            
            def loss_step(fitness, key_x_y_mask):
                key, x, y, mask = key_x_y_mask
                x = x[None, ...]
                model_key, loss_key = jrng.split(key)
                pred = model(model_key, x, model_params)
                loss = jax.vmap(loss_function, in_axes=(0,None,None))(
                    pred, y, mask)
                fitness = fitness - loss
                
                return fitness, None
            
            step_keys = jrng.split(key, config.batches_per_step+1)
            key, step_keys = step_keys[0], step_keys[1:]
            fitness, _ = jax.lax.scan(
                loss_step,
                jnp.zeros(population_size),
                (step_keys, x, y, mask),
            )
            
            # compute elites and dunces
            _, elite_ids = jax.lax.top_k(fitness, config.elites)
            _, non_dunce_ids = jax.lax.top_k(
                fitness, config.population_size-config.dunces)
            
            key, children_key, population_key = jrng.split(key, 3)
            survived = jnp.zeros(population_size, dtype=jnp.bool)
            survived = survived.at[elite_ids].set(True)
            num_children = config.population_size-config.elites
            children = jrng.choice(
                children_key, non_dunce_ids, shape=(num_children, 1))
            _, model_params, mutate_params = population_step(
                population_key,
                survived,
                children,
                model_params,
                mutate,
                mutate_params,
            )
            
            return (model_params, mutate_params), fitness
        
        num_batches = tree_len(x)
        batch_keys = jrng.split(key, num_batches)
        (model_params, mutate_params), fitness = jax.lax.scan(
            train_batch,
            (model_params, mutate_params),
            (batch_keys, x, y, mask),
        )
        return model_params, mutate_params, fitness
    
    def test(key, x, y, model_params):
        population_size = tree_len(model_params)
        keys = jrng.split(key, population_size)
        multi_eval =  jax.vmap(
            batch_eval, in_axes=(0, None, 0, None, None, None, None))
        return multi_eval(
            keys, model, model_params, test_function, config.batch_size, x, y)
    
    return init, train, test
