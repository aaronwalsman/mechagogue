import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static_dataclass import static_dataclass
from mechagogue.tree import (
    tree_len, shuffle_tree, pad_tree_batch_size, batch_tree, tree_getitem)
from mechagogue.arg_wrappers import ignore_unused_args
from mechagogue.population import population_model
from mechagogue.eval.batch_eval import batch_evaluator

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
    breed,
    loss_function,
    test_function,
):
    
    # wrap the provided functions
    init_model_params = ignore_unused_args(init_model_params,
        ('key',))
    model = ignore_unused_args(model,
        ('key', 'x', 'params'))
    step_population = population_model(breed)
    loss_function = ignore_unused_args(loss_function,
        ('pred', 'y', 'mask'))
    
    evaluator = batch_evaluator(model, test_function, config.batch_size)
    population_evaluator = jax.vmap(evaluator, in_axes=(0, None, None, 0))
    
    def init(key):
        model_keys = jrng.split(key, config.population_size)
        model_params = jax.vmap(init_model_params)(model_keys)
        return model_params
    
    def train(key, x, y, model_params):
        
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
        
        def train_batch(model_params, key_x_y_mask):
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
                jnp.zeros(config.population_size),
                (step_keys, x, y, mask),
            )
            
            # compute elites and dunces
            _, elite_ids = jax.lax.top_k(fitness, config.elites)
            _, non_dunce_ids = jax.lax.top_k(
                fitness, config.population_size-config.dunces)
            
            key, children_key, population_key = jrng.split(key, 3)
            alive = jnp.zeros(config.population_size, dtype=jnp.bool)
            alive = alive.at[elite_ids].set(True)
            num_children = config.population_size-config.elites
            children = jrng.choice(
                children_key, non_dunce_ids, shape=(num_children, 1))
            _, model_params = step_population(
                population_key, alive, children, model_params)
            
            return model_params, fitness
        
        num_batches = tree_len(x)
        batch_keys = jrng.split(key, num_batches)
        model_params, fitness = jax.lax.scan(
            train_batch, model_params,(batch_keys, x, y, mask))
        return model_params, fitness
    
    def test(key, x, y, model_params):
        num_models = tree_len(model_params)
        keys = jrng.split(key, num_models)
        return population_evaluator(keys, x, y, model_params)
    
    return init, train, test
