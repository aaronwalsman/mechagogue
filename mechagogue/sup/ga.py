import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static import static_data, static_functions
from mechagogue.tree import (
    tree_len,
    shuffle_tree,
    pad_tree_batch_size,
    batch_tree,
    tree_getitem,
    tree_setitem,
)
from mechagogue.standardize import standardize_args
from mechagogue.eval.batch_eval import batch_evaluator
from mechagogue.nn.layer import standardize_layer

@static_data
class GAParams:
    population_size: int = 256
    batch_size: int = 256
    shuffle=True,
    batches_per_step: int = 1
    elites: int = 1
    dunces: int = 0
    share_keys: bool = False

def make_ga(
    params,
    #init_model,
    #model,
    model,
    breed,
    loss_function,
    test_function,
):
    
    # wrap the provided functions
    #init_model = standardize_args(init_model, ('key',))
    #init_model = jax.vmap(init_model)
    if params.share_keys:
        parallel_model_axes = (None, None, 0)
    else:
        parallel_model_axes = (0, None, 0)
    model = standardize_layer(model)
    init_model = jax.vmap(model.init)
    parallel_model = jax.vmap(model.forward, in_axes=parallel_model_axes)
    # need to figure out how to vmap post ignore_unused_args
    #parallel_model = jax.vmap(model, in_axes=parallel_model_axes)
    #model = standardize_args(model, ('key', 'x', 'state'))
    breed = standardize_args(breed, ('key', 'state'))
    breed = jax.vmap(breed)
    loss_function = standardize_args(loss_function, ('pred', 'y', 'mask', 'model_state'))
    
    evaluator = batch_evaluator(model, test_function, params.batch_size)
    population_evaluator = jax.vmap(evaluator, in_axes=(0, None, None, 0))
    
    @static_functions
    class GA:
        def init(key):
            model_keys = jrng.split(key, params.population_size)
            model_state = init_model(model_keys)
            return model_state
        
        def train(key, x, y, model_state):
            
            # shuffle and batch the data
            if params.shuffle:
                key, shuffle_key = jrng.split(key)
                x, y = shuffle_tree(shuffle_key, (x, y))
            (x, y), mask = pad_tree_batch_size(
                (x, y), params.batch_size * params.batches_per_step)
            x, y, mask = batch_tree(
                (x, y, mask), params.batch_size * params.batches_per_step)
            x, y, mask = batch_tree(
                (x, y, mask), params.batch_size, axis=1)
            
            def train_batch(model_state, key_x_y_mask):
                key, x, y, mask = key_x_y_mask
                
                def loss_step(fitness, key_x_y_mask):
                    key, x, y, mask = key_x_y_mask
                    #x = x[None, ...]
                    model_key, loss_key = jrng.split(key)
                    if params.share_keys:
                        pred = parallel_model(model_key, x, model_state)
                    else:
                        model_keys = jrng.split(
                            model_key, params.population_size)
                        pred = parallel_model(model_keys, x, model_state)
                    loss = jax.vmap(loss_function, in_axes=(0,None,None,0))(
                        pred, y, mask, model_state)
                    fitness = fitness - loss
                    
                    return fitness, None
                
                step_keys = jrng.split(key, params.batches_per_step+1)
                key, step_keys = step_keys[0], step_keys[1:]
                fitness, _ = jax.lax.scan(
                    loss_step,
                    jnp.zeros(params.population_size),
                    (step_keys, x, y, mask),
                )
                
                # compute elites and dunces
                _, elite_ids = jax.lax.top_k(fitness, params.elites)
                _, non_dunce_ids = jax.lax.top_k(
                    fitness, params.population_size-params.dunces)
                
                key, parent_key, breed_key = jrng.split(key, 3)
                #alive = jnp.zeros(params.population_size, dtype=jnp.bool)
                #alive = alive.at[elite_ids].set(True)
                num_children = params.population_size-params.elites
                parents = jrng.choice(
                    parent_key, non_dunce_ids, shape=(num_children, 1))
                breed_keys = jrng.split(breed_key, num_children)
                parent_state = tree_getitem(model_state, parents)
                child_state = breed(breed_keys, parent_state)
                elite_state = tree_getitem(model_state, elite_ids)
                model_state = tree_setitem(
                    model_state, jnp.arange(params.elites), elite_state)
                model_state = tree_setitem(
                    model_state,
                    jnp.arange(params.elites, params.population_size),
                    child_state,
                )
                #_, model_state = step_population(
                #    population_key, alive, children, model_state)
                
                return model_state, fitness
            
            num_batches = tree_len(x)
            batch_keys = jrng.split(key, num_batches)
            model_state, fitness = jax.lax.scan(
                train_batch, model_state,(batch_keys, x, y, mask))
            return model_state, fitness
        
        def test(key, x, y, model_state):
            num_models = tree_len(model_state)
            keys = jrng.split(key, num_models)
            return population_evaluator(keys, x, y, model_state)
    
    return GA
