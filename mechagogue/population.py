import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.arg_wrappers import ignore_unused_args, split_random_keys
from mechagogue.tree import tree_len,tree_len,  tree_getitem, tree_setitem

raise Exception('DEPRECATED')

def population_model_old(
    breed,
    children_per_chunk=None,
    empty=-1,
):
    
    breed = ignore_unused_args(breed, ('key', 'params'))
    # TODO: vmapped random function
    breed = split_random_keys(jax.vmap(breed))
    
    def step(key, alive, children, population):
        population_size = tree_len(population)
        num_children, parents_per_child = children.shape
        
        parent_data = tree_getitem(population, children)
        breed_keys = jrng.split(key, num_children)
        child_data = jax.vmap(breed)(breed_keys, parent_data)
        write_locations, = jnp.nonzero(
            ~alive, size=num_children, fill_value=population_size)
        alive = alive.at[write_locations].set(True)
        population = tree_setitem(population, write_locations, child_data)
        
        return alive, population
    
    def chunk_step(key, alive, children, population):
        def cond(k_a_c_pd):
            _, _, children,  _ = k_a_c_pd
            return jnp.any(children[:,0] != empty)
        
        def body(k_a_c_pd):
            key, alive, children, population = k_a_c_pd
            
            chunk_ids = jnp.nonzero(
                children[:,0] != empty,
                size=children_per_chunk,
                fill_value=empty,
            )
            chunk_children = children[chunk_ids]
            key, step_key = jrng.split(key)
            alive, population = step(
                step_keykey, alive, chunk_children, population)
            children.at[chunk_ids].set(empty)
            
            return key, alive, children, population
        
        _, alive, _, population = jax.lax.while_loop(
            cond, body, (key, survived, children, population))
        
        return alive, population
    
    if children_per_chunk is None:
        return step
    
    else:
        return chunk_step

def population_model_still_no(
    breed,
    children_per_chunk,
    empty=-1
):
    
    breed = ignore_unused_args(breed, ('key', 'params'))
    # TODO: vmapped random function
    breed = split_random_keys(jax.vmap(breed))
    
    def step_chunk(key, population_data, prev_alive, alive, parents):
        max_population, = alive.shape
        new_children, = jnp.nonzero(
            (prev_alive != alive) & (alive != empty),
            size=max_population,
            fill_value=max_population,
        )
        new_parents = parents[new_children]
        child_data = breed(key, population_data[new_parents])
        population_data = population_data.at[new_children].set(child_data)
        
        return population_data
    
    def step(key, population_data, prev_alive, alive, parents):
        max_population, = alive.shape
        new_mask = (prev_alive != alive) & (alive != empty)
        new_children, = jnp.nonzero(
            (prev_alive != alive) & (alive != empty),
            size=max_population,
            fill_value=max_population,
        )
        
        step_chunk

def population_model(
    init_params,
    breed,
    empty=-1,
):
    init_params = ignore_unused_args(init_params, ('key',))
    init_params = jax.vmap(init_params)
    breed = ignore_unused_args(breed, ('key', 'params'))
    breed = jax.vmap(breed)
    
    def init(key, population_size):
        keys = jrng.split(key, population_size)
        params = init_params(keys)
        return params
    
    def step(key, population_params, prev_players, players, parents, children):
        num_children, = children.shape
        parent_data = population_params[parents]
        keys = jrng.split(key, num_children)
        child_data = breed(keys, population_params[parents])
        population_params = population_params.at[children].set(child_data)
        return population_params
        
    return init, step
