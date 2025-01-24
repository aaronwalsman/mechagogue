import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.arg_wrappers import ignore_unused_args
from mechagogue.tree import tree_len,tree_len,  tree_getitem, tree_setitem

def population_NO(
    init_model_params,
    model,
    init_breed_params,
    breed,
):
    
    init_model_params = ignore_unused_args(init_model_params,
        ('key,',))
    model = ignore_unused_args(model,
        ('key', 'x', 'params'))
    init_breeder_params = ignore_unused_args(init_breeder_params,
        ('key',))
    breed = ignore_unused_args(breed,
        ('key', 'model_params', 'breed_params'))

    def init_population(key):
        model_key, breed_key = jrng.split(key)
        
        model_keys = jrng.split(model_key, population_size)
        model_params = jax.vmap(init_model_params)(model_keys)
        
        breed_params = init_breed_params(breed_key)
        
        return model_params, breed_params
    
    def model_population(key, x, params):
        model_params, breed_params = params
        keys = jrng.split(key, population_size)
        return vmap(model)(keys, x, model_params)
        
    
    def step_population(key, survived, children, params):
        model_params, breed_params = params
        def cond(k_s_c_p_bp):
            key, survived, children, population, breed_params = k_s_c_p_bp
            return jnp.any(children[:,0] != empty)
        
        def body(k_s_c_p_bp):
            key, survived, children, population, breed_params = k_s_c_p_bp
            
            max_population, = survived.shape
            max_children, parents_per_children = children.shape
            
            if children_per_step is None:
                step_children = max_population
            else:
                step_children = children_per_step
            
            # figure out where to write the new children
            population_write_ids, = jnp.nonzero(
                ~survived, size=step_children, fill_value=max_population)
            
            # find locations in the children tensor that will produce children
            # this step
            parent_ids = jnp.nonzero(
                children[:,0] != empty,
                size=step_children,
                fill_value=max_children,
            )
            population_read_ids = children[parent_ids]
            
            # breed
            breeder_keys = jrng.split(key, step_children+1)
            key, breeder_keys = breeder_keys[0], breeder_keys[1:]
            parent_data = tree_getitem(population, population_read_ids)
            # TODO: poor performance when vmapping a random function
            child_data, breeder_params = jax.vmap(breeder, in_axes=(0,0,None))(
                breeder_keys, parent_data, breeder_params)
            
            survived = survived.at[population_write_ids].set(True)
            population = tree_setitem(
                population, population_write_ids, child_data)
            children = children.at[parent_ids].set(-1)
            
            return key, survived, children, population, breeder_params
        
        _, alive, _, population, breeder_params = jax.lax.while_loop(
            cond, body, (key, survived, children, population, breeder_params))
        
        #return alive, population, breeder_params
        return model_params, breed_params
        
    return init_population, model_population, step_population

def population_model(
    breed,
    children_per_chunk=None,
    empty=-1,
):
    
    breed = ignore_unused_args(breed, ('key', 'params'))
    
    def step(key, alive, children, population):
        population_size = tree_len(population)
        num_children, parents_per_child = children.shape
        
        parent_data = tree_getitem(population, children)
        breed_keys = jrng.split(key, num_children)
        # TODO: vmapped random function
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
            
            chunk_ids = jrng.nonzero(
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

def population_step(
    key,
    survived,
    children,
    population,
    breeder,
    breeder_params,
    children_per_step=None,
    empty=-1,
):
    def cond(k_s_c_p_bp):
        key, survived, children, population, breeder_params = k_s_c_p_bp
        return jnp.any(children[:,0] != empty)
    
    def body(k_s_c_p_bp):
        key, survived, children, population, breeder_params = k_s_c_p_bp
        
        max_population, = survived.shape
        max_children, parents_per_children = children.shape
        
        if children_per_step is None:
            step_children = max_population
        else:
            step_children = children_per_step
        
        # figure out where to write the new children
        population_write_ids, = jnp.nonzero(
            ~survived, size=step_children, fill_value=max_population)
        
        # find locations in the children tensor that will produce children
        # this step
        parent_ids = jnp.nonzero(
            children[:,0] != empty, size=step_children, fill_value=max_children)
        population_read_ids = children[parent_ids]
        
        # breed
        breeder_keys = jrng.split(key, step_children+1)
        key, breeder_keys = breeder_keys[0], breeder_keys[1:]
        parent_data = tree_getitem(population, population_read_ids)
        # TODO: poor performance when vmapping a random function
        child_data, breeder_params = jax.vmap(breeder, in_axes=(0,0,None))(
            breeder_keys, parent_data, breeder_params)
        
        survived = survived.at[population_write_ids].set(True)
        population = tree_setitem(population, population_write_ids, child_data)
        children = children.at[parent_ids].set(-1)
        
        return key, survived, children, population, breeder_params
    
    _, alive, _, population, breeder_params = jax.lax.while_loop(
        cond, body, (key, survived, children, population, breeder_params))
    
    return alive, population, breeder_params
