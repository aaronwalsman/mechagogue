import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.tree import tree_getitem, tree_setitem

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
