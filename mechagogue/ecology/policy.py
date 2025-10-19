'''
Ecology policy and population abstractions.

Provides interfaces for individual agent policies and vectorized populations
with actions, adaptation, breeding, and trait inheritance.
'''

import jax
import jax.random as jrng

from mechagogue.static import static_data, static_functions
from mechagogue.standardize import standardize_interface, standardize_args
from mechagogue.tree import tree_getitem, tree_setitem

def standardize_ecology_policy(ecology_policy):
    return standardize_interface(
        ecology_policy,
        init=(('key',), lambda : None),
        act=(('key', 'obs', 'state'), None),
        adapt=(('key', 'obs', 'state'), lambda state : state),
        traits=(('state',), lambda : None),
    )

def standardize_ecology_population(ecology_population):
    return standardize_interface(
        ecology_population,
        init=(('key', 'population_size', 'max_population_size'), lambda : None),
        act=(('key', 'obs', 'state'), None),
        adapt=(('key', 'obs', 'state'), lambda state : state),
        traits=(('state',), None),
        breed=(('key', 'state', 'parents', 'children'), lambda state : state)
    )

def make_ecology_population(
    policy,
    max_population_size,
    breed = None
):
    
    policy = standardize_ecology_policy(policy)
    _breed = None
    if breed is not None:
        _breed = standardize_args(breed, ('key', 'parent_state'))
    
    @static_functions
    class EcologyPopulation:

        def init(key, population_size, max_population_size):
            keys = jrng.split(key, max_population_size)
            state = jax.vmap(policy.init)(keys)
            return state

        def act(key, obs, state):
            keys = jrng.split(key, max_population_size)
            actions = jax.vmap(policy.act)(keys, obs, state)
            return actions

        def adapt(key, obs, state):
            keys = jrng.split(key, max_population_size)
            state = jax.vmap(policy.adapt)(keys, obs, state)
            return state

        traits = jax.vmap(policy.traits)
        
        def get_members(state, locations):
            return tree_getitem(state, locations)
        
        def set_members(state, locations, values):
            return tree_setitem(state, locations, values)

        def breed(key, state, parents, children):
            if _breed is not None:
                keys = jrng.split(key, max_population_size)
                parent_states = EcologyPopulation.get_members(state, parents)
                child_states = jax.vmap(_breed)(keys, parent_states)
                state = EcologyPopulation.set_members(
                    state, children, child_states)
            return state

    return EcologyPopulation
