'''
The natural selection algorithm simulates a population of players forward
over multiple time steps.  Each step uses a model function to compute an
action for each agent, then uses those actions to compute environment
dynamics.  In addition to a state and observation, the environment should
produce a list of players that currently exist, and their parents.  This
information is then used to construct the weights of new players.

This algorithm does not have an optimization objective, but instead relies
on the environment dynamics to update the population over time.
'''

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static import static_data, static_functions
from mechagogue.tree import tree_getitem, tree_setitem
from mechagogue.dp.poeg import standardize_poeg
from mechagogue.ecology.policy import standardize_ecology_population

@static_dataclass
class NaturalSelectionParams:
    max_population : int

@static_dataclass
class NaturalSelectionState:
    env_state : Any
    obs : Any
    population_state : Any

def natural_selection(
    params,
    env,
    population,
):
    env = standardize_poeg(env)
    population = standardize_ecology_population(population)
    
    @static_functions
    class NaturalSelection:
        def init(key):
            
            # generate keys
            env_key, model_key = jrng.split(key)
            
            # reset the environment
            env_state, obs, active_players = env.init(env_key)
            
            # build the population_state
            population_size = jnp.sum(active_players)
            population_state = population.init(
                model_key, population_size, params.max_population)
            
            next_state = NaturalSelectionState(
                env_state, obs, population_state)
            return next_state, active_players
        
        def step(key, state):
            
            # generate keys
            action_key, adapt_key, env_key, breed_key = jrng.split(key, 4)
            
            # compute the traits that will be passed to the environment
            traits = state.population_state.traits(state.population_state)
            
            # compute actions
            actions = population.act(
                action_key, state.obs, state.population_state)
            #action_keys = jrng.split(action_key, params.max_population)
            #actions, adaptations = model(
            #    action_keys, state.obs, state.population_state)
            
            # modify the model state according to the adaptations signal
            #adapt_keys = jrng.split(adapt_key, params.max_population)
            #population_state = adapt(
            #    adapt_keys, adaptations, state.population_state)
            population_state = population.adapt(
                adapt_key, state.obs, state.population_state)
            
            # step the environment
            env_state, obs, active_players, parents, children = env.step(
                env_key, state.env_state, actions, traits)
            
            # update the model state
            max_num_children, = children.shape
            #parent_state = tree_getitem(population_state, parents)
            #parent_state = population.get_members(population_state, parents)
            #child_state = population.breed(breed_key, parent_state)
            #population_state = tree_setitem(
            #    population_state, children, child_state)
            #population_state = population.set_members(
            #    population_state, children, child_state)
            state = population.breed(
                breed_key, state.population_state, parents, children)
            
            # build the next state
            next_state = state.replace(
                env_state=env_state,
                obs=obs,
                population_state=population_state,
            )
            
            return (
                next_state,
                active_players,
                parents,
                children,
                actions,
                traits,
                adaptations,
            )
    
    return NaturalSelection
