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

import math
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static import static_data, static_functions
from mechagogue.tree import tree_getitem, tree_setitem
from mechagogue.dp.poeg import standardize_poeg
from mechagogue.ecology.policy import standardize_ecology_population
from mechagogue.serial import save_leaf_data, load_example_data

@static_data
class NaturalSelectionParams:
    max_players : int = 512
    population_blocks : int = 8

@static_data
class NaturalSelectionState:
    env_state : Any
    obs : Any
    population_state : Any

def make_natural_selection(
    params,
    env,
    population,
):
    env = standardize_poeg(env)
    population = standardize_ecology_population(population)
    
    @static_functions
    class NaturalSelection:
        # annotate that the init and step functions produce auxilliary data
        init_has_aux = True
        step_has_aux = True
        
        def init(key):
            # generate keys
            env_key, model_key = jrng.split(key)
            
            # reset the environment
            env_state, obs, active_players = env.init(env_key)
            
            # build the population_state
            population_size = jnp.sum(active_players)
            population_state = population.init(
                model_key, population_size, params.max_players)
            
            next_state = NaturalSelectionState(
                env_state, obs, population_state)
            return next_state, active_players
        
        def step(key, state):
            # generate keys
            action_key, adapt_key, env_key, breed_key = jrng.split(key, 4)
            
            # compute the traits
            traits = population.traits(state.population_state)
            
            # compute actions
            actions = population.act(
                action_key, state.obs, state.population_state)
            
            # modify the population state according to the adaptations signal
            population_state = population.adapt(
                adapt_key, state.obs, state.population_state)
            
            # step the environment
            env_state, obs, active_players, parents, children = env.step(
                env_key, state.env_state, actions, traits)
            
            # update the model state
            population_state = population.breed(
                breed_key, population_state, parents, children)
            
            # build the next state
            next_state = state.replace(
                env_state=env_state,
                obs=obs,
                population_state=population_state,
            )
            
            return (
                next_state, active_players, parents, children, actions, traits)
        
        def save_state(state, path, compress=False):
            env_obs_state = (state.env_state, state.obs)
            env_obs_path = path.replace('.data', '.env.data')
            save_leaf_data(env_obs_state, env_obs_path, compress=compress)
            
            population_state = (state.population_state)
            population_block_size = int(math.ceil(
                params.max_players / params.population_blocks))
            for i in range(params.population_blocks):
                population_block = tree_getitem(
                    population_state,
                    slice(
                        i*population_block_size,
                        (i+1)*population_block_size
                    )
                )
                block_path = path.replace('.data', f'.population.{i}.data')
                save_leaf_data(
                    population_block, block_path, compress=compress)
        
        def load_state(path):
            env_obs_path = path.replace('.data', '.env.data')
            example_state, _ = NaturalSelection.init(jrng.key(0))
            env_state, obs = load_example_data(
                (example_state.env_state, example_state.obs), env_obs_path)
            
            population_blocks = []
            for i in range(params.population_blocks):
                block_path = path.replace('.data', f'.population.{i}.data')
                population_block = load_example_data(
                    example_state.population_state,
                    block_path,
                )
                population_blocks.append(population_block)
            
            if params.population_blocks > 1:
                def concatenate(*x):
                    return jnp.concatenate(x)
                population_state = jax.tree.map(concatenate, *population_blocks)
            else:
                population_state = population_blocks[0]
            
            return NaturalSelectionState(env_state, obs, population_state)

    return NaturalSelection
