'''
The natural selection algorithm simulates a population of players forward
over multiple time steps.  Each step uses a policy function to compute an
action for each agent, then uses those actions to compute environment
dynamics.  In addition to a state and observation, the environment should
produce a list of players that currently exist, and their parents.  This
information is then used to construct the weights of new players.

This algorithm does not have an optimization objective, but instead relies
on the environment dynamics to update the population over time.
'''

from typing import Sequence

import jax
import jax.random as jrng
import jax.numpy as jnp

from flax.struct import dataclass

@dataclass
class NaturalSelectionParams:
    '''
    Configuration parameters.  This should only contain values that will be
    fixed throughout training.
    '''
    rollout_steps: int = 256

def natural_selection(
    params,
    reset_env,
    step_env,
    policy,
    initialize_weights,
    update_weights,
):
    '''
    Bundles a set of parameters, an environment reset_env and step_env
    functions, a policy and a way to initialize and copy weights into
    reset_natural_selection and step_natural_selection functions.
    
    The reset_natural_selection function is meant to be called once at the
    beginning of a training session to initialize the state information for
    a single run of the natural selection algorithm.
    
    The step_natural_selection function can be called iteratively to compute
    a dynamic population of players.
    '''
    
    def reset_natural_selection(
        key,
    ):
        # generate keys
        reset_key, weight_key = jrng.split(key)
        
        # reset the environment
        state, obs, players, parents = reset_env(reset_key)
        num_players = jnp.sum(players != -1)
        max_players = players.shape[0]
        
        # generate new weights
        weight_keys = jrng.split(weight_key, max_players)
        weights = jax.vmap(initialize_weights, in_axes=(0,))(weight_keys)
        
        return state, obs, players, parents, weights

    def step_natural_selection(
        key,
        state,
        obs,
        players,
        parents,
        weights,
    ):
        # rollout trajectories
        def rollout(rollout_state, key):
            
            # unpack
            state, obs, players, parents, weights = rollout_state
            
            # sample an action
            key, action_key = jrng.split(key)
            action_sampler, _ = policy(weights, obs)
            action = action_sampler(action_key)
            
            # take an environment step
            key, step_key = jrng.split(key)
            next_state, next_obs, next_players, next_parents = step_env(
                step_key, state, action)
            
            # TODO: copy weights as necessary
            weights = update_weights(
                weights, players, parents, next_players, next_parents)
            
            # pack
            rollout_state = (
                next_state, next_obs, next_players, next_parents, weights)
            transition = (obs, action, players, parents)
            
            return rollout_state, transition
        
        # scan rollout_step to accumulate trajectories
        key, rollout_key = jrng.split(key)
        (state, obs, players, parents, weights), trajectories = jax.lax.scan(
            rollout,
            (state, obs, players, parents, weights),
            jrng.split(rollout_key, params.rollout_steps),
            params.rollout_steps,
        )
        
        return (state, obs, players, parents, weights), trajectories
    
    return reset_natural_selection, step_natural_selection
