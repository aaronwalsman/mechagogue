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
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.population import population_step

#@dataclass
#class NaturalSelectionParams:
#    '''
#    Configuration parameters.  This should only contain values that will be
#    fixed throughout training.
#    '''
#    rollout_steps: int = 256

def natural_selection(
    params,
    reset_env,
    step_env,
    init_policy_params,
    policy,
    init_breeder_params,
    breeder,
    rollout_steps : int = 256
):
    
    def reset_natural_selection(key):
        
        # generate keys
        reset_key, policy_key, breed_key = jrng.split(key)
        
        # reset the environment
        state, obs, alive, children = reset_env(reset_key)
        max_players, = alive.shape
        
        # generate new policy params
        policy_keys = jrng.split(policy_key, max_players)
        policy_params = jax.vmap(init_policy_params)(weight_keys)
        
        # generate new breeder params
        breeder_params = init_breeder_params(breeder_key)
        
        return state, obs, alive, children, policy_params, breeder_params

    def step_natural_selection(
        key,
        state,
        obs,
        policy_params,
    ):
        ## rollout trajectories
        #def rollout(rollout_state, key):
            
        #    # unpack
        #    state, obs, policy_params = rollout_state
            
        # sample an action
        key, action_key = jrng.split(key)
        action = policy(key, obs, policy_params)
        
        # take an environment step
        key, env_key = jrng.split(key)
        next_state, next_obs, survived, children = step_env(
            env_key, state, action)
        
        # update the population
        key, population_key = jrng.split(key)
        _, next_policy_params, next_breeder_params = population_step(
            population_key,
            survived,
            children,
            policy_params,
            breeder,
            breeder_params,
            children_per_step=children_per_step
        )
        
        # pack
        rollout_state = (
            next_state,
            next_obs,
            next_policy_params,
        )
        transition = (obs, action, survived, children)
        
        return rollout_state, transition
        
        ## scan rollout_step to accumulate trajectories
        #key, rollout_key = jrng.split(key)
        #(state, obs, players, parents, weights), trajectories = jax.lax.scan(
        #    rollout,
        #    (state, obs, players, parents, weights),
        #    jrng.split(rollout_key, params.rollout_steps),
        #    params.rollout_steps,
        #)
        #
        #return (state, obs, policy_params), trajectories
    
    return reset_natural_selection, step_natural_selection
