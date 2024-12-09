from typing import Sequence

import jax
import jax.random as jrng
import jax.numpy as jnp

from flax.struct import dataclass

#from mechagogue.wrappers import parallel_env_wrapper

@dataclass
class ObjectiveFreeParams:
    parallel_envs: int = 32
    rollout_steps: int = 256

def objective_free(
    params,
    reset_env,
    step_env,
    policy,
    initialize_weights,
    copy_weights,
):
    
    # wrap the environment and policy
    #reset_env, step_env = parallel_env_wrapper(reset_env, step_env)
    
    def reset_objective_free(
        key,
    ):
        # generate keys
        reset_key, weight_key = jrng.split(key)
        
        # reset the environment
        #reset_keys = jrng.split(reset_key, params.parallel_envs)
        state, obs, players, parents = reset_env(reset_key)
        num_players = jnp.sum(players != -1)
        max_players = players.shape[0]
        
        # generate new weights
        weight_keys = jrng.split(weight_key, max_players)
        weights = jax.vmap(initialize_weights, in_axes=(0,))(weight_keys)
        
        return state, obs, players, parents, weights

    def step_objective_free(
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
            #step_keys = jrng.split(step_key, params.parallel_envs)
            next_state, next_obs, next_players, next_parents = step_env(
                step_key, state, action)
            
            # TODO: copy weights as necessary
            #weights = copy_weights(
            #    weights, players, parents, next_players, next_parents)
            
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
    
    return reset_objective_free, step_objective_free
