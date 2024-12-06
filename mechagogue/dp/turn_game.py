from typing import Any, Callable

import jax.random as jrng

from decision_process.filter_args import filter_args

def turn_game(
    params: Any,
    initialize_fn: Callable,
    transition_fn: Callable,
    players_fn: Callable,
    current_player_fn: Callable,
    observe_fn: Callable,
    reward_fn: Callable,
    done_fn: Callable,
    reward_format: str = 'san',
    done_format: str = 'san',
) -> Callable:
    
    def reset(key):
        # make new keys
        initialize_key, observe_key = jrng.split(key, 2)
        
        # make the first state
        state = initialize_fn(initialize_key, params)
        
        # compute the number of players and the current player
        players = players_fn(params, state)
        current_player = current_player_fn(params, state)
        
        # compute the observation
        obs = observe_fn(observe_key, params, state)
        
        # return
        return state, players, current_player, obs
    
    def step(key, state, action):
        # make new keys
        transition_key, observe_key, reward_key, done_key = jrng.split(key, 4)
        
        # compute the next state from the current state and action
        next_state = transition_fn(transition_key, params, state, action)
        
        # compute the number of players and the current player
        players = players_fn(params, next_state)
        current_player = current_player_fn(params, next_state)
        
        # compute the observation, rewards and done
        obs = observe_fn(observe_key, params, next_state)
        reward_args = filter_args(reward_format, state, action, next_state)
        reward = reward_fn(reward_key, params, *reward_args)
        done_args = filter_args(done_format, state, action, next_state)
        done = done_fn(done_key, params, *done_args)
        
        # return
        return next_state, num_players, current_player, obs, reward, done
