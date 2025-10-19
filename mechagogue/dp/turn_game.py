'''
Turn-based game environment builder.

Creates multi-player turn-based games with configurable state transitions,
observations, rewards, and player tracking.
'''

from typing import Any, Callable

import jax.random as jrng

def turn_game(
    initialize_fn: Callable,
    transition_fn: Callable,
    players_fn: Callable,
    current_player_fn: Callable,
    observe_fn: Callable,
    reward_fn: Callable,
    done_fn: Callable,
    config: Any = None,
) -> Callable:
    
    initialize_fn = ignore_unused_args(initialize_fn, ('key', 'config'))
    transition_fn = ignore_unused_args(
        transition_fn, ('key', 'config', 'state', 'action'))
    
    def reset(key):
        # make new keys
        initialize_key, observe_key = jrng.split(key, 2)
        
        # make the first state
        state = initialize_fn(initialize_key, config)
        
        # compute the number of players and the current player
        players = players_fn(config, state)
        current_player = current_player_fn(config, state)
        
        # compute the observation
        obs = observe_fn(observe_key, config, state)
        
        return state, players, current_player, obs
    
    def step(key, state, action):
        # make new keys
        transition_key, observe_key, reward_key, done_key = jrng.split(key, 4)
        
        # compute the next state from the current state and action
        next_state = transition_fn(transition_key, config, state, action)
        
        # compute the number of players and the current player
        players = players_fn(config, next_state)
        current_player = current_player_fn(config, next_state)
        
        # compute the observation, rewards and done
        obs = observe_fn(observe_key, config, next_state)
        reward_args = filter_args(reward_format, state, action, next_state)
        reward = reward_fn(reward_key, config, *reward_args)
        done_args = filter_args(done_format, state, action, next_state)
        done = done_fn(done_key, config, *done_args)
        
        return next_state, num_players, current_player, obs, reward, done
