'''
Multi-site population evolution simulation.

Tests natural selection dynamics where agents compete for energy at different
sites, reproduce based on fitness, and evolve their strategies over time.
'''

from typing import Any

import jax
import jax.random as jrng
import jax.numpy as jnp

from mechagogue.dp.population_game import population_game
from mechagogue.pop.natural_selection import (
    NaturalSelectionConfig, natural_selection)
from mechagogue.static_dataclass import static_dataclass
from mechagogue.player_list import birthday_player_list, player_family_tree

@static_dataclass
class MultiSiteConfig:
    num_sites : int = 8
    initial_players : int = 8
    max_players : int = 256
    num_channels : int = 32
    energy_loss_per_step : float = 0.05
    initial_energy : float = 0.2
    reproduce_energy : float = 1.
    energy_per_site : float = 1
    energy_allocation : str = 'winner'
    allocation_sharpness : float = 1.

@static_dataclass
class MultiSiteState:
    sites : jnp.array
    family_tree : Any
    energy : jnp.array

def multisite(config):
    
    init_players, step_players, active_player_list = birthday_player_list(
        config.max_players)
    init_family_tree, step_family_tree, active_family_tree = player_family_tree(
        init_players, step_players, active_player_list, 1)
    
    def init_state(key):
        
        # initialize sites
        sites = jrng.normal(key, shape=(config.num_sites, config.num_channels))
        
        # initialize player family tree
        family_tree = init_family_tree(config.initial_players)
        
        # initialize energy
        energy = jnp.zeros((config.max_players,))
        energy = energy.at[:config.initial_players].set(config.initial_energy)
        
        return MultiSiteState(
            sites,
            family_tree,
            energy,
        )
    
    def transition(state, action):
        
        # divide the new energy between each player
        offsets = state.sites[:,None] - action[None,:]
        distances = jnp.linalg.norm(offsets, axis=-1)
        distances = jnp.where(
            active_family_tree(state.family_tree), distances, jnp.inf)
        
        if config.energy_allocation == 'proportional':
            exponent = config.allocation_sharpness * (config.num_channels**0.5)
            raw_share = 1. / distances ** exponent
            raw_share = jnp.where(alive, raw_share, 0.)
            portions = raw_share / jnp.sum(raw_share, axis=1, keepdims=True)
            breakpoint()
            assert False, 'update this'
        
        elif config.energy_allocation == 'winner':
            min_distance = jnp.argmin(distances, axis=-1)
            energy = state.energy.at[min_distance].add(config.energy_per_site)
        
        # metabolism
        # - first subtract out the energy costs of living
        energy = energy - config.energy_loss_per_step
        energy = jnp.clip(energy, min=0.)
        
        # anything without enough energy remaining will die
        deaths = energy <= 0.
        
        # anything with energy greater than reproduce_energy will reproduce
        reproduce = energy > config.reproduce_energy
        parent_locations, = jnp.nonzero(
            reproduce, size=config.max_players, fill_value=config.max_players)
        parent_locations = parent_locations[:,None]
        
        family_tree, child_locations = step_family_tree(
            state.family_tree, deaths, parent_locations)
        
        # update the energy based on reproduction
        energy = energy.at[parent_locations].add(-config.initial_energy)
        energy = energy.at[child_locations].add(config.initial_energy)
        
        return MultiSiteState(
            state.sites,
            family_tree,
            energy,
        )
    
    def observe(state):
        return None
    
    def active_players(state):
        return active_family_tree(state.family_tree)
    
    def family_info(next_state):
        birthdays = next_state.family_tree.player_list.players[...,0]
        current_time = next_state.family_tree.player_list.current_time
        child_locations, = jnp.nonzero(
            birthdays == current_time,
            size=config.max_players,
            fill_value=config.max_players,
        )
        parent_info = next_state.family_tree.parents[child_locations]
        parent_locations = parent_info[...,1]
        
        return parent_locations, child_locations
    
    return population_game(
        init_state,
        transition,
        observe,
        active_players,
        family_info,
    )

epochs = 10
steps_per_epoch = 1000
num_channels = 32
mutation = 0.01
max_players = 64
env_config = MultiSiteConfig(
    num_sites=4,
    energy_per_site=0.25,
    num_channels=num_channels,
    max_players=max_players,
    initial_players=2,
)
reset_env, step_env = multisite(env_config)

def init_model_state(key):
    return jrng.normal(key, shape=(num_channels))

def model(state):
    return state

def breed(key, state):
    return state[0] + jrng.normal(key, shape=state[0].shape) * mutation

reset_train, step_train = natural_selection(
    NaturalSelectionConfig(max_players),
    reset_env,
    step_env,
    init_model_state,
    model,
    breed,
)

def go(key):
    init_key, epoch_key = jrng.split(key)
    train_state, active_players = reset_train(init_key)
    
    def train_block(train_state_active, key):
        '''
            One epoch.
        '''
        train_state, active_players = train_state_active
        offsets = (
            train_state.env_state.sites[:,None] - 
            (train_state.model_params)[None,:]
        )
        distances = jnp.linalg.norm(offsets, axis=-1)
        assignment = jnp.argmin(distances, axis=0)
        assignment = jnp.where(active_players, assignment, -1)
        min_distance = jnp.min(distances, axis=0)
        min_distance = jnp.where(active_players, min_distance, jnp.inf)
        jax.debug.print('----------')
        jax.debug.print('assign {a}', a=assignment)
        jax.debug.print('distances {md}', md=min_distance)
        
        def scan_body(train_state_active, key):
            train_state, _ = train_state_active
            next_state, active_players, parents, children = step_train(
                key, train_state)
            return (
                (next_state, active_players),
                (active_players, parents, children),
            )
        
        train_state_active, _ = jax.lax.scan(
            scan_body,
            (train_state, active_players),
            jrng.split(key, steps_per_epoch),
        )
        
        return train_state_active, None
    
    train_state_active, _ = jax.lax.scan(
        train_block,
        (train_state, active_players),
        jrng.split(epoch_key, epochs),
    )
    
    train_state, _ = train_state_active
    return train_state

go = jax.jit(go)
key = jrng.key(1234)
train_state = go(key)
