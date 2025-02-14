from typing import Any

import jax
import jax.random as jrng
import jax.numpy as jnp

from mechagogue.dp.population_game import population_game
from mechagogue.pop.natural_selection import (
    NaturalSelectionConfig, natural_selection)
from mechagogue.static_dataclass import static_dataclass
from mechagogue.player_list import birthday_player_list

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

@static_dataclass
class MultiSiteState:
    sites : jnp.array
    player_list : Any
    #next_new_player_id : int
    #players : jnp.array
    #parents : jnp.array
    #children : jnp.array
    energy : jnp.array

def multisite(config):
    
    init_player_list, add_players, remove_players = birthday_player_list(
        config.initial_players, config.max_players)
    
    def init_state(key):
        
        # initialize sites
        sites = jrng.normal(key, shape=(config.num_sites, config.num_channels))
        
        # initialize players and parents
        #next_new_player_id, players, children = init_population(
        player_list = init_player_list
            config.initial_players, config.max_players)
        parents = jnp.full(
            (config.max_players, 1), -1, dtype=jnp.int32)
        
        # initialize energy
        energy = jnp.zeros((config.max_players,))
        energy = energy.at[:config.initial_players].set(config.initial_energy)
        
        return MultiSiteState(
            sites,
            player_list,
            #next_new_player_id,
            #players,
            #parents,
            #children,
            energy,
        )
    
    def transition(state, action):
        
        # metabolism
        # - first subtract out the energy costs of living
        energy = state.energy - config.energy_loss_per_step
        energy = jnp.clip(energy, min=0.)
        
        # kill off anything that has run out of energy
        deaths = energy <= 0.
        
        # divide the new energy between each player
        offsets = state.sites[:,None] - action[None,:]
        distances = jnp.linalg.norm(offsets, axis=-1)
        distances = jnp.where(deaths, jnp.inf, distances)
        
        # proportional
        '''
        sharpness=2
        raw_share = 1. / distances ** (sharpness * (config.num_channels**0.5))
        raw_share = jnp.where(alive, raw_share, 0.)
        proportions = raw_share / jnp.sum(raw_share, axis=1, keepdims=True)
        '''
        
        # winner take all
        min_distance = jnp.argmin(distances, axis=-1)
        energy = energy.at[min_distance].add(config.energy_per_site)
        
        # determine who will reproduce this round
        reproduce = energy > config.reproduce_energy
        births = jnp.sum(reproduce)
        '''
        parents, = jnp.nonzero(
            reproduce,
            size=config.max_players,
            fill_value=config.max_players,
        )
        parents = parents[:,None]
        '''
        player_list = step_players(player_list, births, deaths)
        
        # update the energy based on reproduction
        energy = energy.at[parents].add(-config.initial_energy)
        energy = energy.at[children].add(config.initial_energy)
        
        #next_new_player_id = state.next_new_player_id + num_new_players
        return MultiSiteState(
            state.sites,
            player_list,
            #next_new_player_id,
            #players,
            #parents,
            #children,
            energy,
        )
    
    def observe(state):
        return None
    
    def active_players(state):
        return state.player_list.birthdays != -1
    
    def family_info(next_state):
        birthdays = next_state.player_list.birthdays
        current_time = next_state.player_list.current_time
        children, = jnp.nonzero(birthdays == current_time)
        parents
    
    return population_game(init_state, transition, observe, active_players)

epochs = 10
steps_per_epoch = 1000
num_channels = 32
mutation = 0.01
env_config = MultiSiteConfig(
    num_sites=4,
    energy_per_site=0.25,
    num_channels=num_channels,
    max_players=64,
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
    NaturalSelectionConfig(),
    reset_env,
    step_env,
    init_model_state,
    model,
    breed,
)

def go(key):
    init_key, epoch_key = jrng.split(key)
    train_state = reset_train(init_key)
    
    def train_block(train_state, key):
        '''
            One epoch.
        '''
        alive = train_state.players != -1
        offsets = (
            train_state.env_state.sites[:,None] - 
            (train_state.model_params)[None,:]
        )
        distances = jnp.linalg.norm(offsets, axis=-1)
        assignment = jnp.argmin(distances, axis=0)
        assignment = jnp.where(alive, assignment, -1)
        min_distance = jnp.min(distances, axis=0)
        min_distance = jnp.where(alive, min_distance, jnp.inf)
        #jax.debug.print('----------')
        #jax.debug.print('players {p}', p=train_state.players)
        #jax.debug.print('parents {p}', p=train_state.parents[:,0])
        #jax.debug.print('children {c}', c=train_state.children)
        #jax.debug.print('EN {e}', e=train_state.env_state.energy)
        jax.debug.print('assign {a}', a=assignment)
        jax.debug.print('distances {md}', md=min_distance)
        #jax.debug.print('params {mp}', mp=train_state.model_params)
        #jax.debug.print('sites {sites}', sites=train_state.env_state.sites)
        
        
        train_state, _ = jax.lax.scan(
            lambda train_state, key : (step_train(key, train_state), None),
            train_state,
            jrng.split(key, steps_per_epoch),
        )
        
        return train_state, None
    
    train_state, _ = jax.lax.scan(
        train_block, train_state, jrng.split(epoch_key, epochs))
    
    return train_state

go = jax.jit(go)
key = jrng.key(1234)
train_state = go(key)
