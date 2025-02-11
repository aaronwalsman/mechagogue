import jax
import jax.random as jrng
import jax.numpy as jnp

from mechagogue.dp.population_game import population_game
from mechagogue.pop.natural_selection import (
    NaturalSelectionConfig, natural_selection)
from mechagogue.static_dataclass import static_dataclass

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
    sites : jnp.array = None
    players : jnp.array = None
    parents : jnp.array = None
    children : jnp.array = None
    next_player_id : int = 0
    energy : jnp.array = None

def multisite(config):
    
    def init_state(key):
        
        # initialize sites
        sites = jrng.normal(key, shape=(config.num_sites, config.num_channels))
        
        # initialize players and parents
        players = jnp.full((config.max_players,), -1, dtype=jnp.int32)
        players = players.at[:config.initial_players].set(
            jnp.arange(config.initial_players))
        parents = jnp.full(
            (config.max_players, 1), -1, dtype=jnp.int32)
        children = jnp.full(
            (config.max_players,), -1, dtype=jnp.int32)
        next_player_id = config.initial_players
        
        # initialize energy
        energy = jnp.zeros((config.max_players,))
        energy = energy.at[:config.initial_players].set(config.initial_energy)
        
        return MultiSiteState(
            sites,
            players,
            parents,
            children,
            next_player_id,
            energy,
        )
    
    def transition(state, action):
        
        # first subtract out the energy costs of living
        energy = state.energy - config.energy_loss_per_step
        
        # kill off anything that has run out of energy
        players = jnp.where(energy > 0., state.players, -1)
        energy = jnp.clip(energy, min=0.)
        
        # divide the new energy between each player
        alive = players != -1
        offsets = state.sites[:,None] - action[None,:]
        distances = jnp.linalg.norm(offsets, axis=-1)
        distances = jnp.where(alive, distances, jnp.inf)
        
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
        parents, = jnp.nonzero(
            reproduce,
            size=config.max_players,
            fill_value=config.max_players,
        )
        parents = parents[:,None]
        num_new_players = jnp.sum(reproduce)
        
        # find locations for the new children
        all_locations = jnp.arange(config.max_players)
        active_children = all_locations < num_new_players
        
        available_locations, = jnp.nonzero(
            ~alive, size=config.max_players, fill_value=config.max_players)
        children = jnp.where(
            active_children, available_locations, config.max_players)
        child_ids = jnp.where(
            active_children, all_locations + state.next_player_id, -1)
        players = players.at[children].set(child_ids)
        
        # update the energy based on reproduction
        energy = energy.at[parents].add(-config.initial_energy)
        energy = energy.at[children].add(config.initial_energy)
        
        next_player_id = state.next_player_id + num_new_players
        return MultiSiteState(
            state.sites,
            players,
            parents,
            children,
            next_player_id,
            energy,
        )
    
    def observe(state):
        return None
    
    def player_info(state):
        return state.players, state.parents, state.children
    
    return population_game(init_state, transition, observe, player_info)

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

def init_model_params(key):
    return jrng.normal(key, shape=(num_channels))

def model(params):
    return params

def breed(key, params):
    return params[0] + jrng.normal(key, shape=params[0].shape) * mutation

reset_train, step_train = natural_selection(
    NaturalSelectionConfig(),
    reset_env,
    step_env,
    init_model_params,
    model,
    breed,
)

def go(key):
    init_key, epoch_key = jrng.split(key)
    train_state = reset_train(init_key)
    
    def train_block(train_state, key):
        
        
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
