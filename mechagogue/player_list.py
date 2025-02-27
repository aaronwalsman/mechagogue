import jax
import jax.numpy as jnp

from mechagogue.static_dataclass import static_dataclass

@static_dataclass
class AnonymousPlayerListState:
    players : jnp.array

def anonymous_player_list(max_players):
    def init(initial_players):
        players = jnp.zeros((max_players,), dtype=jnp.bool)
        players[:initial_players] = True
        return AnonymousPlayerListState(players)

    def step(state, add, remove):
        
        # remove
        players = state.players & ~remove
        
        # add
        # - find space for the newly added players
        available_locations, = jnp.nonzero(
            state.players, size=max_players, fill_value=max_players)
        add_n_hot = jnp.arange(max_players) < add
        add_locations = jnp.where(add_n_hot, available_locations, max_players)
        # - update the player mask
        players = state.players.at[add_locations].set(True)
        
        return AnonymousPlayerListState(players), add_locations
    
    def active(state):
        return state.players
    
    return init, step, active

@static_dataclass
class IdentifiedPlayerListState:
    players : jnp.array
    next_new_player_id : int

def identified_player_list(max_players):
    def init(initial_players):
        players = jnp.full((max_players,), -1, dtype=jnp.int32)
        players = players.at[:initial_players].set(jnp.arange(initial_players))
        next_new_player_id = initial_players
        
        return IdentifiedPlayerListState(players, next_new_player_id)
    
    def step(state, remove, add):
        
        # remove
        players = jnp.where(remove, -1, state.players)
        
        # add
        # - find space for the newly added players
        available_locations, = jnp.nonzero(
            (players == -1), size=max_players, fill_value=max_players)
        add_n_hot = jnp.arange(max_players) < add
        add_locations = jnp.where(add_n_hot, available_locations, max_players)
        # - construct the new player ids 
        new_player_ids = jnp.where(
            add_n_hot, all_locations + state.next_new_player_id, -1)
        # - update the player ids
        players = players.at[add_locations].set(new_player_ids)
        
        # update the next_new_player_id
        next_new_player_id = state.next_new_player_id + n
        
        return IdentifiedPlayerState(players, next_new_player_id), add_locations
    
    def active(state):
        return state.players != -1
    
    return init, step, active

@static_dataclass
class BirthdayPlayerListState:
    players : jnp.array
    current_time : int = 0

def birthday_player_list(max_players):
    def init(initial_players):
        n_hot = jnp.arange(max_players) < initial_players
        birthdays = jnp.where(n_hot, 0, -1)
        locations = jnp.where(n_hot, jnp.arange(max_players), -1)
        players = jnp.stack((birthdays, locations), axis=1)
        return BirthdayPlayerListState(players)

    def step(state, remove, add):
        
        # increment the current time
        current_time = state.current_time + 1
        
        # remove
        players = jnp.where(remove[:,None], -1, state.players)
        
        # add
        # - find locations for the newly added players
        available_locations, = jnp.nonzero(
            (players[...,0] == -1), size=max_players, fill_value=max_players)
        add_locations = jnp.where(
            jnp.arange(max_players) < add, available_locations, max_players)
        # - update the player birthdays and locations
        players = players.at[add_locations,0].set(current_time)
        players = players.at[add_locations,1].set(add_locations)
        
        return BirthdayPlayerListState(players, current_time), add_locations
    
    def active(state):
        return state.players[..., 0] != -1
    
    return init, step, active

@static_dataclass
class PlayerFamilyTreeState:
    player_list : jnp.array
    parents : jnp.array

def player_family_tree(
    init_player_list,
    step_player_list,
    active_players,
    parents_per_child=1,
):
    def init(initial_players):
        player_list = init_player_list(initial_players)
        #parents = jax.vmap(init_player_list, out_axes=-1)(
        #    jnp.zeros((parents_per_child,), dtype=jnp.int32)).players
        # this is a for loop because parents_per_child will be small, constant
        # and I couldn't get vmap to put the new dimension in the middle
        parents = jnp.stack(
            [init_player_list(initial_players).players
                for _ in range(parents_per_child)],
            axis=1,
        )
        return PlayerFamilyTreeState(player_list, parents)
    
    def step(state, deaths, parent_locations):
        
        # determine how many children to make
        num_children = jnp.sum(
            (parent_locations[...,0] >= 0) &
            (parent_locations[...,0] < state.player_list.players.shape[0])
        )
        
        # remove dead players and add new ones
        player_list, child_locations = step_player_list(
            state.player_list, deaths, num_children)
        
        # update the parent information
        new_parents = state.player_list.players[parent_locations]
        parents = state.parents.at[child_locations].set(new_parents)
        
        return PlayerFamilyTreeState(player_list, parents), child_locations
    
    def active(state):
        return active_players(state.player_list)
    
    return init, step, active
