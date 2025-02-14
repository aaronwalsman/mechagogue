import jax.numpy as jnp

from mechagogue.static_dataclass import static_dataclass

@static_dataclass
class AnonymousPlayerListState:
    players : jnp.array

def anonymous_player_list(initial_players, max_players):
    def init():
        players = jnp.zeros((max_players,), dtype=jnp.bool)
        players[:initial_players] = True
        return AnonymousPlayerListState(players)

    def add(state, n):
        
        # find space for the new children
        available_locations, = jnp.nonzero(
            state.players, size=max_players, fill_value=max_players)
        
        child_values = jnp.arange(max_players) < n
        children = jnp.where(child_values, available_locations, max_players)
        
        players = state.players.at[children].set(True)
        
        return AnonymousPlayerListState(players)
    
    def remove(state, to_remove):
        players = state.players & ~to_remove
        return state.replace(players=players)
    
    return init, add, remove

@static_dataclass
class IdentifiedPlayerListState:
    players : jnp.array
    next_new_player_id : int

def identified_player_list(initial_players, max_players):
    def init():
        players = jnp.full((max_players,), -1, dtype=jnp.int32)
        players = players.at[:initial_players].set(jnp.arange(initial_players))
        next_new_player_id = initial_players
        children = jnp.full((max_players,), -1, dtype=jnp.int32)
        
        return IdentifiedPlayerListState(players, next_new_player_id), children
        
    def add(state, n):
        
        # find space for the new players
        available_locations, = jnp.nonzero(
            (state.players == -1), size=max_players, fill_value=max_players)
        
        # make the new child ids
        all_locations = jnp.arange(max_players)
        active_children = all_locations < n
        children = jnp.where(
            active_children, available_locations, max_players)
        child_ids = jnp.where(
            active_children, all_locations + next_new_player_id, -1)
        
        # add the child ids into the new players
        players = state.players.at[children].set(child_ids)
        
        # update the next_new_player_id
        next_new_player_id = state.next_new_player_id + n
        
        return IdentifiedPlayerState(players, next_new_player_id), children
    
    def remove(state, to_remove):
        players = jnp.where(to_remove, -1, state.players)
        return state.replace(players=players)

    return init, add, remove

@static_dataclass
class BirthdayPlayerListState:
    birthdays : jnp.array
    current_time = 0

def birthday_player_list(initial_players, max_players):
    def init():
        birthdays = jnp.full((max_players,), -1, dtype=jnp.int32)
        birthdays = birthdays.at[:initial_players].set(0)
        return BirthdayPlayerListState(birthdays)

    def step(state, remove, add):
        
        # increment the current time
        current_time = state.current_time + 1
        
        # remove
        birthdays = jnp.where(remove, -1, state.birthdays)
        
        # add
        # - find locations for the new players
        available_locations, = jnp.nonzero(
            (birthdays == -1), size=max_players, fill_value=max_players)
        new_players = jnp.where(
            jnp.arange(max_players) < n, available_locations, max_players)
        
        birthdays = birthdays.at[new_players].set(current_time)
        
        return BirthdayPlayerListState(birthdays, current_time)
    
    #def remove(state, to_remove):
    #    birthdays = jnp.where(to_remove, -1, state.birthdays)
    #    return state.replace(birthdays=birthdays)

@static_dataclass
def family_tree(init_player_list, step_player_list):
    
