'''
Player list data structures for managing dynamic populations in multi-agent systems.

Provides different player tracking schemes: anonymous (boolean masks), identified
(unique IDs), birthday-based (timestamped), and family trees (parent-child relationships).
'''

import jax
import jax.numpy as jnp

from mechagogue.static import static_data, static_functions

def anonymous_player_list(max_players):
    
    @static_data
    class AnonymousPlayerListState:
        players : jnp.array
        capacity_reached : bool = False
    
    @static_functions
    class AnonymousPlayerList:
    
        def init(initial_players):
            players = jnp.zeros((max_players,), dtype=jnp.bool)
            players[:initial_players] = True
            return AnonymousPlayerListState(players, capacity_reached=False)

        def step(state, add, remove):
            
            # remove
            players = state.players & ~remove
            
            # add
            # - find space for the newly added players
            available_locations, = jnp.nonzero(
                state.players, size=max_players, fill_value=max_players)
            add_n_hot = jnp.arange(max_players) < add
            add_locations = jnp.where(
                add_n_hot, available_locations, max_players)
            # - update the player mask
            valid = add_locations < max_players
            safe_locations = jnp.where(valid, add_locations, 0)
            players = players.at[safe_locations].set(
                jnp.where(valid, True, players[safe_locations]))
            
            available_count = jnp.sum(~state.players)
            capacity_reached = add > available_count
            next_state = AnonymousPlayerListState(players, capacity_reached)
            added_players = jnp.where(valid, players[safe_locations], False)
            return next_state, add_locations, added_players
        
        def active(state):
            return state.players
        
        def locations_to_ids(state, locations):
            return state.players[locations]
    
    return init, step, active


def identified_player_list(max_players):
    
    @static_data
    class IdentifiedPlayerListState:
        players : jnp.array
        next_new_player_id : int
        capacity_reached : bool = False
    
    @static_functions
    class IdentifiedPlayerList:
        
        def init(initial_players):
            players = jnp.full((max_players,), -1, dtype=jnp.int32)
            players = players.at[:initial_players].set(
                jnp.arange(initial_players))
            next_new_player_id = initial_players
            
            return IdentifiedPlayerListState(
                players, next_new_player_id, capacity_reached=False)
        
        def step(state, remove, add):
            
            # remove
            players = jnp.where(remove, -1, state.players)
            
            # add
            # - find space for the newly added players
            available_locations, = jnp.nonzero(
                (players == -1), size=max_players, fill_value=max_players)
            add_n_hot = jnp.arange(max_players) < add
            add_locations = jnp.where(
                add_n_hot, available_locations, max_players)
            # - construct the new player ids 
            new_player_ids = jnp.where(
                add_n_hot, all_locations + state.next_new_player_id, -1)
            # - update the player ids
            valid = add_locations < max_players
            safe_locations = jnp.where(valid, add_locations, 0)
            players = players.at[safe_locations].set(
                jnp.where(valid, new_player_ids, players[safe_locations]))
            
            # update the next_new_player_id
            next_new_player_id = state.next_new_player_id + n
            
            available_count = jnp.sum(players == -1)
            capacity_reached = add > available_count
            next_state = IdentifiedPlayerState(
                players, next_new_player_id, capacity_reached)
            added_players = jnp.where(valid, players[safe_locations], -1)
            return next_state, add_locations, added_players
        
        def active(state):
            return state.players != -1
        
        def locations_to_ids(state, locations):
            return state.players[locations]
    
    return IdentifiedPlayerList

def birthday_player_list(max_players):

    @static_data
    class BirthdayPlayerListState:
        players : jnp.array
        current_time : int = 0
        capacity_reached : bool = False
    
    @static_functions
    class BirthdayPlayerList:
        
        def init(initial_players):
            n_hot = jnp.arange(max_players) < initial_players
            birthdays = jnp.where(n_hot, 0, -1)
            locations = jnp.where(n_hot, jnp.arange(max_players), -1)
            players = jnp.stack((birthdays, locations), axis=1)
            return BirthdayPlayerListState(players, capacity_reached=False)

        def step(
            state : BirthdayPlayerListState,
            remove : jnp.ndarray,
            add : int,
        ):
            
            # increment the current time
            current_time = state.current_time + 1
            
            # remove
            players = jnp.where(remove[:,None], -1, state.players)
            
            # add
            # - find locations for the newly added players
            available_locations, = jnp.nonzero(
                (players[...,0] == -1),
                size=max_players,
                fill_value=max_players,
            )
            add_locations = jnp.where(
                jnp.arange(max_players) < add, available_locations, max_players)
            available_count = jnp.sum(players[...,0] == -1)
            # - update the player birthdays and locations
            valid = add_locations < max_players
            safe_locations = jnp.where(valid, add_locations, 0)
            players = players.at[safe_locations,0].set(
                jnp.where(valid, current_time, players[safe_locations,0]))
            players = players.at[safe_locations,1].set(
                jnp.where(valid, safe_locations, players[safe_locations,1]))
            
            capacity_reached = add > available_count
            next_state = BirthdayPlayerListState(
                players, current_time, capacity_reached)
            added_players = jnp.where(
                valid[:,None], players[safe_locations], -1)
            return next_state, add_locations, added_players
        
        def active(state):
            return state.players[..., 0] != -1
        
        def locations_to_ids(state, locations):
            return state.players[locations]
    
    return BirthdayPlayerList

def birthday_hometown_player_list(max_players, home_town=0, axis_name=None):

    @static_data
    class BirthdayHometownPlayerListState:
        players : jnp.array
        current_time : int = 0
        home_town : int = 0
        capacity_reached : bool = False

    @static_functions
    class BirthdayHometownPlayerList:
        
        def init(initial_players):
            n_hot = jnp.arange(max_players) < initial_players
            birthdays = jnp.where(n_hot, 0, -1)
            locations = jnp.where(n_hot, jnp.arange(max_players), -1)
            if axis_name is None:
                home_town_value = jnp.array(home_town, dtype=jnp.int32)
            else:
                from jax import lax
                home_town_value = lax.axis_index(axis_name)
            home_towns = jnp.where(n_hot, home_town_value, -1)
            birth_index = locations
            players = jnp.stack(
                (birthdays, birth_index, home_towns, locations), axis=1)
            return BirthdayHometownPlayerListState(
                players,
                home_town=home_town_value,
                capacity_reached=False,
            )

        def step(
            state : BirthdayHometownPlayerListState,
            remove : jnp.ndarray,
            add : int,
        ):
            
            # increment the current time
            current_time = state.current_time + 1
            
            # remove
            players = jnp.where(remove[:,None], -1, state.players)
            
            # add
            # - find locations for the newly added players
            available_locations, = jnp.nonzero(
                (players[...,0] == -1),
                size=max_players,
                fill_value=max_players,
            )
            add_locations = jnp.where(
                jnp.arange(max_players) < add, available_locations, max_players)
            available_count = jnp.sum(players[...,0] == -1)
            # - update the player birthdays and locations
            valid = add_locations < max_players
            safe_locations = jnp.where(valid, add_locations, 0)
            players = players.at[safe_locations,0].set(
                jnp.where(valid, current_time, players[safe_locations,0]))
            # birth_index (column 1) is permanent; set only on birth
            players = players.at[safe_locations,1].set(
                jnp.where(valid, safe_locations, players[safe_locations,1]))
            players = players.at[safe_locations,2].set(
                jnp.where(valid, state.home_town, players[safe_locations,2]))
            # location (column 3) is current physical slot
            players = players.at[safe_locations,3].set(
                jnp.where(valid, safe_locations, players[safe_locations,3]))
            
            capacity_reached = add > available_count
            next_state = BirthdayHometownPlayerListState(
                players, current_time, state.home_town, capacity_reached)
            added_players = jnp.where(
                valid[:,None], players[safe_locations], -1)
            return next_state, add_locations, added_players
        
        def active(state):
            return state.players[..., 0] != -1
        
        def locations_to_ids(state, locations):
            return state.players[locations]
    
    return BirthdayHometownPlayerList

def player_family_tree(
    player_list,
    parents_per_child=1,
):
    
    @static_data
    class PlayerFamilyTreeState:
        player_state : jnp.array
        parents : jnp.array
    
    @static_functions
    class PlayerFamilyTree:
        
        def init(initial_players):
            player_state = player_list.init(initial_players)
            #parents = jax.vmap(init_player_list, out_axes=-1)(
            #    jnp.zeros((parents_per_child,), dtype=jnp.int32)).players
            # this is a for loop because parents_per_child will be small,
            # constant and I couldn't get vmap to put the new dimension in
            # the middle
            parents = jnp.stack(
                [player_list.init(initial_players).players
                    for _ in range(parents_per_child)],
                axis=1,
            )
            return PlayerFamilyTreeState(player_state, parents)
        
        def step(state, deaths, parent_locations):
            
            # determine how many children to make
            num_children = jnp.sum(
                (parent_locations[...,0] >= 0) &
                (parent_locations[...,0] < state.player_state.players.shape[0])
            )
            
            # remove dead players and add new ones
            player_state, child_locations, child_ids = player_list.step(
                state.player_state, deaths, num_children)
            
            # update the parent information
            new_parents = state.player_state.players[parent_locations]
            parents = state.parents.at[child_locations].set(new_parents)
            
            next_state = PlayerFamilyTreeState(player_state, parents)
            
            return next_state, child_locations, child_ids
        
        def active(state):
            return player_list.active(state.player_state)
        
        def locations_to_ids(state, locations):
            return player_list.locations_to_ids(state.player_state, locations)
        
        def locations_to_parents(state, locations):
            return state.parents[locations]
    
    return PlayerFamilyTree
