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
        def _assign_slots(available_mask, incoming_mask):
            available_idx, = jnp.nonzero(
                available_mask, size=max_players, fill_value=-1)
            incoming_idx, = jnp.nonzero(
                incoming_mask, size=max_players, fill_value=-1)
            available_valid = available_idx >= 0
            incoming_valid = incoming_idx >= 0
            available_count = jnp.sum(available_valid)
            incoming_count = jnp.sum(incoming_valid)
            take = jnp.arange(max_players) < jnp.minimum(
                available_count, incoming_count)
            slot_idx = jnp.where(take, available_idx, -1)
            source_idx = jnp.where(take, incoming_idx, -1)
            capacity_reached = incoming_count > available_count
            return slot_idx, source_idx, capacity_reached

        def remove(state, remove):
            players = state.players & ~remove
            return AnonymousPlayerListState(
                players,
                capacity_reached=state.capacity_reached,
            )

        def place_payload(state, incoming_mask, incoming_players):
            available_mask = ~state.players
            slot_idx, source_idx, capacity_reached = (
                AnonymousPlayerList._assign_slots(
                    available_mask, incoming_mask)
            )
            valid = slot_idx >= 0
            safe_slots = jnp.where(valid, slot_idx, 0)
            players = state.players.at[safe_slots].set(
                jnp.where(valid, True, state.players[safe_slots]))
            added_players = jnp.where(valid, True, False)
            next_state = AnonymousPlayerListState(
                players,
                capacity_reached=state.capacity_reached | capacity_reached,
            )
            return next_state, slot_idx, source_idx, added_players
    
        def init(initial_players):
            players = jnp.zeros((max_players,), dtype=jnp.bool)
            players[:initial_players] = True
            return AnonymousPlayerListState(players, capacity_reached=False)

        def step(state, add, remove):
            
            # remove
            players = state.players & ~remove
            
            # add
            available_mask = ~players
            incoming_mask = jnp.arange(max_players) < add
            slot_idx, source_idx, capacity_reached = (
                AnonymousPlayerList._assign_slots(
                    available_mask, incoming_mask)
            )
            valid = slot_idx >= 0
            safe_slots = jnp.where(valid, slot_idx, 0)
            players = players.at[safe_slots].set(
                jnp.where(valid, True, players[safe_slots]))
            
            add_locations = jnp.where(valid, slot_idx, max_players)
            next_state = AnonymousPlayerListState(
                players, capacity_reached)
            added_players = jnp.where(valid, True, False)
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
        def _assign_slots(available_mask, incoming_mask):
            available_idx, = jnp.nonzero(
                available_mask, size=max_players, fill_value=-1)
            incoming_idx, = jnp.nonzero(
                incoming_mask, size=max_players, fill_value=-1)
            available_valid = available_idx >= 0
            incoming_valid = incoming_idx >= 0
            available_count = jnp.sum(available_valid)
            incoming_count = jnp.sum(incoming_valid)
            take = jnp.arange(max_players) < jnp.minimum(
                available_count, incoming_count)
            slot_idx = jnp.where(take, available_idx, -1)
            source_idx = jnp.where(take, incoming_idx, -1)
            capacity_reached = incoming_count > available_count
            return slot_idx, source_idx, capacity_reached

        def remove(state, remove):
            players = jnp.where(remove, -1, state.players)
            return IdentifiedPlayerListState(
                players,
                state.next_new_player_id,
                capacity_reached=state.capacity_reached,
            )

        def place_payload(state, incoming_mask, incoming_players):
            available_mask = state.players == -1
            slot_idx, source_idx, capacity_reached = (
                IdentifiedPlayerList._assign_slots(
                    available_mask, incoming_mask)
            )
            valid = slot_idx >= 0
            safe_slots = jnp.where(valid, slot_idx, 0)
            safe_src = jnp.where(valid, source_idx, 0)
            placed = incoming_players[safe_src]
            players = state.players.at[safe_slots].set(
                jnp.where(valid, placed, state.players[safe_slots]))
            added_players = jnp.where(valid, placed, -1)
            next_state = IdentifiedPlayerListState(
                players,
                state.next_new_player_id,
                capacity_reached=state.capacity_reached | capacity_reached,
            )
            return next_state, slot_idx, source_idx, added_players
        
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
            available_mask = players == -1
            incoming_mask = jnp.arange(max_players) < add
            slot_idx, source_idx, capacity_reached = (
                IdentifiedPlayerList._assign_slots(
                    available_mask, incoming_mask)
            )
            valid = slot_idx >= 0
            safe_slots = jnp.where(valid, slot_idx, 0)
            safe_src = jnp.where(valid, source_idx, 0)
            new_player_ids = state.next_new_player_id + jnp.arange(max_players)
            incoming_players = jnp.where(incoming_mask, new_player_ids, -1)
            placed = incoming_players[safe_src]
            players = players.at[safe_slots].set(
                jnp.where(valid, placed, players[safe_slots]))
            
            next_new_player_id = state.next_new_player_id + add
            
            add_locations = jnp.where(valid, slot_idx, max_players)
            next_state = IdentifiedPlayerListState(
                players, next_new_player_id, capacity_reached)
            added_players = jnp.where(valid, placed, -1)
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
        def _assign_slots(available_mask, incoming_mask):
            available_idx, = jnp.nonzero(
                available_mask, size=max_players, fill_value=-1)
            incoming_idx, = jnp.nonzero(
                incoming_mask, size=max_players, fill_value=-1)
            available_valid = available_idx >= 0
            incoming_valid = incoming_idx >= 0
            available_count = jnp.sum(available_valid)
            incoming_count = jnp.sum(incoming_valid)
            take = jnp.arange(max_players) < jnp.minimum(
                available_count, incoming_count)
            slot_idx = jnp.where(take, available_idx, -1)
            source_idx = jnp.where(take, incoming_idx, -1)
            capacity_reached = incoming_count > available_count
            return slot_idx, source_idx, capacity_reached

        def remove(state, remove):
            players = jnp.where(remove[:, None], -1, state.players)
            return BirthdayPlayerListState(
                players,
                current_time=state.current_time,
                capacity_reached=state.capacity_reached,
            )

        def place_payload(state, incoming_mask, incoming_players):
            available_mask = state.players[..., 0] == -1
            slot_idx, source_idx, capacity_reached = (
                BirthdayPlayerList._assign_slots(
                    available_mask, incoming_mask)
            )
            valid = slot_idx >= 0
            safe_slots = jnp.where(valid, slot_idx, 0)
            safe_src = jnp.where(valid, source_idx, 0)
            placed = incoming_players[safe_src]
            placed = placed.at[:, 1].set(jnp.where(valid, safe_slots, placed[:, 1]))
            players = state.players.at[safe_slots].set(
                jnp.where(valid[:, None], placed, state.players[safe_slots]))
            added_players = jnp.where(valid[:, None], placed, -1)
            next_state = BirthdayPlayerListState(
                players,
                current_time=state.current_time,
                capacity_reached=state.capacity_reached | capacity_reached,
            )
            return next_state, slot_idx, source_idx, added_players
        
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
            available_mask = players[..., 0] == -1
            incoming_mask = jnp.arange(max_players) < add
            slot_idx, source_idx, capacity_reached = (
                BirthdayPlayerList._assign_slots(
                    available_mask, incoming_mask)
            )
            valid = slot_idx >= 0
            safe_slots = jnp.where(valid, slot_idx, 0)
            safe_src = jnp.where(valid, source_idx, 0)

            incoming_birthdays = jnp.where(incoming_mask, current_time, -1)
            incoming_loc = jnp.where(incoming_mask, -1, -1)
            incoming_players = jnp.stack(
                (incoming_birthdays, incoming_loc),
                axis=1,
            )
            placed = incoming_players[safe_src]
            placed = placed.at[:, 1].set(jnp.where(valid, safe_slots, placed[:, 1]))
            players = players.at[safe_slots].set(
                jnp.where(valid[:, None], placed, players[safe_slots]))
            
            add_locations = jnp.where(valid, slot_idx, max_players)
            next_state = BirthdayPlayerListState(
                players, current_time, capacity_reached)
            added_players = jnp.where(
                valid[:, None], placed, -1)
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
        def _assign_slots(available_mask, incoming_mask):
            available_idx, = jnp.nonzero(
                available_mask, size=max_players, fill_value=-1)
            incoming_idx, = jnp.nonzero(
                incoming_mask, size=max_players, fill_value=-1)
            available_valid = available_idx >= 0
            incoming_valid = incoming_idx >= 0
            available_count = jnp.sum(available_valid)
            incoming_count = jnp.sum(incoming_valid)
            take = jnp.arange(max_players) < jnp.minimum(
                available_count, incoming_count)
            slot_idx = jnp.where(take, available_idx, -1)
            source_idx = jnp.where(take, incoming_idx, -1)
            capacity_reached = incoming_count > available_count
            return slot_idx, source_idx, capacity_reached

        def remove(state, remove):
            players = jnp.where(remove[:, None], -1, state.players)
            return BirthdayHometownPlayerListState(
                players,
                current_time=state.current_time,
                home_town=state.home_town,
                capacity_reached=state.capacity_reached,
            )

        def place_payload(state, incoming_mask, incoming_players):
            available_mask = state.players[..., 0] == -1
            slot_idx, source_idx, capacity_reached = (
                BirthdayHometownPlayerList._assign_slots(
                    available_mask, incoming_mask)
            )
            valid = slot_idx >= 0
            safe_slots = jnp.where(valid, slot_idx, 0)
            safe_src = jnp.where(valid, source_idx, 0)
            placed = incoming_players[safe_src]
            placed = placed.at[:, 3].set(jnp.where(valid, safe_slots, placed[:, 3]))
            players = state.players.at[safe_slots].set(
                jnp.where(valid[:, None], placed, state.players[safe_slots]))
            added_players = jnp.where(valid[:, None], placed, -1)
            next_state = BirthdayHometownPlayerListState(
                players,
                current_time=state.current_time,
                home_town=state.home_town,
                capacity_reached=state.capacity_reached | capacity_reached,
            )
            return next_state, slot_idx, source_idx, added_players
        
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
            available_mask = players[..., 0] == -1
            incoming_mask = jnp.arange(max_players) < add
            slot_idx, source_idx, capacity_reached = (
                BirthdayHometownPlayerList._assign_slots(
                    available_mask, incoming_mask)
            )
            valid = slot_idx >= 0
            safe_slots = jnp.where(valid, slot_idx, 0)
            safe_src = jnp.where(valid, source_idx, 0)

            incoming_birthdays = jnp.where(incoming_mask, current_time, -1)
            incoming_birth_index = jnp.where(incoming_mask, -1, -1)
            incoming_home = jnp.where(incoming_mask, state.home_town, -1)
            incoming_loc = jnp.where(incoming_mask, -1, -1)
            incoming_players = jnp.stack(
                (incoming_birthdays, incoming_birth_index,
                 incoming_home, incoming_loc),
                axis=1,
            )
            placed = incoming_players[safe_src]
            placed = placed.at[:, 1].set(jnp.where(valid, safe_slots, placed[:, 1]))
            placed = placed.at[:, 3].set(jnp.where(valid, safe_slots, placed[:, 3]))
            players = players.at[safe_slots].set(
                jnp.where(valid[:, None], placed, players[safe_slots]))

            add_locations = jnp.where(valid, slot_idx, max_players)
            next_state = BirthdayHometownPlayerListState(
                players,
                current_time=current_time,
                home_town=state.home_town,
                capacity_reached=capacity_reached,
            )
            added_players = jnp.where(valid[:, None], placed, -1)
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

        def remove(state, remove_mask):
            player_state = player_list.remove(state.player_state, remove_mask)
            parents = jnp.where(
                remove_mask[:, None, None], -1, state.parents)
            return PlayerFamilyTreeState(player_state, parents)

        def place_payload(state, incoming_mask, incoming_players, incoming_parents):
            player_state, slot_idx, source_idx, added_players = (
                player_list.place_payload(
                    state.player_state, incoming_mask, incoming_players)
            )
            valid = slot_idx >= 0
            safe_slots = jnp.where(valid, slot_idx, 0)
            safe_src = jnp.where(valid, source_idx, 0)
            placed_parents = incoming_parents[safe_src]
            parents = state.parents.at[safe_slots].set(
                jnp.where(valid[:, None, None], placed_parents, state.parents[safe_slots]))
            next_state = PlayerFamilyTreeState(player_state, parents)
            return next_state, slot_idx, source_idx, added_players
        
        def active(state):
            return player_list.active(state.player_state)
        
        def locations_to_ids(state, locations):
            return player_list.locations_to_ids(state.player_state, locations)
        
        def locations_to_parents(state, locations):
            return state.parents[locations]
    
    return PlayerFamilyTree
