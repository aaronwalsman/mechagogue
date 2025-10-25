from typing import Optional

import jax
import jax.numpy as jnp

from mechagogue.static import static_data, static_functions

def insert_players(
    players : jax.Array,
    active : jax.Array,
    n : int,
    ids : jax.Array,
):
    max_players = active.shape[0]
    available_locations, = jnp.nonzero(
        ~active, size=max_players, fill_value=max_players)
    add_n_hot = jnp.arange(max_players) < n
    add_locations = jnp.where(
        add_n_hot, available_locations, max_players)
    players = players.at[add_locations].set(ids)
    return players, add_locations

def remove_players(
    players : jax.Array,
    remove_mask : jax.Array,
    empty_value : jax.Array,
):
    return jnp.where(remove_mask, empty_value, players)

def make_anonymous_player_list(max_players):
    
    @static_functions
    class AnonymousPlayerList:
        
        empty_player = False
        max_players = max_players
        
        @static_data
        class State:
            players : jax.Array
    
        def init(
            initial_players : int
        ):
            players = jnp.zeros((max_players,), dtype=jnp.bool)
            players = players.at[:initial_players].set(True)
            return AnonymousPlayerList.State(players)
        
        def insert(
            state : State,
            n : int,
            ids : bool | jax.Array,
        ):
            active = AnonymousPlayerList.active(state)
            players, insert_locations = insert_players(
                state.players, active, n, ids)
            return state.replace(players=players), insert_locations
        
        def remove(
            state : State,
            remove_mask : jax.Array,
        ):
            players = remove_players(
                state.players,
                remove_mask,
                AnyonymousPlayerList.empty,
            )
            return state.replace(players=players)
        
        def step(
            state : State,
            remove : jax.Array,
            add : int,
        ):
            state = AnonymousPlayerList.remove(state, remove)
            state, add_locations = AnonymousPlayerList.insert(state, add, True)
            
            return state, add_locations, state.players[add_locations]
        
        def active(
            state : State,
        ):
            return state.players
        
        def get_ids(
            state : State,
            locations : jax.Array,
        ):
            return state.players[locations]
    
    return AnonymousPlayerList

def make_integer_player_list(max_players):
    
    @static_functions
    class IntegerPlayerList:
        
        empty_player = -1
        max_players = max_players
        
        @static_data
        class State:
            players : jax.Array
            next_new_player_id : int
        
        def init(
            initial_players : int
        ):
            players = jnp.full(
                (max_players,), IntegerPlayerList.empty_player, dtype=jnp.int32)
            players = players.at[:initial_players].set(
                jnp.arange(initial_players))
            next_new_player_id = initial_players
            
            return IntegerPlayerList.State(players, next_new_player_id)
        
        def insert(
            state : State,
            n : jax.Array,
            ids : jax.Array,
        ):
            active = IntegerPlayerList.active(state)
            players, insert_locations = insert_players(
                state.players, active, n, ids)
            return state.replace(players=players), insert_locations
        
        def remove(
            state : State,
            remove_mask : jax.Array,
        ):
            players = remove_players(
                state.players, remove_mask, IntegerPlayerList.empty_player)
            return state.replace(players=players)
        
        def step(
            state : State,
            remove : jax.Array = None,
            add : int = None,
        ):
            
            state = IntegerPlayerList.remove(state, remove)
            state, add_locations = IntegerPlayerList.insert(
                state,
                add,
                jnp.arange(max_players) + state.next_new_player_id,
            )
            state = state.replace(
                next_new_player_id=state.next_new_player_id + add)
            
            return state, add_locations, state.players[add_locations]
        
        def active(
            state : State,
        ):
            return state.players != IntegerPlayerList.empty_player
        
        def get_ids(
            state : State,
            locations : jax.Array,
        ):
            return state.players[locations]
    
    return IntegerPlayerList

def make_birthday_player_list(max_players, location_offset=0):

    @static_functions
    class BirthdayPlayerList:
        
        empty_player = jnp.array([-1,-1])
        max_players = max_players
        
        @static_data
        class State:
            players : jax.Array
            current_time : int = 0
        
        def init(initial_players):
            n_hot = jnp.arange(max_players) < initial_players
            birthdays = jnp.where(n_hot, 0, -1)
            locations = jnp.where(
                n_hot, jnp.arange(max_players) + location_offset, -1)
            players = jnp.stack((birthdays, locations), axis=1)
            return BirthdayPlayerList.State(players)
        
        def insert(
            state : State,
            n : jax.Array,
            ids : jax.Array,
        ):
            active = BirthdayPlayerList.active(state)
            players, insert_locations = insert_players(
                state.players, active, n, ids)
            return state.replace(players=players), insert_locations
        
        def remove(
            state : State,
            remove_mask : jax.Array,
        ):
            players = remove_players(
                state.players,
                remove_mask[:,None],
                BirthdayPlayerList.empty_player,
            )
            return state.replace(players=players)
        
        def step(
            state : State,
            remove_mask : jax.Array,
            add : int,
        ):
            
            # increment the current time
            current_time = state.current_time + 1
            
            # remove
            state = BirthdayPlayerList.remove(state, remove_mask)
            
            # add
            active = BirthdayPlayerList.active(state)
            birthdays = state.players[:,0]
            birthdays, add_locations = insert_players(
                birthdays, active, add, current_time)
            players = state.players.at[:,0].set(birthdays)
            players = players.at[add_locations,1].set(add_locations)
            state = state.replace(players=players, current_time=current_time)
            return state, add_locations, players[add_locations]
        
        def active(state):
            return state.players[..., 0] != -1
        
        def get_ids(state, locations):
            return state.players[locations]
    
    return BirthdayPlayerList

def make_player_family_tree(
    player_list,
    parents_per_child=1,
):
    
    @static_functions
    class PlayerFamilyTree:
        
        @static_data
        class State:
            player_state : player_list.State
            parents : jax.Array
        
        def init(initial_players):
            player_state = player_list.init(initial_players)
            # initialize parents to empty
            parent_shape = (
                player_list.max_players,
                parents_per_child,
                *jnp.shape(player_list.empty_player)
            )
            parents = jnp.full(parent_shape, player_list.empty_player)
            return PlayerFamilyTree.State(player_state, parents)
        
        def insert(state, n, ids, parent_ids):
            player_state, insert_locations = player_list.insert(
                state.player_state, n, ids)
            parents = state.parents.at[insert_locations].set(parent_ids)
            return state.replace(player_state=player_state, parents=parents)
        
        def remove(state, remove_mask):
            player_state = player_list.remove(state.player_state, remove_mask)
            parents = state.parents.at[remove_mask].set(
                player_list.empty_player)
            return state.replace(player_state=player_state, parents=parents)
        
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
            new_parents = player_list.get_ids(
                state.player_state, parent_locations)
            parents = state.parents.at[child_locations].set(new_parents)
            
            next_state = PlayerFamilyTree.State(player_state, parents)
            
            return next_state, child_locations, child_ids
        
        def active(state):
            return player_list.active(state.player_state)
        
        def get_ids(state, locations):
            return player_list.get_ids(state.player_state, locations)
        
        def get_parents(state, locations):
            return state.parents[locations]
    
    return PlayerFamilyTree

if __name__ == '__main__':
    
    import jax.random as jrng
    
    key = jrng.key(1234)
    
    initial_players = 8
    max_players = 16
    dead_per_turn = 4
    children_per_turn = 4
    steps = 4
    
    player_list = make_integer_player_list(max_players)
    player_family_tree = make_player_family_tree(player_list)
    
    family_tree_state = player_family_tree.init(initial_players)
    
    for i in range(steps):
        print('------')
        print(i)
        print(family_tree_state.player_state.players)
        active, = jnp.nonzero(player_family_tree.active(family_tree_state))
        key, deaths_key = jrng.split(key)
        death_indices = jrng.choice(
            deaths_key, active, (dead_per_turn,), replace=False)
        deaths = jnp.zeros(
            max_players, dtype=jnp.bool).at[death_indices].set(True)
        print('deaths', deaths)
        key, parents_key = jrng.split(key)
        parent_ids = jrng.choice(
            parents_key, active, (children_per_turn,), replace=False)
        parents = jnp.full(
            (max_players,), max_players).at[:children_per_turn].set(parent_ids)
        print('parents', parents)
        parents = parents[:,None]
        family_tree_state, _, _ = player_family_tree.step(
            family_tree_state, deaths, parents)
        print('abs parents', family_tree_state.parents)
        print(family_tree_state.player_state.players)
    
    remove_mask = jnp.zeros(max_players, dtype=jnp.bool)
    remove_mask = remove_mask.at[jnp.array([3,4])].set(True)
    family_tree_state = player_family_tree.remove(
        family_tree_state, remove_mask)
    print(family_tree_state.player_state.players)
    print(family_tree_state.parents)
    
    insert_ids = jnp.array([
        82,
        91,
        95,
        182,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    ])
    
    insert_parents = jnp.array([
        [81],
        [90],
        [94],
        [181],
        [-1,],
        [-1,],
        [-1,],
        [-1,],
        [-1,],
        [-1,],
        [-1,],
        [-1,],
        [-1,],
        [-1,],
        [-1,],
        [-1,],
    ])
    family_tree_state = player_family_tree.insert(
        family_tree_state,
        4,
        insert_ids,
        insert_parents,
    )
    
    print(family_tree_state.player_state.players)
    print(family_tree_state.parents)
