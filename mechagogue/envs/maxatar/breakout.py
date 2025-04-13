import jax.random as jrng
import jax.numpy as jnp

from mechagogue.static_dataclass import static_dataclass
from mechagogue.dp.pomdp import pomdp

TOTAL_CHANNELS = 4
PADDLE_CHANNEL = 0
BALL_CHANNEL = 1
TRAIL_CHANNEL = 2
BRICK_CHANNEL = 3

@static_dataclass
class BreakoutState:
    ball_y : int
    ball_x : int
    ball_dir : int
    pos : int
    brick_map : jnp.ndarray
    strike : bool
    last_x : int
    last_y : int
    terminal : bool

def init_state(key):
    ball_start = jrng.randint(key, (), 0, 2)
    initial_states = jnp.array([[0,2],[9,3]], dtype=jnp.int32)
    ball_x, ball_dir = initial_states[ball_start]
    
    starting_map = jnp.zeros((10,10), dtype=jnp.int32)
    starting_map = starting_map.at[1:4,:].set(1)
    
    return BreakoutState(
        ball_y=3,
        ball_x=ball_x,
        ball_dir=ball_dir,
        pos = 4,
        brick_map = starting_map,
        strike = False,
        last_x=ball_x,
        last_y=3,
        terminal=False,
    )

def transition(key, state, action):
    # Resolve player action
    pos = state.pos - (action == 2) + (action == 4)
    pos = jnp.clip(pos, min=0, max=9)
    
    # Update ball position
    last_x = state.ball_x
    last_y = state.ball_y
    new_x = (last_x
        + ((state.ball_dir == 1) | (state.ball_dir == 2))
        - ((state.ball_dir == 0) | (state.ball_dir == 3))
    )
    new_y = (last_y
        + ((state.ball_dir == 2) | (state.ball_dir == 3))
        - ((state.ball_dir == 0) | (state.ball_dir == 1))
    )
    
    strike_toggle = False
    ball_dir = state.ball_dir
    hit_wall = (new_x < 0) | (new_x > 9)
    hit_ceil = new_y < 0
    new_x = jnp.clip(new_x, min=0, max=9)
    
    transition_1 = jnp.array([1,0,3,2], dtype=jnp.int32)
    transition_2 = jnp.array([3,2,1,0], dtype=jnp.int32)
    ball_dir = ball_dir * ~hit_wall + transition_1[ball_dir] * hit_wall
    ball_dir = ball_dir * ~hit_ceil + transition_2[ball_dir] * hit_ceil
    
    hit_brick = state.brick_map[new_y, new_x] == 1
    hit_and_not_strike = hit_brick & ~state.strike
    strike_toggle |= hit_brick
    strike = hit_and_not_strike
    brick_map = state.brick_map.at[new_y, new_x].set(
        state.brick_map[new_y, new_x] * ~hit_and_not_strike +
        0 * hit_and_not_strike
    )
    new_y = (
        new_y * ~hit_and_not_strike +
        state.last_y * hit_and_not_strike
    )
    ball_dir = (
        ball_dir * ~hit_and_not_strike +
        transition_2[ball_dir] * hit_and_not_strike
    )
    
    hit_bottom = new_y == 9
    hit_bottom_and_empty = hit_bottom & (jnp.sum(brick_map) == 0)
    brick_map = (
        brick_map * ~hit_bottom_and_empty +
        brick_map.at[1:4,:].set(1) * hit_bottom_and_empty
    )
    
    terminal = (
        (new_y == 9) &
        (state.pos != new_x) &
        (pos != new_x)
    )
    #    next_state.ball_y == 9 and
    #    state.pos != next_state.ball_x and
    #    next_state.pos != next_state.ball_x
    #)
    
    return BreakoutState(
        ball_y=new_y,
        ball_x=new_x,
        ball_dir=ball_dir,
        pos=pos,
        brick_map=brick_map,
        strike=strike,
        last_x=last_x,
        last_y=last_y,
        terminal=terminal,
    )

def observe(state):
    obs = jnp.zeros((10,10,TOTAL_CHANNELS), dtype=jnp.bool)
    obs = obs.at[state.ball_y, state.ball_x, BALL_CHANNEL].set(1)
    obs = obs.at[9, state.pos, PADDLE_CHANNEL].set(1)
    obs = obs.at[state.last_y, state.last_x, TRAIL_CHANNEL].set(1)
    obs = obs.at[:,:,BRICK_CHANNEL].set(state.brick_map)
    return obs

def reward(state, next_state):
    return (jnp.sum(state.brick_map) - jnp.sum(next_state.brick_map)) == 1

def terminal(state):
    return state.terminal

def render(obs):
    assert obs.shape == (10,10,4)
    return (
        obs[...,0] * 1 +
        obs[...,1] * 2 +
        obs[...,2] * 3 +
        obs[...,3] * 4
    )

reset, step = pomdp(
    init_state,
    transition,
    observe,
    reward,
    terminal,
)
