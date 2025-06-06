import jax
import jax.random as jrng
import jax.numpy as jnp

from mechagogue.static_dataclass import static_dataclass
from mechagogue.dp.pomdp import pomdp

TOTAL_CHANNELS = 4
PADDLE_CHANNEL = 0
BALL_CHANNEL = 1
TRAIL_CHANNEL = 2
BRICK_CHANNEL = 3

# Difficulty ramping settings
MAX_SPEED = 3          # cap (MinAtar uses 3)
SPEED_STEP = 1         # delta each time bricks respawn

# Direction codes
# 0 : left-up   ◤
# 1 : right-up  ◥
# 2 : right-down◢
# 3 : left-down ◣
TRANSITION_WALL  = jnp.array([1, 0, 3, 2], dtype=jnp.int32)  # reflect on vertical wall
TRANSITION_CEIL  = jnp.array([3, 2, 1, 0], dtype=jnp.int32)  # reflect on horizontal wall
TRANSITION_BRICK = TRANSITION_CEIL                           # identical to ceiling bounce

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
    speed : int               # how many cells the ball moves / env-step
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
        speed=1,
        terminal=False,
    )

# ============================================================
# 1.  One-cell update of the ball and environment (“inner step”)
# ============================================================
def _single_ball_update(state, pos):
    """
    Advances the ball exactly one grid cell 
    and returns a new BreakoutState (speed unchanged).
    `pos` is the current paddle position after resolving the action.
    """
    # Save previous ball coordinates
    last_x, last_y = state.ball_x, state.ball_y

    # Move ball one cell according to its current direction
    new_x = last_x + ((state.ball_dir == 1) | (state.ball_dir == 2)) \
                     - ((state.ball_dir == 0) | (state.ball_dir == 3))
    new_y = last_y + ((state.ball_dir == 2) | (state.ball_dir == 3)) \
                     - ((state.ball_dir == 0) | (state.ball_dir == 1))

    # Reflect off side walls (x) and ceiling (y == -1)
    hit_wall = (new_x < 0) | (new_x > 9)
    hit_ceil = new_y < 0
    new_x = jnp.clip(new_x, 0, 9)
    ball_dir = state.ball_dir
    ball_dir = jnp.where(hit_wall, TRANSITION_WALL[ball_dir], ball_dir)
    ball_dir = jnp.where(hit_ceil, TRANSITION_CEIL[ball_dir], ball_dir)

    # Brick collision
    hit_brick = state.brick_map[new_y, new_x] == 1
    hit_and_not_strike = hit_brick & ~state.strike                 # prevent multi-break
    brick_map = state.brick_map.at[new_y, new_x].set(
        state.brick_map[new_y, new_x] * ~hit_and_not_strike
    )
    strike = hit_and_not_strike
    new_y = jnp.where(hit_and_not_strike, last_y, new_y)            # bounce vertically
    ball_dir = jnp.where(hit_and_not_strike, TRANSITION_BRICK[ball_dir], ball_dir)

    # Paddle / bottom interaction
    hit_bottom = new_y == 9
    hit_bottom_and_empty = hit_bottom & (jnp.sum(brick_map) == 0)
    brick_map = jax.lax.cond(
        hit_bottom_and_empty,
        lambda _: brick_map.at[1:4, :].set(1),     # respawn rows
        lambda _: brick_map,
        operand=None,
    )

    hit_paddle = (new_y == 9) & ((state.pos == new_x) | (pos == new_x))
    ball_dir = jnp.where(hit_paddle, TRANSITION_CEIL[ball_dir], ball_dir)
    new_y = jnp.where(hit_paddle, 8, new_y)         # lift the ball after paddle hit

    terminal = (new_y == 9) & ~hit_paddle           # ball missed → episode over

    # Assemble next state (speed unchanged)
    return state.replace(
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

# ============================================================
# 2.  External transition (moves the ball `speed` cells / step)
# ============================================================
def transition(key, state, action, *, ramping=True):
    """
    Environment-level step:
    • resolves player action → paddle position
    • repeats `_single_ball_update` `state.speed` times
    • optionally increases `speed` when bricks respawn
    Returns a new BreakoutState.
    `key` is ignored (kept for POMDP signature compatibility).
    """
    # Update paddle according to action (noop / left / right)
    pos = state.pos - (action == 2) + (action == 4)
    pos = jnp.clip(pos, 0, 9)

    # Advance the ball `speed` times (speed ∈ {1,2,3})
    def _body(_, inner_state):
        return _single_ball_update(inner_state, pos)
    state_after = jax.lax.fori_loop(0, state.speed, _body, state)

    # Difficulty ramping: raise speed whenever new bricks appear
    if ramping:
        bricks_prev = jnp.sum(state.brick_map)
        bricks_now  = jnp.sum(state_after.brick_map)
        spawned = (bricks_prev == 0) & (bricks_now > 0)
        new_speed = jnp.minimum(
            state_after.speed + SPEED_STEP * spawned.astype(jnp.int32),
            MAX_SPEED,
        )
        state_after = state_after.replace(speed=new_speed)

    return state_after

def observe(state):
    obs = jnp.zeros((10,10,TOTAL_CHANNELS), dtype=jnp.bool_)
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

# -------------------------------------------------------------------
# Factory: build a (reset, step) pair with or without ramping
# -------------------------------------------------------------------
def make_env(*, ramping: bool = True):
    """
    Returns (reset, step) functions whose `transition` uses the
    supplied `ramping` flag.

    Example:
        reset_env, step_env = make_env(ramping=False)  # fixed speed
    """
    # bind the flag via a closure / partial
    def _transition(key, state, action):
        return transition(key, state, action, ramping=ramping)

    return pomdp(
        init_state,
        _transition,     # ← uses the chosen ramping behaviour
        observe,
        reward,
        terminal,
    )
