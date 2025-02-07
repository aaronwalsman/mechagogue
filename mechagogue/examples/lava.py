import jax.random as jrng

from mechagogue.dp.pomdp import pomdp
from mechagogue.static_dataclass import static_dataclass

@static_dataclass
class LavaConfig:
    max_agents = 64
    map_size = (64,64)
    dense_reward = True

def lava(config):
    def init(key):
        position_key, volcano_key = jrng.split(key)
        positions = jrng.randint(
            position_key,
            shape=(config.max_agents,2),
            minval=jnp.zeros(2, dtype=jnp.int32),
            maxval=jnp.array(map_size)
        )
