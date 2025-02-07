import jax.random as jrng

from mechagogue.dp.pomdp import pomdp
from mechagogue.static_dataclass import static_dataclass

@static_dataclass
class GotoConfig:
    map_size = (64,64)
    dense_reward = True

def gogo(config):
    def init(key):
        position_key, target_key = jrng.split(key)
        position = jrng.randint(
            position_key,
            shape=(2,),
            minval=jnp.zeros
        target = jnp.zeros(len(config.map_size), dtype=jnp.int32)
