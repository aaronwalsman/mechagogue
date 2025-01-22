import jax.numpy as jnp
import jax.nn as jnn

from mechagogue.static_dataclass import static_dataclass
from mechagogue.nn.initializers import kaiming, zero

raise Exception('Deprecated')

@static_dataclass
class NNConfig:
    dtype = jnp.float32

DEFAULT_CONFIG = NNConfig()
