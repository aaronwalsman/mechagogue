import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax

class NomPolicy(nn.Module):
    activation: str = 'tanh'
    
    @nn.compact
    def __call__(self, observation):
        if self.activation == 'relu':
            activation = nn.relu
        
        elif self.activation == 'tanh':
            activation = nn.tanh
        else:
            raise NotImplementedError

        view = observation.view
        *b,h,w = view.shape

        view = nn.Embed(
            3,
            8,
            embedding_init=orthogonal(jnp.sqrt(2))
        )(view)
        view = jnp.reshape(view, (*b,h*w*8,))

        view = nn.Dense(
            64,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.)
        )(view)

        health = nn.Dense(
            64,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.)
        )(observation.health)

        x = view + health
        
        x = nn.Dense(
            64,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.)
        )(x)
        x = activation(x)
        x = nn.Dense(
            64,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.)
        )(x)
        x = activation(x)

        forward_logits = nn.Dense(
            2,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.)
        )(x)
        forward_distribution = distrax.Categorical(logits=forward_logits)

        rotate_logits = nn.Dense(
            4,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.)
        )(x)
        rotate_distribution = distrax.Categorical(logits=rotate_logits)

        action_distribution = distrax.Joint(
            NomAction(forward_distribution, rotate_distribution))

        return action_distribution

def nom_policy():
    policy = NomPolicy()
    weights = policy.init(key, obs)
    
