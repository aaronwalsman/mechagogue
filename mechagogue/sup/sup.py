import jax
import jax.random as jrng

class SupervisedLearningParams:
    batch_size: int = 64

def supervised_learning(
    params,
    initialize_weights,
    forward,
    optimize,
    loss_function,
):
    
    reset_supervised_learning = initialize_weights
    
    def step_supervised_learning(
        key,
        x,
        y,
        weights,
    ):
        num_examples, *x_shape = x.shape
        num_examples, *y_shape = y.shape
        num_batches = num_examples // batch_size
        clipped_examples = num_batches * batch_size
        
        key, shuffle_key = jrng.split(key)
        shuffled_x = jrng.permutation(shuffle_key, x)[:clipped_examples]
        shuffled_y = jrng.permutation(shuffle_key, y)[:clipped_examples]
        batched_x = shuffled_x.reshape(num_batches, batch_size, *x_shape)
        batched_y = shuffled_y.reshape(num_batches, batch_size, *y_shape)
        
        def train_batch(weights, batch):
            key, x, y = batch
            
            def compute_loss(key, x, y, weights):
                forward_key, loss_key = jrng.split(key)
                x = forward(key, x, weights)
                return loss_function(key, x, y)
            
            loss_and_grad = jax.value_and_grad(compute_loss, argnums=3)
            key, loss_key = jrng.split(key)
            loss, grad = loss_and_grad(loss_key, x, y, weights)
            
            key, optimizer_key = jrng.split(key)
            weights = optimize(weights, optimizer_parameters, grad)
            
            return weights, loss
        
        batch_keys = jrng.split(key, num_batches+1)
        key, batch_keys = batch_keys[0], batch_keys[1:]
        weights, losses = jax.lax.scan(
            train_batch, weights, (batch_keys, batched_x, batched_y))
        
        return weights, losses
    
    def eval_supervised_learning(
        key,
        x,
        y,
        weights,
    ):
        key, forward_key = jrng.split(key)
        x = forward(key, x, weights)
        loss = loss_function(loss_key, x, y, weights)
