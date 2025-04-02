import jax
import jax.random as jrng

from mechagogue.sup.ga import ga, GAParams
import mechagogue.sup.tasks.classify as classify
from mechagogue.nn.mlp import mlp
from mechagogue.breed.normal import normal_mutate
from mechagogue.tree import tree_getitem
from mechagogue.data.example_data import make_example_data

def train():
    
    key = jrng.key(1234)
    
    in_channels = 16
    hidden_channels = 32
    num_classes = 10
    num_train = 50000
    num_test = 10000
    num_epochs = 20
    batch_size = 64
    batches_per_step = 16
    learning_rate = 3e-3
    population_size = 256
    
    key, data_key = jrng.split(key)
    x, y = make_example_data(
        data_key, 
        num_train + num_test,
        num_classes=num_classes,
        in_channels=in_channels,
        data_noise=0.1,
    )
    train_x, test_x = x[:num_train], x[num_train:]
    train_y, test_y = y[:num_train], y[num_train:]
    
    init_mlp, model_mlp = mlp(
        hidden_layers=4,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=num_classes
    )
    breed = normal_mutate(
        learning_rate=learning_rate,
    )
    
    ga_config = GAParams(
        batch_size=batch_size,
        batches_per_step=batches_per_step,
        population_size=population_size,
    )
    init_ga, train_ga, test_ga = ga(
        ga_config,
        init_mlp,
        model_mlp,
        breed,
        classify.loss,
        classify.accuracy,
    )
    
    key, init_key = jrng.split(key)
    model_params = init_ga(init_key)
    
    def epoch(model_params, key):
        train_key, test_key = jrng.split(key)
        model_params, fitness = train_ga(
            train_key, train_x, train_y, model_params)
        
        _, elites = jax.lax.top_k(fitness[-1], ga_config.elites)
        elite_model_params = tree_getitem(model_params, elites)
        
        accuracy = test_ga(test_key, test_x, test_y, elite_model_params)
        jax.debug.print('Accuracy: {a}', a=accuracy)
        
        return model_params, (fitness, accuracy)
    
    epoch_keys = jrng.split(key, num_epochs)
    model_params, (fitness, accuracy) = jax.lax.scan(
        epoch, model_params, epoch_keys)
    fitness = fitness.reshape(-1)
    
    return model_params, fitness, accuracy

if __name__ == '__main__':
    train = jax.jit(train)
    model_params, losses, accuracy = train()
