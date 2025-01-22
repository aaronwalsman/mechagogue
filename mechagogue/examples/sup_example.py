import jax
import jax.random as jrng

from mechagogue.sup.sup import sup
import mechagogue.sup.tasks.classify as classify
from mechagogue.nn.mlp import mlp
from mechagogue.opt.sgd import sgd
from mechagogue.data.example_data import make_example_data

def train():
    
    key = jrng.key(1234)
    
    in_channels = 16
    hidden_channels = 32
    num_classes = 200
    num_train = 50000
    num_test = 10000
    num_epochs = 20
    batch_size = 64
    
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
    init_sgd, optimize_sgd = sgd(
        learning_rate=3e-4,
        momentum=0.9,
    )
    
    init_sup, train_sup, test_sup = sup(
        init_mlp,
        model_mlp,
        init_sgd,
        optimize_sgd,
        classify.loss,
        classify.accuracy,
        batch_size=batch_size,
    )
    
    key, init_key = jrng.split(key)
    model_params, optimizer_params = init_sup(init_key)
    
    def epoch(params, key):
        model_params, optimizer_params = params
        train_key, test_key = jrng.split(key)
        model_params, optimizer_params, losses = train_sup(
            train_key, train_x, train_y, model_params, optimizer_params)
        
        accuracy = test_sup(test_key, test_x, test_y, model_params)
        
        return (model_params, optimizer_params), (losses, accuracy)
    
    epoch_keys = jrng.split(key, num_epochs)
    (model_params, optimizer_params), (losses, accuracy) = jax.lax.scan(
        epoch,
        (model_params, optimizer_params),
        epoch_keys,
    )
    losses = losses.reshape(-1)
    
    return model_params, optimizer_params, losses, accuracy

if __name__ == '__main__':
    train = jax.jit(train)
    model_params, optimizer_params, losses, accuracy = train()
    
    print(accuracy)
