import argparse

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.commandline import commandline_interface
from mechagogue.static import static_data
from mechagogue.sup.sup import SupParams, sup
import mechagogue.sup.tasks.classify as classify
from mechagogue.nn.linear import linear_layer, grouped_linear_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.mlp import mlp
from mechagogue.nn.nonlinear import ReluLayer
from mechagogue.nn.permute import permute_layer
from mechagogue.nn.debug import breakpoint_layer
from mechagogue.optim.adamw import adamw
import mechagogue.data.maxnist as maxnist

# specify params and read them from the commandline
@commandline_interface
@static_data
class MaxnistParams:
    
    # rng
    seed : int = 1234
    
    # data   
    digits : int = 1
    train_examples : int = 5000
    test_examples : int = 1000
    data_noise : float = 0.025
    
    # model
    model_class : str = 'mlp'
    hidden_layers : int = 1
    hidden_channels : int = 256
    
    # optim
    learning_rate : float = 3e-3
    
    # training
    #epochs : int = 40
    visualize_examples : int = 0
    trainer_params : SupervisedBackpropParams = SupervisedBackpropParams()
        batch_size=64,
        shuffle=True,
    )
    simulate_params : SimulateParams = SimulateParams(
        epochs : 10,
        steps_per_epoch : 1,
    )

def main(params):
    # initialize the random key
    key = jrng.key(params.seed)

    # make the train and test data
    key, train_key, test_key = jrng.split(key, 3)
    train_x, train_y = maxnist.make_multidigit_dataset(
        train_key,
        params.digits,
        params.train_examples,
        include_shifts=True,
        noise=params.data_noise,
    )
    test_x, test_y = maxnist.make_multidigit_dataset(
        test_key,
        params.digits,
        params.test_examples,
        include_shifts=True,
        noise=params.data_noise,
    )
    
    # build the data loaders
    train_loader = on_device_loader(
        train_x, train_y, params.batch_size, shuffle=True, auto_reset=True)
    test_loader = on_device_loader(
        test_x, test_y, params.batch_size, shuffle=False, auto_reset=False)
    
    # build the model
    in_channels = 49*params.digits
    num_classes = 10**params.digits
    
    if params.model_class == 'mlp':
        model = layer_sequence((
            make_layer(forward=lambda x : x.reshape(-1,in_channels)), # flatten
            mlp(
                hidden_layers=params.hidden_layers,
                in_channels=in_channels,
                hidden_channels=params.hidden_channels,
                out_channels=num_classes,
                p_dropout=0.1,
            ),
        ))
    else:
        raise ValueError('Unknown model class')

    # build the optimizer
    optimizer = adamw(learning_rate=params.learning_rate)
    
    # build the supervised learning algorithm
    trainer = supervised_backprop(
        model, optimizer, classify.loss, classify.accuracy)
    
    '''
    @static_functions
    class MaxnistSystem:
        def init(key):
            trainer_key, dataset_key = jrng.split(key)
            trainer_state = trainer.init(trainer_key)
        
        def forward(key, state):
            state, losses = trainer.train(key, train_x, train_y, state)
            #accuracy = trainer.test(
            #    key, test_x, test_y, trainer_state)
            return state, losses
        
        def make_report(state, losses):
            accuracy = trainer.test(
                key, test_x, test_y, MASK, state)
            return losses, accuracy
        
        def log(key, state, reports, epoch):
            losses, accuracy = reports
            print(f'Epoch {epoch}')
            if params.visualize_examples:
                key, model_key = jrng.split(key)
                logits = model(
                    model_key,
                    test_x[:params.visualize_examples],
                    state.model_state,
                )
                pred = jnp.argmax(logits, axis=-1)
            print(f'  accuracy: {accuracy}')
    
    system = MaxnistSystem()
    '''
    #simulate(key, params.simulate_params, system)
    
    '''
    # iterate through each epoch
    for epoch in range(params.epochs):
        print(f'Epoch: {epoch}')
        
        # train
        key, train_key = jrng.split(key)
        trainer_state, losses = trainer.train(
            train_key, train_x, train_y, trainer_state)
        
        # visualize
        if params.visualize_examples:
            key, model_key = jrng.split(key)
            logits = model(
                model_key,
                test_x[:params.visualize_examples],
                trainer_state.model_state,
            )
            pred = jnp.argmax(logits, axis=-1)
        
        # test
        key, test_key = jrng.split(key)
        accuracy = trainer.test(test_key, test_x, test_y, trainer.model_state)
        print(f'  accuracy: {accuracy}')
    '''

if __name__ == '__main__':
    params = MaxnistParams().from_commandline()
    main(params)
