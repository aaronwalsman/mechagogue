import argparse

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

import wandb

from mechagogue.commandline import commandline_interface
from mechagogue.static_dataclass import static_dataclass
from mechagogue.sup.ga import GAParams, ga
import mechagogue.sup.tasks.classify as classify
from mechagogue.nn.linear import linear_layer, grouped_linear_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.mlp import mlp
from mechagogue.nn.permute import permute_layer
from mechagogue.nn.nonlinear import relu_layer
from mechagogue.nn.regularizer import dropout_layer
import mechagogue.data.maxnist as maxnist
from mechagogue.breed.normal import normal_mutate
from mechagogue.tree import tree_getitem

# specify params and read them from the commandline
@commandline_interface
@static_dataclass
class MaxnistParams:
    
    # run name
    run_name : str = 'default'
    
    # rng
    seed : int = 1234
    
    # data   
    digits : int = 1
    train_examples : int = 50000
    test_examples : int = 10000
    data_noise : float = 0.025
    
    # model
    hidden_layers : int = 1
    hidden_channels : int = 512
    
    # optim
    learning_rate : float = 1e-3
    
    # training
    epochs : int = 2000
    batch_size : int = 64
    visualize_examples : int = 0

params = MaxnistParams().from_commandline()

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

# build the mutator
breed = normal_mutate(learning_rate=params.learning_rate)

# build the model
in_channels = 49*params.digits
num_classes = 10**params.digits
model_class = 'mutable'

if model_class == 'mlp':
    init_model, model = layer_sequence((
        (lambda : None, lambda x : x.reshape(-1, in_channels)), # flatten
        mlp(
            hidden_layers=params.hidden_layers,
            in_channels=in_channels,
            hidden_channels=params.hidden_channels,
            out_channels=num_classes,
            p_dropout=0.1,
        ),
    ))
    
    def get_labelled_weights(model_state):
        breakpoint()
        return {}

elif model_class == 'mutable':
    from dirt.models.mutable_model import (
        mutable_mlp, mutate_mutable_mlp)
    hidden_channels = 256
    
    init_model, model = layer_sequence((
        (lambda: None, lambda x : x.reshape(-1, in_channels)),
        mutable_mlp(
            in_channels,
            num_classes,
            32,
            hidden_channels,
            initial_hidden_layers=1,
            max_hidden_layers=4,
        )
    ))
    
    mutate_mlp = mutate_mutable_mlp(
        32,
        256,
        params.learning_rate,
        params.learning_rate,
        0.05,
        0.01,
    )
    
    def mutate(key, state):
        _, mlp_state = state
        mlp_state = tree_getitem(mlp_state, 0)
        mlp_state = mutate_mlp(key, mlp_state)
        return [None, mlp_state]
    
    breed = mutate
    
    '''
    def get_labelled_weights(model_state):
        labelled_weights = {
            'encoder' : model_state[1][0],
            'decoder' : model_state[3][0],
        }
        labelled_weights.update({
            f'weight_{i}' : layer_state[0]
            for i, layer_state in enumerate(model_state[2][0])
        })
        return labelled_weights
    
    def get_labelled_weight_std_target(model_state):
        weight_info = backbone_weight_info(
            model_state[2], shared_dynamic_channels)
        breakpoint()
    '''

# build the supervised learning algorithm
ga_params = GAParams(
    elites=10,
    batch_size=params.batch_size,
    batches_per_step=16,
    population_size=128,
    share_keys=True,
)
init_train, train_model, test_model = ga(
    ga_params,
    init_model,
    model,
    breed,
    classify.loss,
    classify.accuracy,
)
train_model = jax.jit(train_model)
test_model = jax.jit(test_model)

# initialize the supervised learning algorithm
key, init_key = jrng.split(key)
model_state = init_train(init_key)

wandb.init(
    project='maxnist_ga',
    name=params.run_name,
    entity='harvardml',
)

#def weight_mean_std(weight):
#    n = weight.shape[0]
#    weight = weight.reshape(n, -1)
#    #weight_mean = weight.mean(axis=-1)
#    #weight_std = weight.std(axis=-1)
#    return weight_mean, weight_std
#
#def dynamic_weight_mean_std(weight, in_channels, out_channels):
#    n,i,o = weight.shape
#    weight_sum = jnp.sum(weight.reshape(n, -1), axis=1)
#    weight_mean = weight_sum / (in_channels * out_channels)
#    var = (weight_mean.reshape(n, None, None) - weight)**2
#    breakpoint()
#    #var = jnp.where(jnp.arange(i)[None,:] < in_channels, 

def log(model_state, accuracy):
    datapoint = {
        'accuracy' : accuracy.mean()
    }
    
    '''
    labelled_weights = get_labelled_weights(model_state)
    datapoint.update({
        f'std/{weight_name}' : dynamic_weight_mean_std(weight, )[1].mean()
        for weight_name, weight in labelled_weights.items()
    })
    '''
    
    '''
    weight_std_target = get_labelled_weight_std_target(model_state)
    
    if model_class == 'mutable_channels':
        dynamic_channel_state = model_state[2][1]
        datapoint.update({
            'channels':
            dynamic_channel_state[...,0].astype(jnp.float32).mean(),
        })
    '''
    wandb.log(datapoint)

#labelled_weights = get_labelled_weights(model_state)
#for weight_name, weights in labelled_weights.items():
#    weight_mean, weight_std = weight_mean_std(weights)
#    wandb.log({f'std/{weight_name}':weight_std.mean()})
log(model_state, np.zeros((ga_params.population_size,)))

# iterate through each epoch
for epoch in range(params.epochs):
    print(f'Epoch: {epoch}')
    
    # train
    key, train_key = jrng.split(key)
    model_state, fitness = train_model(train_key, train_x, train_y, model_state)
    
    # visualize
    if params.visualize_examples:
        key, model_key = jrng.split(key)
        logits = model(
            model_key, test_x[:params.visualize_examples], model_state)
        pred = jnp.argmax(logits, axis=-1)
    
    # test
    key, test_key = jrng.split(key)
    accuracy = test_model(test_key, test_x, test_y, model_state)
    accuracy = np.array(accuracy).astype(np.float32)
    print(f'  accuracy: {accuracy}')
    log(model_state, accuracy)
