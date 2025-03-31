import argparse

import jax
import jax.numpy as jnp
import jax.random as jrng

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

# specify params and read them from the commandline
@commandline_interface
@static_dataclass
class MaxnistParams:
    
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
model_class = 'mutable_channels'

if model_class == 'mlp':
    init_model, model = layer_sequence((
        (lambda : None, lambda x : x.reshape(-1,in_channels)), # flatten
        mlp(
            hidden_layers=params.hidden_layers,
            in_channels=in_channels,
            hidden_channels=params.hidden_channels,
            out_channels=num_classes,
            p_dropout=0.1,
        ),
    ))
elif model_class == 'grouped_linear':
    def group_block(in_channels, hidden_channels, out_channels, groups):
        '''
        A single linear would be  
        (in, out)
        This is
        (groups, in//groups, hidden//groups) +                                          (groups, hidden//groups, out//groups)
        Comparing
        in*out <?> groups*(in//groups)*(hidden//groups) + groups*(hidden//groups)*(out//groups)
        in*out <?> in * (hidden//groups) + (hidden//groups) * out
        in*out <?> (hidden//groups)(in+out)       
        '''
        permutation = jnp.arange(hidden_channels).reshape(
            hidden_channels//groups, groups).T.reshape(-1) 
        return layer_sequence((                                                 
            grouped_linear_layer(in_channels, hidden_channels, groups),
            permute_layer(permutation),                                    
            grouped_linear_layer(hidden_channels, out_channels, groups),
        ))
    
    hidden_channels = 256 
    groups = 8
    permutation = jnp.arange(hidden_channels).reshape(groups,hidden_channels//groups).T.reshape(-1)
    init_model, model = layer_sequence((
        (lambda : None, lambda x :                                            
            jnp.pad(x.reshape(-1,in_channels), ((0,0),(0,15)), mode='empty')),
        group_block(64, 512, hidden_channels, groups),
        #dropout_layer(0.1),
        relu_layer(),
        #permute_layer(permutation),
        group_block(hidden_channels, 512, 32, groups),
        (lambda : None, lambda x : x[...,:10]),
    ))

elif model_class == 'another':
    hidden_channels = 1024
    groups = 16
    permutation = jnp.arange(hidden_channels).reshape(groups, hidden_channels//groups).T.reshape(-1)
    init_model, model = layer_sequence((
        (lambda : None, lambda x : x.reshape(-1, in_channels)),
        linear_layer(in_channels, hidden_channels),
        relu_layer(),
        grouped_linear_layer(hidden_channels, hidden_channels, groups),
        #permute_layer(permutation),
        relu_layer(),
        #grouped_linear_layer(hidden_channels, hidden_channels, groups),
        #relu_layer(),
        linear_layer(hidden_channels, num_classes),
    ))
        
elif model_class == 'mutable_channels':
    from dirt.models.mutable_model import (
        backbone, mutate_backbone, virtual_parameters)
    hidden_channels = 256
    init_model, model = layer_sequence((
        (lambda: None, lambda x : x.reshape(-1, in_channels)),
        linear_layer(in_channels, hidden_channels, dtype=jnp.bfloat16),
        backbone(32, hidden_channels, 1, 1),
        linear_layer(hidden_channels, num_classes),
    ))
    
    backbone_mutator = mutate_backbone(
        32,
        256,
        params.learning_rate,
        params.learning_rate,
        0.1,
    )
    
    mutate_encoder_decoder = normal_mutate(
        params.learning_rate, auto_scale=True)
    def mutate(key, state):
        encoder_decoder_key, backbone_key = jrng.split(key)
        _, encoder_state, backbone_state, decoder_state = state
        encoder_state, decoder_state = mutate_encoder_decoder(
            encoder_decoder_key, (encoder_state, decoder_state))
        backbone_state = backbone_mutator(backbone_key, backbone_state)
        return [None, encoder_state, backbone_state, decoder_state]
    
    breed = mutate

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

def weight_mean_std(weight):
    n = weight.shape[0]
    weight = weight.reshape(n, -1)
    weight_mean = weight.mean(axis=-1)
    weight_std = weight.std(axis=-1)
    return weight_mean, weight_std

'''
print('Initialization:')
weight0 = model_state[1][0][0]
n = weight0.shape[0]
weight0 = weight0.reshape(n, -1)
weight0_mean = weight0.mean(axis=-1)
weight0_std = weight0.std(axis=-1)
print(f'  weight0 mean min/max/mean: {weight0_mean.min():.04}/{weight0_mean.max():.04}/{weight0_mean.mean():.04}')
print(f'  weight0 std min/max/mean: {weight0_std.min():.04}/{weight0_std.max():.04}/{weight0_std.mean():.04}')

weight1 = model_state[1][3][0]
weight1 = weight1.reshape(n, -1)
weight1_mean = weight1.mean(axis=-1)
weight1_std = weight1.std(axis=-1)
print(f'  weight1 mean min/max/mean: {weight1_mean.min():.04}/{weight1_mean.max():.04}/{weight1_mean.mean():.04}')
print(f'  weight1 std min/max/mean: {weight1_std.min():.04}/{weight1_std.max():.04}/{weight1_std.mean():.04}')
'''

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
    print(f'  accuracy: {accuracy}')
    
    if model_class == 'mutable_channels':
        dynamic_channel_state = model_state[2][1]
        print(f'  channels: {dynamic_channel_state[...,0]}')
    
    if model_class == 'mutable_channels':
        _, encoder_state, backbone_state, decoder_state = model_state
        encoder_mean, encoder_std = weight_mean_std(encoder_state[0])
        print(f'  encoder std: {encoder_std.mean()}')
        backbone_linear_state, _ = backbone_state
        for i, layer_state in enumerate(backbone_linear_state):
            layer_mean, layer_std = weight_mean_std(layer_state[0])
            print(f'  weight {i} std: {layer_std.mean()}')
        decoder_mean, decoder_std = weight_mean_std(decoder_state[0])
        print(f'  decoder std: {decoder_std.mean()}')
    
    '''
    weight0 = model_state[1][0][0]
    n = weight0.shape[0]
    weight0 = weight0.reshape(n, -1)
    weight0_mean = weight0.mean(axis=-1)
    weight0_std = weight0.std(axis=-1)
    print(f'  weight0 mean min/max/mean: {weight0_mean.min():.04}/{weight0_mean.max():.04}/{weight0_mean.mean():.04}')
    print(f'  weight0 std min/max/mean: {weight0_std.min():.04}/{weight0_std.max():.04}/{weight0_std.mean():.04}')

    weight1 = model_state[1][3][0]
    weight1 = weight1.reshape(n, -1)
    weight1_mean = weight1.mean(axis=-1)
    weight1_std = weight1.std(axis=-1)
    print(f'  weight1 mean min/max/mean: {weight1_mean.min():.04}/{weight1_mean.max():.04}/{weight1_mean.mean():.04}')
    print(f'  weight1 std min/max/mean: {weight1_std.min():.04}/{weight1_std.max():.04}/{weight1_std.mean():.04}')
    '''
