import argparse

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.commandline import commandline_interface
from mechagogue.static_dataclass import static_dataclass
from mechagogue.sup.sup import SupParams, sup
import mechagogue.sup.tasks.classify as classify
from mechagogue.nn.linear import linear_layer, grouped_linear_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.mlp import mlp
from mechagogue.nn.nonlinear import relu_layer
from mechagogue.nn.permute import permute_layer
from mechagogue.nn.debug import breakpoint_layer
from mechagogue.optim.adamw import adamw
import mechagogue.data.maxnist as maxnist

# specify params and read them from the commandline
@commandline_interface
@static_dataclass
class MaxnistParams:
    
    # rng
    seed : int = 1234
    
    # data   
    digits : int = 1
    train_examples : int = 5000
    test_examples : int = 1000
    data_noise : float = 0.025
    
    # model
    hidden_layers : int = 1
    hidden_channels : int = 256
    
    # optim
    learning_rate : float = 3e-3
    
    # training
    epochs : int = 40
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

# build the model
in_channels = 49*params.digits
num_classes = 10**params.digits
model_class = 'grouped_linear'

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
        (groups, in//groups, hidden//groups) +
        (groups, hidden//groups, out//groups)
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
    groups = 2
    permutation = jnp.arange(hidden_channels).reshape(groups,hidden_channels//groups).T.reshape(-1)
    init_model, model = layer_sequence((
        #(lambda : None, lambda x : jnp.concatenate(
        #    (x.reshape(-1,in_channels), jnp.zeros((b,15))))),
        (lambda : None, lambda x :
            jnp.pad(x.reshape(-1,in_channels), ((0,0),(0,15)), mode='empty')),
        #grouped_linear_layer(64, hidden_channels, groups),
        group_block(64, 512, hidden_channels, groups),
        relu_layer(),
        #permute_layer(permutation),
        group_block(hidden_channels, 512, 32, groups),
        #grouped_linear_layer(hidden_channels, 32, groups),
        (lambda : None, lambda x : x[...,:10]),
    ))

# build the optimizer
init_optim, optim = adamw(learning_rate=params.learning_rate)

# build the supervised learning algorithm
sup_params = SupParams(batch_size=params.batch_size)
init_train, train_model, test_model = sup(
    sup_params,
    init_model,
    model,
    init_optim,
    optim,
    classify.loss,
    classify.accuracy,
)
train_model = jax.jit(train_model)
test_model = jax.jit(test_model)

# initialize the supervised learning algorithm
key, init_key = jrng.split(key)
model_state, optim_state = init_train(init_key)

'''
print('initialization')
weight0 = model_state[1][0][0]
weight0_mean = weight0.reshape(-1).mean()
weight0_std = weight0.reshape(-1).std()

weight1 = model_state[1][3][0]
weight1_mean = weight1.reshape(-1).mean()
weight1_std = weight1.reshape(-1).std()

print(f'  weight 0 mean/std {weight0_mean}/{weight0_std}')
print(f'  weight 1 mean/std {weight1_mean}/{weight1_std}')
'''

# iterate through each epoch
for epoch in range(params.epochs):
    print(f'Epoch: {epoch}')
    
    # train
    key, train_key = jrng.split(key)
    model_state, optim_state, losses = train_model(
        train_key, train_x, train_y, model_state, optim_state)
    
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
    
    '''
    weight0 = model_state[1][0][0]
    weight0_mean = weight0.reshape(-1).mean()
    weight0_std = weight0.reshape(-1).std()
    
    weight1 = model_state[1][3][0]
    weight1_mean = weight1.reshape(-1).mean()
    weight1_std = weight1.reshape(-1).std()
    
    print(f'  weight 0 mean/std {weight0_mean}/{weight0_std}')
    print(f'  weight 1 mean/std {weight1_mean}/{weight1_std}')
    '''
