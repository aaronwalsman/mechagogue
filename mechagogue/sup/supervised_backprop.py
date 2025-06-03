import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.standardize import standardize_args
from mechagogue.static import static_data, static_functions
#from mechagogue.tree import (
#    tree_len, shuffle_tree, pad_tree_batch_size, batch_tree)
#from mechagogue.eval.batch_eval import batch_evaluator
from mechagogue.nn.layer import standardize_layer

#@static_dataclass
#class SupervisedBackpropParams:
#    batch_size: int = 64
#    shuffle: bool = True

def supervised_backprop(
    #params,
    model,
    optimizer,
    #data_loader,
    loss_function,
    test_function,
):
    model = standardize_layer(model)
    optimizer = standardize_optimizer(model)
    loss_function = standardize_args(
        loss_function, ('key', 'pred', 'y', 'state', 'mask'))
    
    @static_functions
    class SupervisedBackprop:
        #@static_data
        #class SupervizedBackpropState:
        #    model_state : Any
        #    optimizer_state : Any
        #    data_loader_state : Any
        
        #@static_data
        #class SupervisedBackpropAux:
        #    x : Any
        #    y : Any
        #    mask : Any
        #    loss : Any
        #    grad : Any
        
        #def init(key):
        #    model_key, optim_key, data_loader_key = jrng.split(key, 3)
        #    model_state = model.init(model_key)
        #    optim_state = optimizer.init(optim_key, model_state)
        #    data_loader_state = data_loader.init(data_loader_key)
        #    _, x, _, _ = data_loader.step(data_loader_key, data_loader_state)
        #    b = x.shape[0]
        #    loss = jnp.zeros(b)
        #    return SupervisedBackpropState(
        #        model_state, optimizer_state, data_loader_state)
        
        def train(key, x, y, mask, model_state, optimizer_state):
            def forward(key, x, y, mask, state.model_state):
                model_key, loss_key = jrng.split(key)
                pred = model.forward(model_key, x, state.model_state)
                loss = loss_function(loss_key, pred, y, state.model_state, mask)
                return loss
            
            forward_key, optim_key = jrng.split(key)
            #data_loader_state, x, y, mask = data_loader.forward(
            #    data_loader_key, state.data_loader_state)
            loss, grad = jax.value_and_grad(forward, argnums=4)(
                forward_key, x, y, mask, state.model_state)
            model_state, optim_state = optimizer.optimize(
                optim_key, grad, state.model_state, state.optim_state)
            #state = SupervisedBackpropState(
            #    model_state,
            #    optimizer_state,
            #    data_loader_state,
            #)
            #aux = SupervizedBackpropAux(x, y, mask, loss, grad)
            #return state, aux
            return model_state, optim_state
        
        #def forward(key, x, y, mask, state):
        #    #if params.shuffle:
        #    #    key, shuffle_key = jrng.split(key)
        #    #    x, y = shuffle_tree(shuffle_key, (x, y))
        #    #(x, y), mask = pad_tree_batch_size((x, y), params.batch_size)
        #    #x, y, mask = batch_tree((x, y, mask), params.batch_size)
        #    
        #    def train_batch(state, key_x_y_mask):
        #        key, x, y, mask = key_x_y_mask
        #        
        #        def forward(key, x, y, mask, state.model_state):
        #            pred = model.forward(key, x, state.model_state)
        #            loss = loss_function(pred, y, state.model_state, mask)
        #            return loss
        #        
        #        forward_key, optim_key = jrng.split(key)
        #        loss, grad = jax.value_and_grad(forward, argnums=4)(
        #            forward_key, x, y, mask, state.model_state)
        #        model_state, optim_state = optimizer.optimize(
        #            optim_key, grad, state.model_state, state.optim_state)
        #        
        #        return SupervisedBackpropState(model_state, optim_state), loss
        #    
        #    num_batches = tree_len(x)
        #    batch_keys = jrng.split(key, num_batches)
        #    state, losses = jax.lax.scan(
        #        train_batch, state, (batch_keys, x, y, mask))
        #    return SupervisedBackpropState(model_state, optimizer_state), losses
        
        #test = batch_evaluator(model, test_function, params.batch_size)
    
    return SupervisedBackprop
