import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.standardize import standardize_args
from mechagogue.static import static_data, static_functions
from mechagogue.nn.layer import standardize_layer

def supervised_backprop(model, optimizer, loss_function):
    model = standardize_layer(model)
    optimizer = standardize_optimizer(model)
    loss_function = standardize_args(
        loss_function, ('key', 'pred', 'y', 'state', 'mask'))
    
    @static_functions
    class SupervisedBackprop:
        def init(key):
            model_key, optimizer_key = jrng.split(key)
            model_state = model.init(model_key)
            optimizer_state = optimizer.init(optimizer_key)
            return (model_state, optimizer_state)
        
        def train(key, x, y, mask, state):
            model_state, optimizer_state = state
            def forward(key, x, y, mask, state.model_state):
                model_key, loss_key = jrng.split(key)
                pred = model.forward(model_key, x, state.model_state)
                loss = loss_function(loss_key, pred, y, state.model_state, mask)
                return loss
            
            forward_key, optim_key = jrng.split(key)
            loss, grad = jax.value_and_grad(forward, argnums=4)(
                forward_key, x, y, mask, state.model_state)
            model_state, optim_state = optimizer.optimize(
                optim_key, grad, state.model_state, state.optim_state)
            return (model_state, optim_state), (loss, grad)
        
        TEST = WTF
    
    return SupervisedBackprop

def supervised_backprop(
    model,
    optimizer,
    dataset,
    loss_function,
    track_grad=False,
):
    @static_functions
    class SupervisedBackpropSystem:
        
        @static_data
        class SupervisedBackpropState:
            model_state : Any
            optimizer_state : Any
            dataset_state : Any
            loss : Any
            grad : Any
        
        def init(key):
            model_key, optimizer_key, dataset_key = jrng.split(key)
            model_state = model.init(model_key)
            optimizer_state = optimizer.init(optimizer_key)
            dataset_state = optimizer.init(dataset_key)
            
            if track_grad:
                state = SupervisedBackpropState(
                    model_state, optimizer_state, dataset_state, loss, grad)
            else:
                state = SupervisedBackpropState(
                    model_state, optimizer_state, dataset_state, loss)
            
            return state
        
        def step(key, state):
            model_state, optimizer_state, dataset_state = state
            
            dataset_state, x, y, mask = dataset.step(dataset_key, dataset_state)
            
            def forward(key, x, y, mask, state.model_state):
                model_key, loss_key = jrng.split(key)
                pred = model.forward(model_key, x, state.model_state)
                loss = loss_function(loss_key, pred, y, state.model_state, mask)
                return loss

            forward_key, optim_key = jrng.split(key)
            loss, grad = jax.value_and_grad(forward, argnums=4)(
                forward_key, x, y, mask, state.model_state)
            model_state, optim_state = optimizer.optimize(
                optim_key, grad, state.model_state, state.optim_state)
            
            state = SupervisedBackpropState(
                model_state,
                optimizer_state,
                dataset_state,
                loss,
                grad,
            )
            
            return state
    
    key, state = simulate_epochs(
        key, params, system, pre_epoch, post_epoch, state=state)
