'''
Dynamical system abstractions and combinators.

Provides builders and utilities for creating, composing, and iterating
stateful systems with init/step interfaces.
'''

import jax
import jax.random as jrng

from mechagogue.static import static_functions
from mechagogue.standardize import standardize_interface, standardize_args

default_init = lambda : None
default_step = lambda state : state

def make_system(
    init=default_init,
    step=default_step,
    correct=None,
    init_has_aux=False,
    step_has_aux=False,
):
    
    system_init = init
    system_step = step
    system_correct = correct
    system_init_has_aux = init_has_aux
    system_step_has_aux = step_has_aux
    
    @static_functions
    class System:
        init_has_aux = system_init_has_aux
        step_has_aux = system_step_has_aux
        
        init = system_init
        step = system_step
        if system_correct is not None:
            correct = system_correct
    
    return System

def standardize_system(system):
    next_system = standardize_interface(
        system,
        init = (('key',), default_init),
        step = (('key', 'state'), default_step),
    )
    
    if hasattr(system, 'correct'):
        next_system.correct = staticmethod(
            standardize_args(system.correct, ('state', 'steps')))
    
    next_system.init_has_aux = getattr(system, 'init_has_aux', False)
    next_system.step_has_aux = getattr(system, 'step_has_aux', False)
    
    return next_system

def timed_system(system):
    def step(key, state):
        t0 = time.time()
        state = system.step(key, state)
        state = jax.block_until_ready(state)
        elapsed = time.time() - t0
        print('elapsed: {elapsed}s')
        return state
    
    return make_system(system.init, step)

def jit_system(system):
    return make_system(
        init=jax.jit(system.init),
        step=jax.jit(system.step),
        init_has_aux=system.init_has_aux,
        step_has_aux=system.step_has_aux,
    )

def composite_system(systems):
    def step(key, state):
        for system in systems:
            key, system_key = jrng.split(key)
            state = system.step(system_key, state)
        
        return state
    
    return make_system(systems[0].init, step)

def joint_system(systems):
    @static_functions
    class JointSystem:
        def init(key):
            system_keys = jrng.split(key, len(systems))
            return tuple(
                system.init(system_key)
                for system, system_key in zip(systems, system_keys)
            )
        
        def step(key, state):
            system_keys = jrng.split(key, len(systems))
            return tuple(
                system.step(system_key, system_state)
                for system, system_key, system_state
                in zip(systems, system_keys, state)
            )
    
    return JointSystem

def add_callbacks(system, pre_callbacks=(), post_callbacks=()):
    pre_callbacks = (make_system(step=callback) for callback in pre_callbacks)
    post_callbacks = (make_system(step=callback) for callback in post_callbacks)
    return composite_system((*pre_callbacks, system, *post_callbacks))

def iterated_system_scan(system, steps, collect_aux=False):
    system = standardize_system(system)
    def step(key, state):
        def scan_step(key_state, _):
            key, state = key_state
            key, step_key = jrng.split(key)
            next_state = system.step(step_key, state)
            if system.step_has_aux:
                next_state, *aux = next_state
            
            if not collect_aux or not system.step_has_aux:
                aux = None
            
            return (key, next_state), aux
        
        (key, state), aux = jax.lax.scan(
            scan_step, (key, state), None, length=steps)
        
        if collect_aux:
            return state, *aux
        
        else:
            return state
    
    return make_system(
        init=system.init,
        step=step,
        init_has_aux=system.init_has_aux,
        step_has_aux=(system.step_has_aux and collect_aux)
    )
    
def iterated_system_for(system, steps):
    return composite_system((system,) * steps)
