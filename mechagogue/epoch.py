import time
from typing import Any

import jax
import jax.random as jrng
import jax.numpy as jnp

from mechagogue.standardize import standardize_args
from mechagogue.static import static_data, static_functions 
from mechagogue.serial import save_leaf_data, load_example_data
from mechagogue.ds.system import standardize_system, make_system

@static_data
class EpochState:
    system_state : Any
    epoch : int

def make_epoch_system(
    system,
    steps_per_epoch,
    make_report = None,
    log = lambda state, reports : None,
    output_directory = '.',
    save_states = False,
    save_reports = False,
    compress_saved_data = False,
    verbose = True,
):
    
    log = standardize_args(log, ('key', 'state', 'reports'))
    
    #compiled_system = compile_system(standardize_system(system))
    
    @static_functions
    class EpochSystem:
        
        def state_path(epoch):
            return f'{output_directory}/state_{epoch:08}.data'
        
        def epoch_key_path(epoch):
            return f'{output_directory}/epoch_key_{epoch:08}.data'
        
        def system_state_path(epoch):
            return f'{output_directory}/system_state_{epoch:08}.data'
        
        def reports_path(epoch):
            return f'{output_directory}/reports_{epoch:08}.data'
        
        def _init(key):
            if system.init_has_aux:
                system_state, *aux = system.init(key)
            else:
                system_state = system.init(key)
            
            return EpochState(system_state, 1)
        
        if verbose:
            print('compiling init')
            t_pre_compile = time.time()
        example_key = jrng.key(0)
        abstract_key = jax.ShapeDtypeStruct(
            example_key.shape, example_key.dtype)
        init = jax.jit(_init).lower(abstract_key).compile()
        if verbose:
            t_post_compile = time.time()
            print(f'  took {t_post_compile-t_pre_compile:.04}s')
        
        def _multi_step_report(key, state):
            
            def step_report(key_state, _):
                key, state = key_state
                key, step_key = jrng.split(key)
                state = system.step(step_key, state)
                if system.step_has_aux:
                    state, *aux = state
                else:
                    aux = ()
                if make_report is not None:
                    report = make_report(state, *aux)
                else:
                    report = None
                return (key, state), report
            
            (key, state), reports = jax.lax.scan(
                step_report,
                (key, state),
                None,
                length=steps_per_epoch,
            )
            
            if hasattr(system, 'correct'):
                state = system.correct(state, steps_per_epoch)
            
            return state, reports
        
        if verbose:
            print('compiling epoch step')
            t_pre_compile = time.time()
        abstract_state = jax.eval_shape(system.init, abstract_key)
        if system.init_has_aux:
            abstract_state = abstract_state[0]
        multi_step_report = jax.jit(_multi_step_report).lower(
            abstract_key, abstract_state).compile()
        if verbose:
            t_post_compile = time.time()
            print(f'  took {t_post_compile-t_pre_compile:.04}s')
        
        def step(key, state):
            # get the current epoch
            epoch = state.epoch
            if verbose:
                print(f'epoch: {epoch}')
                t_start = time.time()
            
            # split keys
            key, step_key = jrng.split(key)
            
            # run multiple steps and gather reports
            system_state, reports = EpochSystem.multi_step_report(
                step_key, state.system_state)
            
            # run the log function
            key, log_key = jrng.split(key)
            log(log_key, system_state, reports)
            
            # update the epoch for the next time step
            state = EpochState(system_state, state.epoch+1)
            
            # gather timing information
            if verbose:
                state = jax.block_until_ready(state)
                t_primary = time.time()
            
            # save states
            if save_states:
                state = jax.block_until_ready(state)
                
                system_state_path = EpochSystem.system_state_path(epoch)
                epoch_key_path = EpochSystem.epoch_key_path(epoch)
                
                if verbose:
                    print(f'  saving system state to: {system_state_path}')
                if hasattr(system, 'save_state'):
                    system.save_state(
                        state.system_state,
                        system_state_path,
                        compress=compress_saved_data,
                    )
                else:
                    save_leaf_data(
                        state.system_state,
                        system_state_path,
                        compress=compress_saved_data,
                    )
                
                if verbose:
                    print(f'  saving epoch and key to: {epoch_key_path}')
                save_leaf_data(
                    (state.epoch, key),
                    epoch_key_path,
                    compress=compress_saved_data,
                )
            
            # save reports
            if save_reports:
                reports_path = EpochSystem.reports_path(epoch)
                reports = jax.block_until_ready(reports)
                if verbose:
                    print(f'  saving reports to: {reports_path}')
                save_leaf_data(
                    reports, reports_path, compress=compress_saved_data)
            
            # print
            if verbose:
                state = jax.block_until_ready(state)
                t_end = time.time()
                primary_elapsed = t_primary -t_start
                print(
                    f'  {steps_per_epoch} steps took {primary_elapsed:.04}s '
                    f'({steps_per_epoch/primary_elapsed:.4f}hz)'
                )
                secondary_elapsed = t_end - t_primary
                print(f'  saving data took {secondary_elapsed:.4f}s')
                total_elapsed = t_end - t_start
                print(f'  total elapsed: {total_elapsed:.4f}')
            
            return state
        
        def load_epoch(epoch):
            #state_path = EpochSystem.state_path(epoch)
            #example = jrng.key(0), EpochSystem.init(jrng.key(0))
            #return load_example_data(example, state_path)
            epoch_key_example = (0, jrng.key(0))
            epoch_key_path = EpochSystem.epoch_key_path(epoch)
            next_epoch, key = load_example_data(
                epoch_key_example, epoch_key_path)
            
            system_state_path = EpochSystem.system_state_path(epoch)
            if hasattr(system, 'load_state'):
                system_state = system.load_state(system_state_path)
            else:
                epoch_system_example = EpochSystem.init(jrng.key(0))
                system_state = load_exmaple_data(
                    epoch_system_example.system_state, system_state_path)
            
            return key, EpochState(system_state, next_epoch)
    
    return EpochSystem
