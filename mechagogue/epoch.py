import time
from typing import Any

import jax
import jax.random as jrng

from mechagogue.standardize import standardize_args
from mechagogue.static import static_data, static_functions 
from mechagogue.serial import save_leaf_data, load_example_data
from mechagogue.ds.system import (
    standardize_system, make_system, jit_system, iterated_system_scan)

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
    
    log = standardize_args(log, ('state', 'reports'))
    
    system = standardize_system(system)
    
    '''
    if make_report is not None:
        def step_report(key, state):
            key, step_key = jrng.split(key)
            state = system.step(step_key, state)
            if system.step_has_aux:
                state, *aux = state
            else:
                aux = ()
            report = make_report(state, *aux)
            return state, report
        
        system = make_system(
            init=system.init,
            step=step_report,
            init_has_aux = system.init_has_aux,
            step_has_aux = True,
        )
    system = iterated_system_scan(
        system, steps_per_epoch, collect_aux=(make_report is not None))
    '''
    
    system = jit_system(system)
    
    @static_functions
    class EpochSystem:
        
        def state_path(epoch):
            return f'{output_directory}/state_{epoch:08}.data'
        
        def reports_path(epoch):
            return f'{output_directory}/reports_{epoch:08}.data'
        
        def init(key):
            if system.init_has_aux:
                system_state, *aux = system.init(key)
            else:
                system_state = system.init(key)
            
            return EpochState(system_state, 1)
        
        @jax.jit
        def multi_step_report(key, state):
            
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
            
            return state, reports
        
        def step(key, state):
            epoch = state.epoch
            if verbose:
                print(f'epoch: {epoch}')
                t_start = time.time()
            
            key, step_key = jrng.split(key)
            
            system_state, reports = EpochSystem.multi_step_report(
                step_key, state.system_state)
            
            log(system_state, reports)
            state = EpochState(system_state, state.epoch+1)
            
            if verbose:
                state = jax.block_until_ready(state)
                t_primary = time.time()
            
            if save_states:
                state_path = EpochSystem.state_path(epoch)
                state = jax.block_until_ready(state)
                if verbose:
                    print(f'  saving state to: {state_path}')
                save_leaf_data(
                    (key, state), state_path, compress=compress_saved_data)
            
            if save_reports:
                reports_path = EpochSystem.reports_path(epoch)
                reports = jax.block_until_ready(reports)
                if verbose:
                    print(f'  saving reports to: {reports_path}')
                save_leaf_data(
                    reports, reports_path, compress=compress_saved_data)
            
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
            state_path = EpochSystem.state_path(epoch)
            example = jrng.key(0), EpochSystem.init(jrng.key(0))
            return load_example_data(example, state_path)
    
    return EpochSystem
