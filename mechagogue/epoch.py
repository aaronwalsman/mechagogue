from typing import Any

import jax
import jax.random as jrng

from mechagogue.standardize import standardize_args
from mechagogue.static import static_data, static_functions 
from mechagogue.serial import save_leaf_data
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
        
        def init(key):
            if system.init_has_aux:
                system_state, *aux = system.init(key)
            else:
                system_state = system.init(key)
            
            return EpochState(system_state, 0)
        
        def step(key, state):
            epoch = state.epoch
            
            key, step_key = jrng.split(key)
            
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
            
            (key, system_state), reports = jax.lax.scan(
                step_report,
                (key, state.system_state),
                None,
                length=steps_per_epoch,
            )
            
            '''
            system_state = system.step(key, state.system_state)
            if make_report is not None:
                system_state, reports = state
            else:
                reports = None
            '''
            log(system_state, reports)
            state = EpochState(system_state, state.epoch+1)
            
            if save_states:
                state_path = f'{output_directory}/state_{epoch:08}.data'
                state = jax.block_until_ready(state)
                print(f'saving state to: {state_path}')
                save_leaf_data(state, state_path)
            
            if save_reports:
                reports_path = (
                    f'{output_directory}/reports_{epoch:08}.data')
                reports = jax.block_until_ready(reports)
                print(f'saving reports to: {reports_path}')
                save_leaf_data(reports, reports_path)
            
            return state
    
    return EpochSystem
