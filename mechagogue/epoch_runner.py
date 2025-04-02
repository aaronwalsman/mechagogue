from typing import Any, Callable, Optional

import jax
import jax.random as jrng

import chex

from mechagogue.static_dataclass import static_dataclass
from mechagogue.arg_wrappers import ignore_unused_args
from mechagogue.serial import save_leaf_data, load_example_data

@static_dataclass
class EpochRunnerParams:
    epochs : int
    steps_per_epoch : int
    report_frequency : int = 1
    save_state : bool = False
    save_reports : bool = False
    @property
    def blocks_per_epoch(self):
        return self.steps_per_epoch // self.report_frequency

def epoch_runner(
    key : chex.PRNGKey,
    params : Any,
    init : Callable,
    step : Callable,
    make_report : Callable,
    log : Callable,
    output_directory : str = '.',
    load_state : Optional[str] = None,
):
    
    init = ignore_unused_args(init, ('key',))
    step = ignore_unused_args(step, ('key', 'state'))
    
    def save_state(key, state, epoch):
        save_leaf_data(
            (key, state, epoch),
            f'{output_directory}/train_state_{epoch:08}.state',
        )
    
    def save_reports(reports, epoch):
        save_leaf_data(
            reports,
            f'{output_directory}/report_{epoch:08}.state',
        )
    
    key, init_key = jrng.split(key)
    state, *_ = init(init_key)
    epoch = 0
    if load_state:
        key, state, epoch = load_example_data((key, state, epoch), load_state)
        epoch = epoch + 1
    
    # do one step to get the emission shape
    _, *example_side_effects = step(key, state)
    
    def run_epoch(key, state):
        def block_scan(key_state, _):
            key, state = key_state
            def step_scan(key_state_side_effects, _):
                key, state, _ = key_state_side_effects
                key, step_key = jrng.split(key)
                next_state, *side_effects = step(step_key, state)
                return (key, next_state, side_effects), None
            
            key_state_side_effects, _ = jax.lax.scan(
                step_scan,
                (key, state, example_side_effects),
                None,
                length=params.report_frequency,
            )
            key, state, side_effects = key_state_side_effects
            report = make_report(state, *side_effects)
            
            return (key, state), report
        
        key_state, reports = jax.lax.scan(
            block_scan,
            (key, state),
            None,
            length=params.blocks_per_epoch,
        )
        key, state = key_state
        
        return key, state, reports
    
    run_epoch = jax.jit(run_epoch)
    
    while epoch < params.epochs:
        key, state, reports = run_epoch(key, state)
        log(epoch, reports)
        
        if params.save_state and (epoch+1) % int(params.save_state) == 0:
            save_state(key, state, epoch)
        
        if params.save_reports and (epoch+1) % int(params.save_reports) == 0:
            save_reports(reports, epoch)
        
        epoch += 1
