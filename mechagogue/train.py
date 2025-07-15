import jax.random as jrng

from mechagogue.static import static_data

raise Exception('NO, should be simulate')

@jax.jit
def train(
    key, trainer, steps, state=None, report_frequency=None,
):
    if report_frequency is not None:
        assert steps % report_frequency == 0
        steps_per_block = report_frequency
        num_blocks = steps // report_frequency
    else:
        steps_per_block = steps
        num_blocks = 1
    
    if state is None:
        key, init_key = jrng.split(key)
        state = system.init(key)
    
    def train_block(key_state, _):
        
        def train_step(key_state, _):
            key, state = key_state
            key, step_key = jrng.split(key)
            next_state = trainer.train(step_key, state)
            return (key, next_state), None
        
        key_state, _ = jax.lax.scan(
            train_step,
            (key, state),
            None,
            length=steps_per_block,
        )
        
        if report_frequency is not None:
            report = system.report(state)
        else:
            report = None
        
        return (key, state), report
    
    key_state, reports = jax.lax.scan(
        simulate_step,
        (key, state),
        None,
        length=num_blocks,
    )
    
    key, state = key_state
    if report_frequency is None:
        return key, state
    else:
        return key, state, reports

@static_data
class SimulateEpochsParams:
    epochs : int
    steps_per_epoch : int
    block_frequency : int
    
    report_frequency : int = None
    save_reports : bool = True 
    
    checkpoint_frequency : int = 1
    
    block_padding : int = 4
    epoch_padding : int = 8
    
    #@property
    #def blocks_per_epoch(self):
    #    return self.steps_per_epoch // self.report_frequency

def simulate_epochs(
    key : chex.PRNGKey,
    params : Any,
    system : Any,
    pre_epoch : Any = lambda state : state,
    post_epoch : Any = lambda state : state,
    state : Any = None,
):
    
    system = standardize_system(system)
    
    if state is None:
        key, init_key = jrng.split(key)
        state = system.init(init_key)
    
    while epoch < params.epochs:
        epoch_name = str(epoch).rjust(params.epoch_padding, '0')
        epoch += 1
        
        block = 0
        while block < params.blocks_per_epoch:
            pre_key, block_key, post_key = jrng.split(key)
            
            state = pre_block(pre_key, state)
            _, state, reports = simulate(
                block_key,
                system,
                params.steps_per_epoch,
                state=state,
                report_frequency=params.report_frequency,
            )
            state = post_block(post_key, state, reports)
            
            if params.report_frequency is not None and params.save_reports:
                report_name = str(block).rjust(params.report_padding, '0')
                report_full_name = f'{epoch_name}_{report_name}'
                save_reports(reports, report_full_name)
        
        if epoch % params.checkpoint_frequency == 0:
            save_checkpoint(key, state, epoch)
    
    return key, state
