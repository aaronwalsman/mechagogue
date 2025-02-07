import jax
import jax.random as jrng
import jax.numpy as jnp

from mechagogue.static_dataclass import static_dataclass
from mechagogue.arg_wrappers import ignore_unused_args, split_random_keys
from mechagogue.wrappers import (
    episode_return_wrapper,
    auto_reset_wrapper,
    parallel_env_wrapper,
)
from mechagogue.tree import tree_getitem, tree_setitem, ravel_tree

@static_dataclass
class DQNConfig:
    parallel_envs: int = 32
    num_q_models: int = 2
    replay_buffer_size: int = 5000*32
    rollout_steps: int = 1
    batches_per_step: int = 1
    # TODO:
    #initial_random_data: int = 1000*32,
    
    discount: float = 0.95
    epsilon: float = 0.1
    target_update: float = 0.01
    
    batch_size: int = 32

def dqn(
    config,
    reset_env,
    step_env,
    init_model_params,
    model,
    init_optimizer_params,
    optimize,
    random_action,
):
    
    assert (
        config.replay_buffer_size %
        (config.parallel_envs * config.rollout_steps)
    ) == 0
    
    # auto-reset and parallelize the environment
    reset_env, step_env = auto_reset_wrapper(reset_env, step_env)
    reset_env, step_env = parallel_env_wrapper(
        reset_env, step_env, config.parallel_envs)
    
    # parallelize the model over the number of q functions
    init_model_params = ignore_unused_args(init_model_params, ('key',))
    init_model_params = jax.vmap(init_model_params)
    init_model_params = split_random_keys(
        init_model_params, config.num_q_models)
    single_model = ignore_unused_args(model, ('key', 'x', 'params'))
    model = jax.vmap(single_model, in_axes=(0,None,0))
    model = split_random_keys(model, config.num_q_models)
    
    # wrap the optimizer
    init_optimizer_params = ignore_unused_args(
        init_optimizer_params, ('key', 'model_params'))
    optimize = ignore_unused_args(
        optimize, ('key', 'grad', 'model_params', 'params'))
    
    # parallelize the action sampler
    random_action = ignore_unused_args(random_action, ('key',))
    random_action = jax.vmap(random_action)
    random_action = split_random_keys(random_action, config.parallel_envs)
    
    def init(key):
        reset_key, model_key, optimizer_key, replay_key = jrng.split(key, 4)
        
        state, obs = reset_env(reset_key)
        done = jnp.zeros(config.parallel_envs, dtype=jnp.bool)
        
        # build the model, target and optimizer params
        model_params = init_model_params(model_key)
        target_params = model_params
        optimizer_params = init_optimizer_params(optimizer_key, model_params)
        
        # fill the replay buffer with data from the random policy initially
        def fill_replay_buffer(state_obs_done, key):
            state, obs, done = state_obs_done
            action_key, step_key = jrng.split(key)
            
            # sample a random action
            action = random_action(action_key)
            
            # take an environment step
            next_state, next_obs, reward, next_done = step_env(
                step_key, state, action)
            
            #jax.debug.print('s: {s}\na: {a}\nns: {ns}\nr: {r}\nd: {d}\nnd: {nd}\n\n', s=state[0], a=action, ns=next_state[0], r=reward, d=done, nd=next_done)
            
            return (
                (next_state, next_obs, next_done),
                (obs, action, reward, next_obs, done, next_done),
            )
        
        fill_steps = (
            config.replay_buffer_size // config.parallel_envs)
        (state, obs, done), replay_buffer = jax.lax.scan(
            fill_replay_buffer,
            (state, obs, done),
            jrng.split(replay_key, fill_steps),
        )
        
        replay_buffer = ravel_tree(replay_buffer, 0, 2)
        
        ##################
        #o,a,r,n,d,ns = replay_buffer
        #jax.debug.print('rrr: {r}', r=r.sum())
        ##################
        
        return (
            state,
            obs,
            done,
            model_params,
            target_params,
            optimizer_params,
            replay_buffer,
            0,
        )
    
    def step(
        key,
        state,
        obs,
        done,
        model_params,
        target_params,
        optimizer_params,
        replay_buffer,
        replay_write_location,
    ):
        single_model_params = tree_getitem(model_params, 0)
        
        # generate new data
        def rollout(state_obs_done, key):
            state, obs, done = state_obs_done
            model_key, random_key, selector_key, step_key = jrng.split(key, 4)
            
            # sample an action according to the epsilon-greedy policy
            q = single_model(model_key, obs, single_model_params)
            greedy = jnp.argmax(q, axis=-1)
            random = random_action(random_key)
            selector = jrng.bernoulli(
                selector_key, shape=greedy.shape, p=config.epsilon)
            action = jnp.where(selector, random, greedy)
            
            # take an environment step
            next_state, next_obs, reward, next_done = step_env(
                step_key, state, action)
            
            return (
                (next_state, next_obs, next_done),
                (obs, action, reward, next_obs, done, next_done),
            )
        
        key, rollout_key = jrng.split(key)
        (state, obs, done), trajectories = jax.lax.scan(
            rollout,
            (state, obs, done),
            jrng.split(rollout_key, config.rollout_steps),
        )
        
        trajectories = ravel_tree(trajectories, 0, 2)
        replay_offset = config.rollout_steps * config.parallel_envs
        indices = (
            jnp.arange(config.rollout_steps * config.parallel_envs) +
            replay_write_location
        )
        replay_buffer = tree_setitem(replay_buffer, indices, trajectories)
        
        # train
        def train_batch(params, key):
            model_params, target_params, optimizer_params = params
            
            # sample data from the replay buffer
            data_key, target_key, model_key, optimizer_key = jrng.split(key, 4)
            replay_buffer_indices = jrng.choice(
                data_key,
                config.replay_buffer_size,
                shape=(config.batch_size,),
            )
            obs, action, reward, next_obs, done, next_done = tree_getitem(
                replay_buffer, replay_buffer_indices)
            
            # compute the q targets
            next_q = model(target_key, next_obs, target_params)
            next_q = next_q.min(axis=0)
            next_v = next_q.max(axis=-1)
            target = reward + (~next_done) * config.discount * next_v
            
            #jax.debug.print('target: {t}\nreward: {r}\nnot next_done: {nd}\n? {nddv}\nobs: {o}\nact: {a}\ndone: {d}\nuh: {uh}\n\n', t=target, r=reward, nd=(~next_done), nddv=(~next_done)*config.discount*next_v, o=obs, a=action, d=done, uh=done==~next_done)
            
            #jax.debug.print('o: {o}\na: {a}\nno: {no}\nr: {r}\nd: {d}\nnd: {nd}\nt: {t}\n\n',
            #    o=obs, a=action, no=next_obs, r=reward, d=done, nd=next_done, t=target)
            
            # compute the loss and gradients
            def q_loss(model_key, model_params, obs, action, done, target):
                q = model(model_key, obs, model_params)
                q = q[:,jnp.arange(config.batch_size), action]
                loss = (~done) * (q - target)**2
                return loss.mean()
            
            loss, grad = jax.value_and_grad(q_loss, argnums=1)(
                model_key, model_params, obs, action, done, target)
            
            #jax.debug.print('pre: {mp}', mp=model_params)
            
            # apply the gradients
            model_params, optimizer_params = optimize(
                optimizer_key, grad, model_params, optimizer_params)
            
            #jax.debug.print('post: {mp}', mp=model_params)
            
            # update the target network
            def interpolate_leaf(target_leaf, model_leaf):
                return (
                    target_leaf * (1. - config.target_update) +
                    model_leaf * config.target_update
                )
            target_params = jax.tree.map(
                interpolate_leaf, target_params, model_params)
            
            return (model_params, target_params, optimizer_params), loss
        
        key, batch_key = jrng.split(key)
        params, batch_losses = jax.lax.scan(
            train_batch,
            (model_params, target_params, optimizer_params),
            jrng.split(batch_key, config.batches_per_step),
        )
        model_params, target_params, optimizer_params = params
        
        return (
            state,
            obs,
            done,
            model_params,
            target_params,
            optimizer_params,
            replay_buffer,
            replay_write_location,
        ), batch_losses
    
    return init, step
