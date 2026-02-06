"""Self-play PPO training loop for turn-based games."""

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.game_solvers.wrappers import vectorize_aec
from mechagogue.optim.optimizer import standardize_optimizer
from mechagogue.static import static_data, static_functions
from mechagogue.standardize import standardize_interface
from mechagogue.tree import ravel_tree, shuffle_tree, batch_tree, tree_len


@static_data
class PPOParams:
    parallel_envs: int = 64
    rollout_steps: int = 256
    training_epochs: int = 4
    minibatch_size: int = 1024
    # minibatch_size should divide rollout_steps * parallel_envs * max_players

    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epsilon: float = 1e-5


@static_data
class PPOState:
    env_state: Any
    obs: Any
    player: Any
    done: Any
    model_state: Any
    memory: Any
    optim_state: Any


def _standardize_ppo_env(env):
    return standardize_interface(
        env,
        init=(("key",), None),
        step=(("key", "state", "action"), None),
    )


def _standardize_ppo_policy(policy):
    if hasattr(policy, "init_memory"):
        return standardize_interface(
            policy,
            init=(("key", "obs"), None),
            act=(("key", "obs", "state", "memory"), None),
            evaluate=(("obs", "action", "state", "memory"), None),
            value=(("obs", "state", "memory"), None),
            init_memory=(("key",), None),
        )

    return standardize_interface(
        policy,
        init=(("key", "obs"), None),
        act=(("key", "obs", "state"), None),
        evaluate=(("obs", "action", "state"), None),
        value=(("obs", "state"), None),
    )


def make_ppo_selfplay(
    params: PPOParams,
    env,
    policy,
    optimizer,
):
    """
    Build a PPO self-play trainer.

    Expected interfaces:
      env.init(key) -> env_state, obs, player, done
      env.step(key, env_state, action) ->
          next_env_state, obs, player, done, reward

      policy.init(key, obs) -> model_state
      policy.act(key, obs, state, memory) ->
          action, logp, value, next_memory
      policy.evaluate(obs, action, state, memory) ->
          logp, value, entropy
      policy.value(obs, state, memory) -> value
      policy.init_memory(key) -> memory
      optimizer.init(key, model_state) -> optim_state
      optimizer.optimize(key, grad, model_state, optim_state) ->
          model_state, optim_state

    For memoryless policies:
      policy.act(key, obs, state) -> action, logp, value
      policy.evaluate(obs, action, state) -> logp, value, entropy
      policy.value(obs, state) -> value

    Shapes (convention):
      obs: (parallel_envs, max_players, ...)
      player: (parallel_envs,)
      reward: (parallel_envs, max_players)
      done: (parallel_envs,) or (parallel_envs, max_players)
      For single-player games, use max_players=1.
    """
    env = _standardize_ppo_env(env)
    has_memory = hasattr(policy, "init_memory")
    policy = _standardize_ppo_policy(policy)
    optimizer = standardize_optimizer(optimizer)
    env = vectorize_aec(env, params.parallel_envs)

    @static_functions
    class PPOSelfPlay:
        init_has_aux = False
        step_has_aux = True

        def init(key):
            env_key, policy_key, optim_key = jrng.split(key, 3)
            env_state, obs, player, done = env.init(env_key)

            if has_memory:
                model_key, memory_key = jrng.split(policy_key)
                model_state = policy.init(model_key, obs)
                memory_keys = jrng.split(
                    memory_key, params.parallel_envs * env.num_players
                )
                memory = jax.vmap(policy.init_memory)(memory_keys)
                memory = jax.tree.map(
                    lambda x: x.reshape(
                        params.parallel_envs, env.num_players, *x.shape[1:]
                    ),
                    memory,
                )
            else:
                model_state = policy.init(policy_key, obs)
                memory = None
            optim_state = optimizer.init(optim_key, model_state)
            return PPOState(
                env_state,
                obs,
                player,
                done,
                model_state,
                memory,
                optim_state,
            )

        def step(key, state):
            num_players = env.num_players

            def _select_by_done(done, x_keep, x_reset):
                mask = done
                while mask.ndim < x_keep.ndim:
                    mask = jnp.expand_dims(mask, axis=-1)
                return jnp.where(mask, x_reset, x_keep)

            def _gather_player(tree, player):
                def _leaf(leaf):
                    idx = jnp.expand_dims(player, axis=-1)
                    while idx.ndim < leaf.ndim:
                        idx = jnp.expand_dims(idx, axis=-1)
                    gathered = jnp.take_along_axis(leaf, idx, axis=1)
                    return jnp.squeeze(gathered, axis=1)
                return jax.tree.map(_leaf, tree)

            def _set_player(tree, player, value):
                env_idx = jnp.arange(params.parallel_envs)

                def _leaf(leaf, val):
                    return leaf.at[env_idx, player].set(val)

                return jax.tree.map(_leaf, tree, value)

            def rollout(carry, key_step):
                (
                    env_state,
                    obs,
                    player,
                    done,
                    memory,
                ) = carry
                policy_key, memory_key, env_key = jrng.split(key_step, 3)

                if has_memory:
                    memory_sel = _gather_player(memory, player)
                    action, logp, value, next_memory = policy.act(
                        policy_key,
                        obs,
                        state.model_state,
                        memory_sel,
                    )
                    _, _, entropy = policy.evaluate(
                        obs,
                        action,
                        state.model_state,
                        memory_sel,
                    )
                else:
                    action, logp, value = policy.act(
                        policy_key,
                        obs,
                        state.model_state,
                    )
                    next_memory = None
                    _, _, entropy = policy.evaluate(
                        obs,
                        action,
                        state.model_state,
                    )
                next_state, next_obs, next_player, next_done, reward = (
                    env.step(env_key, env_state, action)
                )

                if has_memory:
                    memory = _set_player(memory, player, next_memory)
                    memory_keys = jrng.split(
                        memory_key, params.parallel_envs * num_players
                    )
                    init_memory = jax.vmap(policy.init_memory)(memory_keys)
                    init_memory = jax.tree.map(
                        lambda x: x.reshape(
                            params.parallel_envs, num_players, *x.shape[1:]
                        ),
                        init_memory,
                    )
                    memory = jax.tree.map(
                        lambda xk, xr: _select_by_done(next_done, xk, xr),
                        memory,
                        init_memory,
                    )
                data = (
                    obs,
                    action,
                    logp,
                    value,
                    reward,
                    next_done,
                    player,
                    memory_sel if has_memory else None,
                    entropy,
                )
                return (
                    next_state,
                    next_obs,
                    next_player,
                    jnp.zeros_like(done),
                    memory,
                ), data
            
            key, rollout_key = jrng.split(key)
            keys = jrng.split(rollout_key, params.rollout_steps)
            step_data, rollout_data = jax.lax.scan(
                rollout,
                (
                    state.env_state,
                    state.obs,
                    state.player,
                    state.done,
                    state.memory,
                ),
                keys,
            )
            (
                env_state,
                obs, player,
                done,
                memory,
            ) = step_data
            
            (
                traj_obs,
                traj_action,
                traj_logp,
                traj_value,
                traj_env_reward,
                traj_done_env,
                traj_player,
                traj_memory,
                traj_entropy,
            ) = rollout_data
            if has_memory:
                last_value = policy.value(
                    obs,
                    state.model_state,
                    _gather_player(memory, player),
                )
            else:
                last_value = policy.value(obs, state.model_state)

            def _expand_done(done, target):
                while done.ndim < target.ndim:
                    done = jnp.expand_dims(done, axis=-1)
                return jnp.broadcast_to(done, target.shape)

            traj_done = _expand_done(traj_done_env, traj_env_reward)

            def _expand_player_axis(x):
                x = jnp.expand_dims(x, axis=1)
                return x

            def _expand_to_player(x, target):
                while x.ndim < target.ndim:
                    x = jnp.expand_dims(x, axis=-1)
                return x

            def _init_player_carry(last_value, player):
                base_shape = (
                    params.parallel_envs, num_players) + last_value.shape[1:]
                next_value = jnp.zeros(base_shape, dtype=last_value.dtype)
                adv_next = jnp.zeros_like(next_value)
                onehot = jax.nn.one_hot(
                    player, num_players, dtype=last_value.dtype)
                onehot = _expand_to_player(onehot, next_value)
                last_value_exp = _expand_player_axis(last_value)
                last_value_exp = _expand_to_player(last_value_exp, next_value)
                next_value = jnp.where(
                    onehot.astype(jnp.bool_), last_value_exp, next_value)
                return next_value, adv_next

            def gae_step(carry, inputs):
                next_value, adv_next = carry
                reward, value, done, player_t = inputs
                not_done = 1.0 - done.astype(jnp.float32)

                onehot = jax.nn.one_hot(
                    player_t, num_players, dtype=value.dtype)
                onehot = _expand_to_player(onehot, next_value)
                value_exp = _expand_player_axis(value)
                value_exp = _expand_to_player(value_exp, next_value)
                value_all = jnp.where(
                    onehot.astype(jnp.bool_), value_exp, next_value
                )
                discount = onehot * params.discount + (1.0 - onehot)

                delta = (
                    reward
                    + discount
                    * next_value
                    * not_done
                    - value_all
                )
                adv = (
                    delta
                    + discount
                    * params.gae_lambda
                    * not_done
                    * adv_next
                )

                next_value = value_all
                adv_next = adv
                adv_sel = jnp.sum(adv * onehot, axis=1)
                return (next_value, adv_next), adv_sel

            init_carry = _init_player_carry(last_value, player)
            (_, _), advantages = jax.lax.scan(
                gae_step,
                init_carry,
                (traj_env_reward, traj_value, traj_done, traj_player),
                reverse=True,
            )

            returns = advantages + traj_value
            
            adv_mean = jnp.mean(advantages)
            adv_std = jnp.std(advantages)

            term_mask = traj_done_env
            term_count = jnp.maximum(jnp.sum(term_mask), 1.0)
            term_env_rewards = jnp.where(
                term_mask[..., None], traj_env_reward, 0.0
            )

            def _episode_length_mean(done_flags):
                def step(carry, done_t):
                    lengths, sum_lengths, count = carry
                    lengths = lengths + 1
                    done_i = done_t.astype(jnp.int32)
                    sum_lengths = sum_lengths + lengths * done_i
                    count = count + done_i
                    lengths = jnp.where(done_t, 0, lengths)
                    return (lengths, sum_lengths, count), None

                init_lengths = jnp.zeros(
                    (done_flags.shape[1],), dtype=jnp.int32
                )
                init_sum = jnp.zeros_like(init_lengths)
                init_count = jnp.zeros_like(init_lengths)
                (lengths, sum_lengths, count), _ = jax.lax.scan(
                    step,
                    (init_lengths, init_sum, init_count),
                    done_flags,
                )
                total_sum = jnp.sum(sum_lengths)
                total_count = jnp.sum(count)
                return total_sum / jnp.maximum(total_count, 1)

            terminal_steps = jnp.sum(traj_done_env.astype(jnp.int32))
            terminal_envs = jnp.sum(jnp.any(traj_done_env, axis=0))
            episode_return_mean = (
                jnp.sum(term_env_rewards, axis=(0, 1)) / term_count
            )
            episode_length_mean = _episode_length_mean(traj_done_env)

            stats = {
                "adv_mean": adv_mean,
                "adv_std": adv_std,
                "terminal_steps": terminal_steps,
                "terminal_envs": terminal_envs,
                "episode_length_mean": episode_length_mean,
            }
            for player_idx in range(num_players):
                stats[f"episode_return_mean_p{player_idx}"] = (
                    episode_return_mean[player_idx]
                )

            def normalize_adv(adv):
                mean = jnp.mean(adv)
                var = jnp.mean((adv - mean) ** 2)
                return (adv - mean) / (jnp.sqrt(var) + params.epsilon)

            advantages = normalize_adv(advantages)

            dataset = (
                traj_obs,
                traj_action,
                traj_logp,
                advantages,
                returns,
                traj_memory,
            )

            dataset = ravel_tree(dataset, 0, 2)

            def train_batch(model_optim, key_batch, batch):
                model_state, optim_state = model_optim
                (
                    obs_b,
                    act_b,
                    logp_b,
                    adv_b,
                    ret_b,
                    mem_b,
                ) = batch

                def loss_fn(model_state):
                    if has_memory:
                        new_logp, value, entropy = policy.evaluate(
                            obs_b,
                            act_b,
                            model_state,
                            mem_b,
                        )
                    else:
                        new_logp, value, entropy = policy.evaluate(
                            obs_b,
                            act_b,
                            model_state,
                        )
                    ratio = jnp.exp(new_logp - logp_b)
                    clipped = jnp.clip(
                        ratio,
                        1.0 - params.clip_eps,
                        1.0 + params.clip_eps,
                    )
                    policy_loss = -jnp.minimum(ratio * adv_b, clipped * adv_b)
                    value_loss = (ret_b - value) ** 2
                    entropy_loss = -entropy

                    loss = (
                        jnp.mean(policy_loss)
                        + params.value_coef * jnp.mean(value_loss)
                        + params.entropy_coef * jnp.mean(entropy_loss)
                    )
                    aux = (
                        jnp.mean(policy_loss),
                        jnp.mean(value_loss),
                        jnp.mean(entropy),
                    )
                    return loss, aux

                (loss, aux), grad = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(model_state)
                model_state, optim_state = optimizer.optimize(
                    key_batch,
                    grad,
                    model_state,
                    optim_state,
                )
                policy_loss, value_loss, entropy = aux
                return (model_state, optim_state), (
                    loss,
                    policy_loss,
                    value_loss,
                    entropy,
                )

            def train_epoch(model_optim, key_epoch):
                shuffle_key, batch_key = jrng.split(key_epoch)
                shuffled = shuffle_tree(shuffle_key, dataset)
                batches = batch_tree(shuffled, params.minibatch_size)
                num_batches = tree_len(batches, axis=0)
                batch_keys = jrng.split(batch_key, num_batches)
                return jax.lax.scan(
                    lambda mo, kb: train_batch(mo, kb[0], kb[1]),
                    model_optim,
                    (batch_keys, batches),
                )

            key, epoch_key = jrng.split(key)
            epoch_keys = jrng.split(epoch_key, params.training_epochs)
            (model_state, optim_state), losses = jax.lax.scan(
                train_epoch, (state.model_state, state.optim_state), epoch_keys
            )
            loss_values, policy_losses, value_losses, entropies = losses
            stats["loss_mean"] = jnp.mean(loss_values)
            stats["policy_loss_mean"] = jnp.mean(policy_losses)
            stats["value_loss_mean"] = jnp.mean(value_losses)
            stats["entropy_mean"] = jnp.mean(entropies)

            next_state = state.replace(
                env_state=env_state,
                obs=obs,
                player=player,
                done=done,
                model_state=model_state,
                memory=memory,
                optim_state=optim_state,
            )
            return next_state, stats

    return PPOSelfPlay
