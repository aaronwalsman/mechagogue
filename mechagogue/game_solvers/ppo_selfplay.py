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

            def rollout(carry, inputs):
                key_step, step_idx = inputs
                (
                    env_state,
                    obs,
                    player,
                    done,
                    memory,
                    reward_buffer,
                    last_action,
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

                last_action = last_action.at[
                    jnp.arange(params.parallel_envs), player
                ].set(step_idx)
                reward_for_action = jnp.take_along_axis(
                    reward_buffer, player[:, None], axis=1
                )[:, 0]
                reward_buffer = reward_buffer.at[
                    jnp.arange(params.parallel_envs), player].set(0.0)
                reward_buffer = reward_buffer + reward
                bonus_val = jnp.where(next_done[:, None], reward_buffer, 0.0)
                bonus_idx = last_action

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
                reward_buffer = _select_by_done(
                    next_done,
                    reward_buffer,
                    jnp.zeros_like(reward_buffer),
                )
                last_action = _select_by_done(
                    next_done,
                    last_action,
                    jnp.full_like(last_action, -1),
                )
                data = (
                    obs,
                    action,
                    logp,
                    value,
                    reward_for_action,
                    next_done,
                    player,
                    memory_sel if has_memory else None,
                    bonus_idx,
                    bonus_val,
                    entropy,
                    reward,
                )
                return (
                    next_state,
                    next_obs,
                    next_player,
                    jnp.zeros_like(done),
                    memory,
                    reward_buffer,
                    last_action,
                ), data
            
            key, rollout_key = jrng.split(key)
            keys = jrng.split(rollout_key, params.rollout_steps)
            step_idx = jnp.arange(params.rollout_steps, dtype=jnp.int32)
            reward_buffer = jnp.zeros(
                (params.parallel_envs, num_players), dtype=jnp.float32
            )
            last_action = jnp.full(
                (params.parallel_envs, num_players), -1, dtype=jnp.int32
            )
            step_data, rollout_data = jax.lax.scan(
                rollout,
                (
                    state.env_state,
                    state.obs,
                    state.player,
                    state.done,
                    state.memory,
                    reward_buffer,
                    last_action,
                ),
                (keys, step_idx),
            )
            (
                env_state,
                obs, player,
                done,
                memory,
                reward_buffer,
                last_action,
            ) = step_data
            
            (
                traj_obs,
                traj_action,
                traj_logp,
                traj_value,
                traj_reward,
                traj_done_env,
                traj_player,
                traj_memory,
                traj_bonus_idx,
                traj_bonus_val,
                traj_entropy,
                traj_env_reward,
            ) = rollout_data

            def _apply_terminal_bonus(traj_reward, bonus_idx, bonus_val):
                env_idx = jnp.arange(params.parallel_envs)
                env_idx = jnp.broadcast_to(
                    env_idx[None, :, None],
                    bonus_idx.shape,
                )
                time_idx = bonus_idx.reshape(-1)
                env_idx = env_idx.reshape(-1)
                bonus_val = bonus_val.reshape(-1)
                valid = time_idx >= 0
                time_idx = jnp.where(valid, time_idx, 0)
                env_idx = jnp.where(valid, env_idx, 0)
                bonus_val = jnp.where(valid, bonus_val, 0.0)
                return traj_reward.at[time_idx, env_idx].add(bonus_val)

            traj_reward = _apply_terminal_bonus(
                traj_reward, traj_bonus_idx, traj_bonus_val
            )
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

            traj_done = _expand_done(traj_done_env, traj_reward)

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
                next_value_sel = jnp.sum(next_value * onehot, axis=1)
                adv_next_sel = jnp.sum(adv_next * onehot, axis=1)

                delta = (
                    reward
                    + params.discount
                    * next_value_sel
                    * not_done
                    - value
                )
                adv = (
                    delta
                    + params.discount
                    * params.gae_lambda
                    * not_done
                    * adv_next_sel
                )

                value_exp = _expand_player_axis(value)
                value_exp = _expand_to_player(value_exp, next_value)
                adv_exp = _expand_player_axis(adv)
                adv_exp = _expand_to_player(adv_exp, adv_next)
                next_value = jnp.where(
                    onehot.astype(jnp.bool_), value_exp, next_value)
                adv_next = jnp.where(
                    onehot.astype(jnp.bool_), adv_exp, adv_next)
                return (next_value, adv_next), adv

            init_carry = _init_player_carry(last_value, player)
            (_, _), advantages = jax.lax.scan(
                gae_step,
                init_carry,
                (traj_reward, traj_value, traj_done, traj_player),
                reverse=True,
            )

            returns = advantages + traj_value
            raw_advantages = advantages

            stats = {
                "adv_mean": jnp.mean(raw_advantages),
                "reward_mean": jnp.mean(traj_reward),
                "entropy_mean": jnp.mean(traj_entropy),
                "terminal_steps": jnp.sum(
                    traj_done_env.astype(jnp.int32)
                ),
                "terminal_envs": jnp.sum(
                    jnp.any(traj_done_env, axis=0)
                ),
                "draw_steps": jnp.sum(
                    (traj_done_env & (traj_reward == 0))
                    .astype(jnp.int32)
                ),
            }
            onehot = jax.nn.one_hot(traj_player, num_players, dtype=jnp.float32)
            reward_exp = traj_reward[..., None]
            per_player_sum = jnp.sum(reward_exp * onehot, axis=0)
            per_player_mean = jnp.mean(per_player_sum, axis=0)
            stats["return_mean_per_player"] = per_player_mean
            if num_players >= 2:
                stats["return_mean_p0"] = per_player_mean[0]
                stats["return_mean_p1"] = per_player_mean[1]

            if isinstance(traj_obs, dict) and "phase" in traj_obs:
                phase = traj_obs["phase"]
                challenge_mask = (phase == 1) & (traj_action == 1)
                challenge_count = jnp.sum(challenge_mask.astype(jnp.int32))
                challenge_wins = jnp.sum(jnp.where(
                    challenge_mask, traj_reward > 0, False).astype(jnp.int32))
                challenge_losses = jnp.sum(jnp.where(
                    challenge_mask, traj_reward < 0, False).astype(jnp.int32)
                )
                stats["challenge_count"] = challenge_count
                stats["challenge_win_count"] = challenge_wins
                stats["challenge_loss_count"] = challenge_losses

            term_mask = traj_done_env
            term_count = jnp.maximum(jnp.sum(term_mask), 1.0)
            term_env_rewards = jnp.where(
                term_mask[..., None], traj_env_reward, 0.0
            )
            stats["episode_return_mean"] = (
                jnp.sum(term_env_rewards) / term_count
            )
            stats["episode_return_mean_p0"] = (
                jnp.sum(term_env_rewards[..., 0]) / term_count
            )
            if num_players > 1:
                stats["episode_return_mean_p1"] = (
                    jnp.sum(term_env_rewards[..., 1]) / term_count
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
                    return loss

                loss, grad = jax.value_and_grad(loss_fn)(model_state)
                model_state, optim_state = optimizer.optimize(
                    key_batch,
                    grad,
                    model_state,
                    optim_state,
                )
                return (model_state, optim_state), loss

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

            next_state = state.replace(
                env_state=env_state,
                obs=obs,
                player=player,
                done=done,
                model_state=model_state,
                memory=memory,
                optim_state=optim_state,
            )
            return next_state, losses, stats

    return PPOSelfPlay
