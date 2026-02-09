"""Multi-policy PPO training loop for turn-based games.

This is a more general variant of PPO self-play that allows:
  - assigning different policies to different players
  - choosing which policies are updated during training

It is intended as a flexible experimentation harness and may be slower
than the specialized self-play implementation.
"""

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.nn.distributions import categorical
from mechagogue.game_solvers.wrappers import vectorize_aec
from mechagogue.optim.optimizer import standardize_optimizer
from mechagogue.static import static_data, static_functions
from mechagogue.standardize import standardize_interface
from mechagogue.tree import ravel_tree, shuffle_tree, batch_tree, tree_len


@static_data
class MultiPPOParams:
    parallel_envs: int = 64
    rollout_steps: int = 256
    training_epochs: int = 4
    num_minibatches: int = 4
    # num_minibatches should divide rollout_steps * parallel_envs
    sequence_len: int = 1
    # sequence_len should divide rollout_steps when using recurrence

    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epsilon: float = 1e-5


@static_data
class MultiPPOState:
    env_state: Any
    obs: Any
    player: Any
    done: Any
    model_states: Any
    memory: Any
    optim_states: Any


def _standardize_env(env):
    return standardize_interface(
        env,
        init=(("key",), None),
        step=(("key", "state", "action"), None),
    )


def _standardize_policy(policy):
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


def make_ppo_multipolicy(
    params: MultiPPOParams,
    env,
    policies: Sequence,
    optimizers: Sequence,
    policy_assignment,  # shape (num_players,), policy index per player
    train_mask=None,  # shape (num_policies,), True to update
):
    """
    Build a general PPO trainer for turn-based games with multiple policies.

    policy_assignment:
      A length-num_players array mapping player_id -> policy_id.
      This mapping is fixed for the duration of the training step.

    train_mask:
      A length-num_policies boolean array indicating which policies
      should be updated from this rollout.
    """
    env = _standardize_env(env)
    env = vectorize_aec(env, params.parallel_envs)
    num_players = env.num_players

    num_policies = len(policies)
    if len(optimizers) != num_policies:
        raise ValueError("optimizers length must match policies length.")

    policies = [_standardize_policy(p) for p in policies]

    policy_assignment = jnp.asarray(policy_assignment, dtype=jnp.int32)
    if policy_assignment.shape != (num_players,):
        raise ValueError("policy_assignment must have shape (num_players,).")
    if train_mask is None:
        train_mask_tuple = tuple(True for _ in range(num_policies))
    else:
        if not isinstance(train_mask, (list, tuple)):
            raise ValueError(
                "train_mask must be a Python list/tuple of bools."
            )
        train_mask_tuple = tuple(bool(x) for x in train_mask)
        if len(train_mask_tuple) != num_policies:
            raise ValueError("train_mask must have shape (num_policies,).")
    train_mask = jnp.asarray(train_mask_tuple, dtype=jnp.bool_)

    has_memory = hasattr(policies[0], "init_memory")
    if any(hasattr(p, "init_memory") != has_memory for p in policies):
        raise ValueError("All policies must agree on memory usage.")
    if params.sequence_len <= 0:
        raise ValueError("sequence_len must be positive.")
    if params.rollout_steps % params.sequence_len != 0:
        raise ValueError("sequence_len must divide rollout_steps.")
    if has_memory and params.sequence_len > 1:
        for i in range(len(policies)):
            if train_mask_tuple[i]:
                if not hasattr(policies[i], "logits"):
                    raise ValueError(
                        "sequence_len>1 requires policy.logits for all "
                        "trainable recurrent policies."
                    )

    rollout_batch = params.rollout_steps * params.parallel_envs
    if params.num_minibatches <= 0:
        raise ValueError("num_minibatches must be positive.")
    if rollout_batch % params.num_minibatches != 0:
        raise ValueError(
            "num_minibatches must divide rollout_steps * parallel_envs."
        )
    minibatch_size = rollout_batch // params.num_minibatches

    optimizers_std = [None] * num_policies
    for i in range(num_policies):
        if train_mask_tuple[i]:
            if optimizers[i] is None:
                raise ValueError(
                    "optimizer is None for a trainable policy. "
                    "Either provide an optimizer or set train_mask[i]=False."
                )
            optimizers_std[i] = standardize_optimizer(optimizers[i])

    def _gather_player(tree, player):
        def _leaf(leaf):
            idx = jnp.expand_dims(player, axis=-1)
            while idx.ndim < leaf.ndim:
                idx = jnp.expand_dims(idx, axis=-1)
            gathered = jnp.take_along_axis(leaf, idx, axis=1)
            return jnp.squeeze(gathered, axis=1)
        return jax.tree.map(_leaf, tree)

    def _gather_player_time(tree, player):
        def _leaf(leaf):
            idx = jnp.expand_dims(player, axis=-1)
            while idx.ndim < leaf.ndim:
                idx = jnp.expand_dims(idx, axis=-1)
            gathered = jnp.take_along_axis(leaf, idx, axis=2)
            return jnp.squeeze(gathered, axis=2)
        return jax.tree.map(_leaf, tree)

    def _set_player(tree, player, value):
        env_idx = jnp.arange(params.parallel_envs)

        def _leaf(leaf, val):
            return leaf.at[env_idx, player].set(val)

        return jax.tree.map(_leaf, tree, value)

    def _select_by_done(done, x_keep, x_reset):
        mask = done
        while mask.ndim < x_keep.ndim:
            mask = jnp.expand_dims(mask, axis=-1)
        return jnp.where(mask, x_reset, x_keep)

    @static_functions
    class PPOMultiPolicy:
        init_has_aux = False
        step_has_aux = True

        def init(key):
            env_key, policy_key, optim_key = jrng.split(key, 3)
            env_state, obs, player, done = env.init(env_key)

            policy_keys = jrng.split(policy_key, num_policies)
            model_states = tuple(
                policies[i].init(policy_keys[i], obs)
                for i in range(num_policies)
            )

            if has_memory:
                memory_keys = jrng.split(
                    policy_key, params.parallel_envs * num_players
                )
                memory = jax.vmap(policies[0].init_memory)(memory_keys)
                memory = jax.tree.map(
                    lambda x: x.reshape(
                        params.parallel_envs, num_players, *x.shape[1:]
                    ),
                    memory,
                )
            else:
                memory = None

            optim_keys = jrng.split(optim_key, num_policies)
            optim_states = []
            for i in range(num_policies):
                if train_mask_tuple[i]:
                    optim_states.append(
                        optimizers_std[i].init(
                            optim_keys[i], model_states[i]
                        )
                    )
                else:
                    optim_states.append(None)
            optim_states = tuple(optim_states)
            return MultiPPOState(
                env_state,
                obs,
                player,
                done,
                model_states,
                memory,
                optim_states,
            )

        def step(key, state):
            def rollout(carry, key_step):
                env_state, obs, player, done, memory = carry
                policy_key, memory_key, env_key = jrng.split(key_step, 3)

                policy_id = policy_assignment[player]
                policy_keys = jrng.split(policy_key, num_policies)

                memory_sel = (
                    _gather_player(memory, player) if has_memory else None)

                def _act_for_policy(i):
                    if has_memory:
                        return policies[i].act(
                            policy_keys[i],
                            obs,
                            state.model_states[i],
                            memory_sel,
                        )
                    return policies[i].act(
                        policy_keys[i],
                        obs,
                        state.model_states[i],
                    )

                acts = [ _act_for_policy(i) for i in range(num_policies) ]
                actions = jnp.stack([a[0] for a in acts], axis=1)
                logps = jnp.stack([a[1] for a in acts], axis=1)
                values = jnp.stack([a[2] for a in acts], axis=1)
                if has_memory:
                    next_memories = jnp.stack([a[3] for a in acts], axis=1)
                else:
                    next_memories = None

                gather_idx = policy_id[:, None]
                action = jnp.take_along_axis(actions, gather_idx, axis=1)[:, 0]
                logp = jnp.take_along_axis(logps, gather_idx, axis=1)[:, 0]
                value = jnp.take_along_axis(values, gather_idx, axis=1)[:, 0]
                if has_memory:
                    next_memory = jnp.take_along_axis(
                        next_memories, gather_idx[..., None], axis=1
                    )[:, 0]
                else:
                    next_memory = None

                next_state, next_obs, next_player, next_done, reward = (
                    env.step(env_key, env_state, action)
                )

                if has_memory:
                    memory = _set_player(memory, player, next_memory)
                    memory_keys = jrng.split(
                        memory_key, params.parallel_envs * num_players
                    )
                    init_memory = jax.vmap(policies[0].init_memory)(memory_keys)
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
                    memory if has_memory else None,
                    policy_id,
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
            env_state, obs, player, done, memory = step_data

            (
                traj_obs,
                traj_action,
                traj_logp,
                traj_value,
                traj_env_reward,
                traj_done_env,
                traj_player,
                traj_memory_all,
                traj_policy_id,
            ) = rollout_data
            expected_policy_id = policy_assignment[traj_player]
            if has_memory:
                if traj_memory_all.shape[0] == traj_player.shape[0]:
                    traj_memory = _gather_player_time(
                        traj_memory_all, traj_player
                    )
                else:
                    traj_memory = _gather_player(traj_memory_all, traj_player)
            else:
                traj_memory = None

            def _episode_return_stats(reward, done):
                reward = reward.astype(jnp.float32)
                done = done.astype(jnp.float32)

                def step(carry, inputs):
                    running, total, count = carry
                    r_t, d_t = inputs
                    running = running + r_t
                    done_mask = d_t[:, None]
                    total = total + jnp.sum(running * done_mask, axis=0)
                    count = count + jnp.sum(done_mask, axis=0)
                    running = jnp.where(
                        done_mask.astype(jnp.bool_),
                        jnp.zeros_like(running),
                        running,
                    )
                    return (running, total, count), None

                init_running = jnp.zeros(
                    (params.parallel_envs, num_players), dtype=reward.dtype
                )
                init_total = jnp.zeros((num_players,), dtype=reward.dtype)
                init_count = jnp.zeros((num_players,), dtype=reward.dtype)
                (_, total, count), _ = jax.lax.scan(
                    step,
                    (init_running, init_total, init_count),
                    (reward, done),
                )
                mean = total / jnp.maximum(count, 1.0)
                return total, count, mean

            ep_return_total, ep_return_count, ep_return_mean = (
                _episode_return_stats(traj_env_reward, traj_done_env)
            )

            if has_memory:
                last_value = policies[0].value(
                    obs,
                    state.model_states[0],
                    _gather_player(memory, player),
                )
            else:
                last_value = policies[0].value(obs, state.model_states[0])

            def _expand_done(done, target):
                while done.ndim < target.ndim:
                    done = jnp.expand_dims(done, axis=-1)
                return jnp.broadcast_to(done, target.shape)

            traj_done = _expand_done(traj_done_env, traj_env_reward)

            def _expand_player_axis(x):
                return jnp.expand_dims(x, axis=1)

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
            advantages_raw = advantages

            def normalize_adv_per_policy(adv, policy_id):
                adv_flat = ravel_tree(adv, 0, 2)
                pid_flat = ravel_tree(policy_id, 0, 2)
                adv_all = adv_flat
                pid_all = pid_flat

                def per_policy(i):
                    mask = (pid_all == jnp.int32(i)).astype(adv_all.dtype)
                    denom = jnp.maximum(jnp.sum(mask), 1.0)
                    mean = jnp.sum(adv_all * mask) / denom
                    var = jnp.sum(((adv_all - mean) ** 2) * mask) / denom
                    return mean, var

                means, vars_ = jax.vmap(per_policy)(jnp.arange(num_policies))
                means = means[pid_flat]
                vars_ = vars_[pid_flat]
                normalized = (adv_flat - means) / (
                    jnp.sqrt(vars_) + params.epsilon
                )
                return normalized.reshape(adv.shape)

            advantages = normalize_adv_per_policy(advantages, traj_policy_id)

            def _policy_stats(values, policy_id):
                values_flat = ravel_tree(values, 0, 2)
                pid_flat = ravel_tree(policy_id, 0, 2)

                def per_policy(i):
                    mask = (pid_flat == jnp.int32(i)).astype(values_flat.dtype)
                    denom = jnp.maximum(jnp.sum(mask), 1.0)
                    mean = jnp.sum(values_flat * mask) / denom
                    var = jnp.sum(((values_flat - mean) ** 2) * mask) / denom
                    std = jnp.sqrt(var + params.epsilon)
                    return mean, std

                return jax.vmap(per_policy)(jnp.arange(num_policies))

            trainable_indices = [
                i for i in range(num_policies) if train_mask_tuple[i]
            ]
            adv_raw_means, adv_raw_stds = _policy_stats(
                advantages_raw, traj_policy_id
            )
            ret_means, ret_stds = _policy_stats(returns, traj_policy_id)
            val_means, val_stds = _policy_stats(traj_value, traj_policy_id)

            if has_memory and params.sequence_len > 1:
                seq_len = params.sequence_len
                num_envs = params.parallel_envs
                num_seq = num_envs * (params.rollout_steps // seq_len)

                def _to_sequences(x):
                    x = jnp.transpose(
                        x, (1, 0) + tuple(range(2, x.ndim))
                    )
                    new_shape = (num_envs, -1, seq_len) + x.shape[2:]
                    x = x.reshape(new_shape)
                    x = x.reshape((num_seq, seq_len) + x.shape[3:])
                    return x

                seq_obs = jax.tree.map(_to_sequences, traj_obs)
                seq_action = _to_sequences(traj_action)
                seq_logp = _to_sequences(traj_logp)
                seq_adv = _to_sequences(advantages)
                seq_ret = _to_sequences(returns)
                seq_done = _to_sequences(traj_done_env)
                seq_pid = _to_sequences(traj_policy_id)
                seq_mem_all = jax.tree.map(_to_sequences, traj_memory_all)

                dataset = (
                    seq_obs,
                    seq_action,
                    seq_logp,
                    seq_adv,
                    seq_ret,
                    seq_done,
                    seq_pid,
                    seq_mem_all,
                )
            else:
                dataset = (
                    traj_obs,
                    traj_action,
                    traj_logp,
                    advantages,
                    returns,
                    traj_memory,
                    traj_policy_id,
                )
                dataset = ravel_tree(dataset, 0, 2)
            
            def train_epoch_for_policy(i, model_optim, key_epoch):
                shuffle_key, batch_key = jrng.split(key_epoch)
                if has_memory and params.sequence_len > 1:
                    shuffled = shuffle_tree(shuffle_key, dataset)
                    total_seq = tree_len(shuffled, axis=0)
                    seq_batch = total_seq // params.num_minibatches
                    batches = batch_tree(shuffled, seq_batch)
                    num_batches = tree_len(batches, axis=0)
                else:
                    shuffled = shuffle_tree(shuffle_key, dataset)
                    batches = batch_tree(shuffled, minibatch_size)
                    num_batches = tree_len(batches, axis=0)
                batch_keys = jrng.split(batch_key, num_batches)

                def train_batch(model_optim, key_batch, batch):
                    model_state, optim_state = model_optim
                    if has_memory and params.sequence_len > 1:
                        (
                            obs_b,
                            act_b,
                            logp_b,
                            adv_b,
                            ret_b,
                            done_b,
                            pid_b,
                            mem_all_b,
                        ) = batch
                    else:
                        (
                            obs_b,
                            act_b,
                            logp_b,
                            adv_b,
                            ret_b,
                            mem_b,
                            pid_b,
                        ) = batch

                    def loss_fn(model_state):
                        if has_memory and params.sequence_len > 1:
                            def _swap_time_batch(x):
                                return jnp.swapaxes(x, 0, 1)

                            obs_t = jax.tree.map(_swap_time_batch, obs_b)
                            act_t = _swap_time_batch(act_b)
                            logp_t = _swap_time_batch(logp_b)
                            adv_t = _swap_time_batch(adv_b)
                            ret_t = _swap_time_batch(ret_b)
                            done_t = _swap_time_batch(done_b)
                            pid_t = _swap_time_batch(pid_b)
                            mem_all_t = jax.tree.map(_swap_time_batch, mem_all_b)

                            init_mem_i = jax.tree.map(
                                lambda x: x[:, 0, i],
                                mem_all_b,
                            )
                            init_reset = policies[i].init_memory(jrng.key(0))

                            def step(carry, inputs):
                                memory_i = carry
                                (
                                    obs_s,
                                    act_s,
                                    logp_s,
                                    adv_s,
                                    ret_s,
                                    done_s,
                                    pid_s,
                                ) = inputs

                                logits, value, next_memory_i, mask = (
                                    policies[i].logits(
                                        obs_s, model_state, memory_i
                                    )
                                )
                                masked_logits = jnp.where(mask, logits, -1e9)
                                dist = categorical(masked_logits)
                                new_logp = dist.logp(act_s)
                                entropy = dist.entropy()

                                ratio = jnp.exp(new_logp - logp_s)
                                clipped = jnp.clip(
                                    ratio,
                                    1.0 - params.clip_eps,
                                    1.0 + params.clip_eps,
                                )
                                policy_loss = -jnp.minimum(
                                    ratio * adv_s, clipped * adv_s
                                )
                                value_loss = (ret_s - value) ** 2
                                entropy_loss = -entropy

                                mask_pid = (pid_s == jnp.int32(i)).astype(
                                    jnp.float32
                                )
                                done_s = done_s.astype(jnp.bool_)

                                def _apply_mask(mask, x_keep, x_update):
                                    while mask.ndim < x_keep.ndim:
                                        mask = jnp.expand_dims(mask, axis=-1)
                                    return jnp.where(mask, x_update, x_keep)

                                reset_mem = jax.tree.map(
                                    lambda x: jnp.broadcast_to(
                                        x, next_memory_i.shape
                                    ),
                                    init_reset,
                                )
                                memory_i = jax.tree.map(
                                    lambda nm, mi: _apply_mask(mask_pid, mi, nm),
                                    next_memory_i,
                                    memory_i,
                                )
                                memory_i = jax.tree.map(
                                    lambda nm, rm: _select_by_done(
                                        done_s, nm, rm
                                    ),
                                    memory_i,
                                    reset_mem,
                                )
                                return memory_i, (
                                    policy_loss,
                                    value_loss,
                                    entropy,
                                    mask_pid,
                                )

                            init_mem_i = jax.tree.map(
                                lambda x: x[:, 0, i],
                                mem_all_b,
                            )
                            _, losses = jax.lax.scan(
                                step,
                                init_mem_i,
                                (
                                    obs_t,
                                    act_t,
                                    logp_t,
                                    adv_t,
                                    ret_t,
                                    done_t,
                                    pid_t,
                                ),
                            )
                            policy_loss, value_loss, entropy, mask_pid = losses
                            denom = jnp.maximum(jnp.sum(mask_pid), 1.0)
                            loss = (
                                jnp.sum(policy_loss * mask_pid) / denom
                                + params.value_coef
                                * (jnp.sum(value_loss * mask_pid) / denom)
                                + params.entropy_coef
                                * (jnp.sum(-entropy * mask_pid) / denom)
                            )
                            aux = (
                                jnp.sum(policy_loss * mask_pid) / denom,
                                jnp.sum(value_loss * mask_pid) / denom,
                                jnp.sum(entropy * mask_pid) / denom,
                            )
                            return loss, aux
                        else:
                            mask = (pid_b == jnp.int32(i)).astype(jnp.float32)
                            denom = jnp.maximum(jnp.sum(mask), 1.0)
                            if has_memory:
                                new_logp, value, entropy = policies[i].evaluate(
                                    obs_b,
                                    act_b,
                                    model_state,
                                    mem_b,
                                )
                            else:
                                new_logp, value, entropy = policies[i].evaluate(
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
                            policy_loss = -jnp.minimum(
                                ratio * adv_b, clipped * adv_b)
                            value_loss = (ret_b - value) ** 2
                            entropy_loss = -entropy

                            loss = (
                                jnp.sum(policy_loss * mask) / denom
                                + params.value_coef
                                * (jnp.sum(value_loss * mask) / denom)
                                + params.entropy_coef
                                * (jnp.sum(entropy_loss * mask) / denom)
                            )
                            aux = (
                                jnp.sum(policy_loss * mask) / denom,
                                jnp.sum(value_loss * mask) / denom,
                                jnp.sum(entropy * mask) / denom,
                            )
                            return loss, aux

                    (loss, aux), grad = jax.value_and_grad(
                        loss_fn, has_aux=True
                    )(model_state)
                    model_state, optim_state = optimizers[i].optimize(
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

                return jax.lax.scan(
                    lambda mo, kb: train_batch(mo, kb[0], kb[1]),
                    model_optim,
                    (batch_keys, batches),
                )

            key, epoch_key = jrng.split(key)
            epoch_keys = jrng.split(epoch_key, params.training_epochs)

            model_states = list(state.model_states)
            optim_states = list(state.optim_states)
            stats = {}

            for i in trainable_indices:
                (model_states[i], optim_states[i]), losses = jax.lax.scan(
                    lambda m, ek: train_epoch_for_policy(i, m, ek),
                    (model_states[i], optim_states[i]),
                    epoch_keys,
                )
                loss_values, policy_losses, value_losses, entropies = losses
                stats_vals = jnp.stack(
                    [
                    jnp.mean(loss_values),
                    jnp.mean(policy_losses),
                    jnp.mean(value_losses),
                    jnp.mean(entropies),
                    ],
                    axis=0,
                )
                stats[f"loss_mean_p{i}"] = stats_vals[0]
                stats[f"policy_loss_mean_p{i}"] = stats_vals[1]
                stats[f"value_loss_mean_p{i}"] = stats_vals[2]
                stats[f"entropy_mean_p{i}"] = stats_vals[3]
                stats[f"adv_raw_mean_p{i}"] = adv_raw_means[i]
                stats[f"adv_raw_std_p{i}"] = adv_raw_stds[i]
                stats[f"ret_mean_p{i}"] = ret_means[i]
                stats[f"ret_std_p{i}"] = ret_stds[i]
                stats[f"value_mean_p{i}"] = val_means[i]
                stats[f"value_std_p{i}"] = val_stds[i]
            for i in range(num_players):
                stats[f"raw_return_mean_p{i}"] = ep_return_mean[i]
                stats[f"raw_return_sum_p{i}"] = ep_return_total[i]
                stats[f"raw_return_count_p{i}"] = ep_return_count[i]

            next_state = state.replace(
                env_state=env_state,
                obs=obs,
                player=player,
                done=done,
                model_states=tuple(model_states),
                memory=memory,
                optim_states=tuple(optim_states),
            )
            return next_state, stats

    return PPOMultiPolicy
