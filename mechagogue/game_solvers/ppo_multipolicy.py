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
    minibatch_size: int = 1024
    # minibatch_size should divide rollout_steps * parallel_envs

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
    optimizers = [standardize_optimizer(o) for o in optimizers]

    policy_assignment = jnp.asarray(policy_assignment, dtype=jnp.int32)
    if policy_assignment.shape != (num_players,):
        raise ValueError("policy_assignment must have shape (num_players,).")
    if train_mask is None:
        train_mask = jnp.ones((num_policies,), dtype=jnp.bool_)
    else:
        train_mask = jnp.asarray(train_mask, dtype=jnp.bool_)
        if train_mask.shape != (num_policies,):
            raise ValueError("train_mask must have shape (num_policies,).")

    has_memory = hasattr(policies[0], "init_memory")
    if any(hasattr(p, "init_memory") != has_memory for p in policies):
        raise ValueError("All policies must agree on memory usage.")

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
            optim_states = tuple(
                optimizers[i].init(optim_keys[i], model_states[i])
                for i in range(num_policies)
            )
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

                memory_sel = _gather_player(memory, player) if has_memory else None

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
                    memory_sel if has_memory else None,
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
                traj_memory,
                traj_policy_id,
            ) = rollout_data

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
                traj_policy_id,
            )
            dataset = ravel_tree(dataset, 0, 2)

            def train_epoch_for_policy(i, model_optim, key_epoch):
                shuffle_key, batch_key = jrng.split(key_epoch)
                shuffled = shuffle_tree(shuffle_key, dataset)
                batches = batch_tree(shuffled, params.minibatch_size)
                num_batches = tree_len(batches, axis=0)
                batch_keys = jrng.split(batch_key, num_batches)

                def train_batch(model_optim, key_batch, batch):
                    model_state, optim_state = model_optim
                    (
                        obs_b,
                        act_b,
                        logp_b,
                        adv_b,
                        ret_b,
                        mem_b,
                        pid_b,
                    ) = batch

                    mask = (pid_b == jnp.int32(i)).astype(jnp.float32)
                    denom = jnp.maximum(jnp.sum(mask), 1.0)

                    def loss_fn(model_state):
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
                        policy_loss = -jnp.minimum(ratio * adv_b, clipped * adv_b)
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

            for i in range(num_policies):
                do_train = jnp.bool_(train_mask[i])

                def _train(mo):
                    (mo, losses) = jax.lax.scan(
                        lambda m, ek: train_epoch_for_policy(i, m, ek),
                        mo,
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
                    return mo, stats_vals

                def _skip(mo):
                    zeros = jnp.zeros((4,), dtype=jnp.float32)
                    return mo, zeros

                (model_states[i], optim_states[i]), stats_vals = jax.lax.cond(
                    do_train,
                    _train,
                    _skip,
                    operand=(model_states[i], optim_states[i]),
                )
                stats[f"loss_mean_p{i}"] = stats_vals[0]
                stats[f"policy_loss_mean_p{i}"] = stats_vals[1]
                stats[f"value_loss_mean_p{i}"] = stats_vals[2]
                stats[f"entropy_mean_p{i}"] = stats_vals[3]

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
