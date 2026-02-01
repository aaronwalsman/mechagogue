"""Independent PPO training loop for turn-based games."""

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
class IPPOParams:
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
class IPPOState:
    env_state: Any
    obs: Any
    player: Any
    done: Any
    model_states: Any
    memories: Any
    optim_states: Any


def _standardize_ippo_env(env):
    return standardize_interface(
        env,
        init=(("key",), None),
        step=(("key", "state", "action"), None),
    )


def _standardize_ippo_policy(policy):
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


def make_ippo(
    params: IPPOParams,
    env,
    policies: Sequence,
    optimizers: Sequence,
):
    """
    Build an independent PPO trainer for multi-agent AEC games.

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
    env = _standardize_ippo_env(env)
    env = vectorize_aec(env, params.parallel_envs)
    num_players = env.num_players

    if len(policies) != num_players:
        raise ValueError(
            "policies length must match env.num_players."
        )
    if len(optimizers) != num_players:
        raise ValueError(
            "optimizers length must match env.num_players."
        )

    has_memory = tuple(hasattr(policy, "init_memory") for policy in policies)
    policies = tuple(_standardize_ippo_policy(p) for p in policies)
    optimizers = tuple(standardize_optimizer(o) for o in optimizers)

    def _obs_for_player(obs, player_idx):
        return jax.tree.map(lambda x: x[:, player_idx], obs)

    def _stack_tree(values):
        return jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *values)

    @static_functions
    class IPPO:
        init_has_aux = False
        step_has_aux = True

        def init(key):
            env_key, policy_key, optim_key = jrng.split(key, 3)
            env_state, obs, player, done = env.init(env_key)

            policy_keys = jrng.split(policy_key, num_players)
            optim_keys = jrng.split(optim_key, num_players)
            model_states = []
            memories = []
            optim_states = []
            for idx in range(num_players):
                obs_i = _obs_for_player(obs, idx)
                if has_memory[idx]:
                    model_key, memory_key = jrng.split(
                        policy_keys[idx], 2
                    )
                    model_state = policies[idx].init(model_key, obs_i)
                    memory = policies[idx].init_memory(memory_key)
                else:
                    model_state = policies[idx].init(
                        policy_keys[idx], obs_i
                    )
                    memory = None
                model_states.append(model_state)
                memories.append(memory)
                optim_state = optimizers[idx].init(
                    optim_keys[idx], model_state
                )
                optim_states.append(optim_state)

            return IPPOState(
                env_state,
                obs,
                player,
                done,
                tuple(model_states),
                tuple(memories),
                tuple(optim_states),
            )

        def step(key, state):
            def _active_from_player(player, done):
                players = jnp.arange(num_players, dtype=jnp.int32)
                active = players == jnp.expand_dims(player, axis=-1)
                done = jnp.asarray(done)
                if done.ndim == 0:
                    done = jnp.expand_dims(done, axis=0)
                done = jnp.expand_dims(done, axis=-1)
                return active & ~done

            def rollout(carry, key_step):
                env_state, obs, player, done, memories = carry
                keys = jrng.split(key_step, num_players * 2 + 1)
                env_key = keys[0]
                policy_keys = keys[1:1 + num_players]
                memory_keys = keys[1 + num_players:]

                actions = []
                logps = []
                values = []
                next_memories = []
                for idx in range(num_players):
                    obs_i = _obs_for_player(obs, idx)
                    if has_memory[idx]:
                        action, logp, value, next_memory = policies[idx].act(
                            policy_keys[idx],
                            obs_i,
                            state.model_states[idx],
                            memories[idx],
                        )
                    else:
                        action, logp, value = policies[idx].act(
                            policy_keys[idx],
                            obs_i,
                            state.model_states[idx],
                        )
                        next_memory = None
                    actions.append(action)
                    logps.append(logp)
                    values.append(value)
                    next_memories.append(next_memory)

                action = _stack_tree(actions)
                logp = _stack_tree(logps)
                value = _stack_tree(values)

                next_state, next_obs, next_player, next_done, reward = (
                    env.step(env_key, env_state, action)
                )

                for idx in range(num_players):
                    if not has_memory[idx]:
                        continue
                    memory_keys_i = jrng.split(
                        memory_keys[idx], params.parallel_envs
                    )
                    init_memory = jax.vmap(
                        policies[idx].init_memory
                    )(memory_keys_i)
                    reset_mask = next_done
                    while reset_mask.ndim < next_memories[idx].ndim:
                        reset_mask = jnp.expand_dims(reset_mask, axis=-1)
                    next_memories[idx] = jnp.where(
                        reset_mask,
                        init_memory,
                        next_memories[idx],
                    )

                active = _active_from_player(player, done)
                data = (
                    obs,
                    action,
                    logp,
                    value,
                    reward,
                    next_done,
                    active,
                    memories,
                )
                return (
                    next_state,
                    next_obs,
                    next_player,
                    jnp.zeros_like(done),
                    tuple(next_memories),
                ), data

            key, rollout_key = jrng.split(key)
            keys = jrng.split(rollout_key, params.rollout_steps)
            (env_state, obs, player, done, memories), rollout_data = (
                jax.lax.scan(
                    rollout,
                    (
                        state.env_state,
                        state.obs,
                        state.player,
                        state.done,
                        state.memories,
                    ),
                    keys,
                )
            )

            (
                traj_obs,
                traj_action,
                traj_logp,
                traj_value,
                traj_reward,
                traj_done_env,
                traj_active,
                traj_memories,
            ) = rollout_data

            last_values = []
            for idx in range(num_players):
                obs_i = _obs_for_player(obs, idx)
                if has_memory[idx]:
                    last_value = policies[idx].value(
                        obs_i, state.model_states[idx], memories[idx]
                    )
                else:
                    last_value = policies[idx].value(
                        obs_i, state.model_states[idx]
                    )
                last_values.append(last_value)
            last_value = jnp.stack(last_values, axis=1)

            def _expand_done(done, target):
                while done.ndim < target.ndim:
                    done = jnp.expand_dims(done, axis=-1)
                return jnp.broadcast_to(done, target.shape)

            traj_done = _expand_done(traj_done_env, traj_reward)

            def gae_step(carry, inputs):
                next_value, adv_next = carry
                reward, value, done = inputs
                not_done = 1.0 - done.astype(jnp.float32)
                delta = reward + params.discount * next_value * not_done - value
                adv = (
                    delta
                    + params.discount
                    * params.gae_lambda
                    * not_done
                    * adv_next
                )
                return (value, adv), adv

            advantages_list = []
            returns_list = []
            for idx in range(num_players):
                init_carry = (
                    last_value[:, idx],
                    jnp.zeros_like(last_value[:, idx]),
                )
                (_, _), advantages = jax.lax.scan(
                    gae_step,
                    init_carry,
                    (
                        traj_reward[:, :, idx],
                        traj_value[:, :, idx],
                        traj_done[:, :, idx],
                    ),
                    reverse=True,
                )
                returns = advantages + traj_value[:, :, idx]
                advantages_list.append(advantages)
                returns_list.append(returns)

            advantages = jnp.stack(advantages_list, axis=2)
            returns = jnp.stack(returns_list, axis=2)

            def masked_mean(values, mask):
                mask_f = mask.astype(jnp.float32)
                denom = jnp.maximum(jnp.sum(mask_f), 1.0)
                return jnp.sum(values * mask_f) / denom

            active_mask = traj_active
            inactive_mask = ~traj_active
            stats = {
                "adv_active_mean": masked_mean(
                    advantages, active_mask
                ),
                "adv_inactive_mean": masked_mean(
                    advantages, inactive_mask
                ),
                "reward_mean": jnp.mean(traj_reward),
                "reward_active_mean": masked_mean(
                    traj_reward, active_mask
                ),
                "terminal_steps": jnp.sum(
                    traj_done_env.astype(jnp.int32)
                ),
                "terminal_envs": jnp.sum(
                    jnp.any(traj_done_env, axis=0)
                ),
                "draw_steps": jnp.sum(
                    (traj_done_env & jnp.all(traj_reward == 0, axis=-1))
                    .astype(jnp.int32)
                ),
            }

            def normalize_adv(adv, mask):
                mask_f = mask.astype(jnp.float32)
                count = jnp.maximum(jnp.sum(mask_f), 1.0)
                mean = jnp.sum(adv * mask_f) / count
                var = jnp.sum(((adv - mean) ** 2) * mask_f) / count
                return (adv - mean) / (jnp.sqrt(var) + params.epsilon)

            norm_advantages = []
            for idx in range(num_players):
                norm_adv = normalize_adv(
                    advantages[:, :, idx],
                    traj_active[:, :, idx],
                )
                norm_advantages.append(norm_adv)
            advantages = jnp.stack(norm_advantages, axis=2)

            losses = []
            next_model_states = []
            next_optim_states = []
            for idx in range(num_players):
                obs_i = jax.tree.map(
                    lambda x: x[:, :, idx], traj_obs
                )
                action_i = jax.tree.map(
                    lambda x: x[:, :, idx], traj_action
                )
                dataset = (
                    obs_i,
                    action_i,
                    traj_logp[:, :, idx],
                    advantages[:, :, idx],
                    returns[:, :, idx],
                    traj_active[:, :, idx],
                    traj_done[:, :, idx],
                    traj_memories[idx],
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
                        mask_b,
                        done_b,
                        mem_b,
                    ) = batch

                    def loss_fn(model_state):
                        if has_memory[idx]:
                            new_logp, value, entropy = (
                                policies[idx].evaluate(
                                    obs_b,
                                    act_b,
                                    model_state,
                                    mem_b,
                                )
                            )
                        else:
                            new_logp, value, entropy = (
                                policies[idx].evaluate(
                                    obs_b,
                                    act_b,
                                    model_state,
                                )
                            )
                        ratio = jnp.exp(new_logp - logp_b)
                        clipped = jnp.clip(
                            ratio,
                            1.0 - params.clip_eps,
                            1.0 + params.clip_eps,
                        )
                        policy_loss = -jnp.minimum(
                            ratio * adv_b, clipped * adv_b
                        )
                        value_loss = (ret_b - value) ** 2
                        entropy_loss = -entropy

                        policy_mask = mask_b.astype(jnp.float32)
                        policy_mask = policy_mask / jnp.maximum(
                            jnp.sum(policy_mask), 1.0
                        )
                        value_mask = jnp.ones_like(policy_mask)
                        done_mask = done_b.astype(jnp.float32)
                        if done_mask.ndim < value_mask.ndim:
                            done_mask = jnp.expand_dims(
                                done_mask, axis=-1
                            )
                        value_mask = value_mask * done_mask
                        value_mask = value_mask / jnp.maximum(
                            jnp.sum(value_mask), 1.0
                        )
                        loss = (
                            jnp.sum(policy_loss * policy_mask)
                            + params.value_coef
                            * jnp.sum(value_loss * value_mask)
                            + params.entropy_coef
                            * jnp.sum(entropy_loss * policy_mask)
                        )
                        return loss

                    loss, grad = jax.value_and_grad(loss_fn)(model_state)
                    model_state, optim_state = optimizers[idx].optimize(
                        key_batch,
                        grad,
                        model_state,
                        optim_state,
                    )
                    return (model_state, optim_state), loss

                def train_epoch(model_optim, key_epoch):
                    shuffle_key, batch_key = jrng.split(key_epoch)
                    shuffled = shuffle_tree(shuffle_key, dataset)
                    batches = batch_tree(
                        shuffled, params.minibatch_size
                    )
                    num_batches = tree_len(batches, axis=0)
                    batch_keys = jrng.split(batch_key, num_batches)
                    return jax.lax.scan(
                        lambda mo, kb: train_batch(
                            mo, kb[0], kb[1]
                        ),
                        model_optim,
                        (batch_keys, batches),
                    )

                key, epoch_key = jrng.split(key)
                epoch_keys = jrng.split(
                    epoch_key, params.training_epochs
                )
                (model_state, optim_state), loss = jax.lax.scan(
                    train_epoch,
                    (state.model_states[idx], state.optim_states[idx]),
                    epoch_keys,
                )
                next_model_states.append(model_state)
                next_optim_states.append(optim_state)
                losses.append(loss)

            next_state = state.replace(
                env_state=env_state,
                obs=obs,
                player=player,
                done=done,
                model_states=tuple(next_model_states),
                memories=memories,
                optim_states=tuple(next_optim_states),
            )
            return next_state, tuple(losses), stats

    return IPPO
