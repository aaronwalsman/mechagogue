import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.envs.nim import make_nim
from mechagogue.epoch import make_epoch_system
from mechagogue.game_solvers.ppo_selfplay import PPOParams, make_ppo_selfplay
from mechagogue.game_solvers.wrappers import vectorize_aec
from mechagogue.nn.distributions import categorical
from mechagogue.nn.linear import embedding_layer, linear_layer
from mechagogue.nn.mlp import mlp
from mechagogue.optim.adam import adam
from mechagogue.static import static_data, static_functions


@static_data
class NimPolicyState:
    pile_embed: jnp.ndarray
    to_play_embed: jnp.ndarray
    player_embed: jnp.ndarray
    trunk: jnp.ndarray
    policy_head: jnp.ndarray
    value_head: jnp.ndarray


def make_nim_policy(max_pile):
    pile_tokens = max_pile + 1
    embed_dim = 16
    trunk_dim = 64
    action_dim = 2

    pile_embed = embedding_layer(pile_tokens, embed_dim)
    to_play_embed = embedding_layer(2, embed_dim)
    player_embed = embedding_layer(2, embed_dim)
    trunk = mlp(2, embed_dim * 3, trunk_dim, trunk_dim, use_bias=True)
    policy_head = linear_layer(trunk_dim, action_dim, use_bias=True)
    value_head = linear_layer(trunk_dim, 1, use_bias=True)

    @static_functions
    class NimPolicy:
        def init(key, obs):
            keys = jrng.split(key, 6)
            model_state = NimPolicyState(
                pile_embed=pile_embed.init(keys[0]),
                to_play_embed=to_play_embed.init(keys[1]),
                player_embed=player_embed.init(keys[2]),
                trunk=trunk.init(keys[3]),
                policy_head=policy_head.init(keys[4]),
                value_head=value_head.init(keys[5]),
            )
            return model_state

        def act(key, obs, state):
            logits, value, mask = _forward(obs, state)
            safe_logits = jnp.where(mask, logits, -1e9)
            dist = categorical(safe_logits)
            action = dist.sample(key)
            logp = dist.logp(action)
            return action, logp, value

        def evaluate(obs, action, state):
            logits, value, mask = _forward(obs, state)
            safe_logits = jnp.where(mask, logits, -1e9)
            dist = categorical(safe_logits)
            logp = dist.logp(action)
            entropy = dist.entropy()
            return logp, value, entropy

        def value(obs, state):
            _, value, _ = _forward(obs, state)
            return value

        def logits(obs, state):
            logits, value, mask = _forward(obs, state)
            return logits, value, mask

    def _forward(obs, state):
        pile = obs["pile"]
        to_play = obs["to_play"]
        player = obs["player"]
        mask = obs["action_mask"]

        pile_emb = pile_embed.forward(pile, state.pile_embed)
        to_play_emb = to_play_embed.forward(to_play, state.to_play_embed)
        player_emb = player_embed.forward(player, state.player_embed)
        x = jnp.concatenate([pile_emb, to_play_emb, player_emb], axis=-1)
        key = jrng.key(0)
        x = trunk.forward(key, x, state.trunk)
        logits = policy_head.forward(x, state.policy_head)
        value = value_head.forward(x, state.value_head)[..., 0]
        return logits, value, mask

    return NimPolicy


def _obs_for_player(obs, player):
    return jax.tree.map(lambda x: x[:, player], obs)


def _make_report(state, losses, stats=None):
    loss = jnp.mean(losses)
    if stats is None:
        done = state.done.astype(jnp.int32)
        done_count = jnp.sum(done)
        winner = state.env_state.winner
        draw = (done == 1) & (winner < 0)
        draw_count = jnp.sum(draw)
    else:
        done_count = stats["terminal_steps"]
        draw_count = stats.get("draw_steps", jnp.int32(0))
    report = {
        "loss_mean": loss,
        "done_count": done_count,
        "draw_count": draw_count,
    }
    if stats is not None:
        report.update(stats)
    return report


def _log(key, epoch, state, reports):
    loss = float(jnp.mean(jax.device_get(reports["loss_mean"])))
    done_count = int(jnp.sum(jax.device_get(reports["done_count"])))
    draw_count = int(jnp.sum(jax.device_get(reports["draw_count"])))
    adv_active = float(jnp.mean(jax.device_get(
        reports["adv_active_mean"]
    )))
    adv_inactive = float(jnp.mean(jax.device_get(
        reports["adv_inactive_mean"]
    )))
    reward_mean = float(jnp.mean(jax.device_get(
        reports["reward_mean"]
    )))
    reward_active = float(jnp.mean(jax.device_get(
        reports["reward_active_mean"]
    )))
    terminal_steps = int(jnp.sum(jax.device_get(
        reports["terminal_steps"]
    )))
    terminal_envs = int(jnp.sum(jax.device_get(
        reports["terminal_envs"]
    )))
    print(
        f"epoch {epoch} loss_mean {loss:.6f} "
        f"done {done_count} draws {draw_count} "
        f"adv_active {adv_active:.4f} "
        f"adv_inactive {adv_inactive:.4f} "
        f"reward {reward_mean:.4f} "
        f"reward_active {reward_active:.4f} "
        f"terminal_steps {terminal_steps} "
        f"terminal_envs {terminal_envs}"
    )


def _optimal_action(pile):
    mod = pile % 3
    take2 = mod == 2
    return jnp.where(take2, jnp.int32(1), jnp.int32(0))


def _eval_vs_optimal(
    key,
    policy,
    model_state,
    max_pile,
    games=128,
    model_player=0,
    deterministic=True,
):
    env = make_nim(max_pile=max_pile, min_pile=max_pile)
    vec_env = vectorize_aec(env, games)
    key, init_key = jrng.split(key)
    env_state, obs, _, _ = vec_env.init(init_key)

    def step_fn(carry, key_step):
        env_state, obs = carry
        key_a, env_key = jrng.split(key_step)
        obs_a = _obs_for_player(obs, 0)
        obs_b = _obs_for_player(obs, 1)

        if model_player == 0:
            logits, _, mask = policy.logits(obs_a, model_state)
            safe_logits = jnp.where(mask, logits, -1e9)
            if deterministic:
                action_a = jnp.argmax(safe_logits, axis=-1)
            else:
                dist = categorical(safe_logits)
                action_a = dist.sample(key_a)
            action_b = _optimal_action(obs_b["pile"])
        else:
            logits, _, mask = policy.logits(obs_b, model_state)
            safe_logits = jnp.where(mask, logits, -1e9)
            if deterministic:
                action_b = jnp.argmax(safe_logits, axis=-1)
            else:
                dist = categorical(safe_logits)
                action_b = dist.sample(key_a)
            action_a = _optimal_action(obs_a["pile"])
        action = jnp.stack([action_a, action_b], axis=-1)
        env_state, obs, _, done, reward = vec_env.step(
            env_key,
            env_state,
            action,
        )
        return (env_state, obs), (reward, done)

    key, rollout_key = jrng.split(key)
    keys = jrng.split(rollout_key, max_pile + 2)
    (env_state, obs), (rewards, dones) = jax.lax.scan(
        step_fn,
        (env_state, obs),
        keys,
    )
    done_any = jnp.any(dones, axis=0)[..., None]
    done_idx = jnp.argmax(dones, axis=0)
    final_reward = rewards[done_idx, jnp.arange(games)]
    final_reward = jnp.where(done_any, final_reward, 0.0)
    idx = 0 if model_player == 0 else 1
    wins = jnp.sum(final_reward[..., idx] > 0)
    losses = jnp.sum(final_reward[..., idx] < 0)
    draws = games - wins - losses
    win_rate = wins / games
    return float(win_rate), int(wins), int(losses), int(draws)


def run_training(
    seed=0,
    epochs=20,
    steps_per_epoch=1,
    parallel_envs=64,
    rollout_steps=64,
    max_pile=7,
    min_pile=1,
    eval_deterministic=True,
    output_directory="training_runs/nim",
):
    key = jrng.key(seed)

    ppo_params = PPOParams(
        parallel_envs=parallel_envs,
        rollout_steps=rollout_steps,
        training_epochs=2,
        minibatch_size=512,
    )

    env = make_nim(max_pile=max_pile, min_pile=min_pile)
    policy = make_nim_policy(max_pile=max_pile)
    optim = adam(learning_rate=3e-4)

    ppo = make_ppo_selfplay(ppo_params, env, policy, optim)
    epoch_system = make_epoch_system(
        ppo,
        steps_per_epoch=steps_per_epoch,
        make_report=_make_report,
        log=_log,
        output_directory=output_directory,
        save_states=1,
        save_reports=1,
        verbose=True,
    )

    key, init_key = jrng.split(key)
    state = epoch_system.init(init_key)
    for _ in range(epochs):
        key, step_key = jrng.split(key)
        state = epoch_system.step(step_key, state)
        eval_key, key = jrng.split(key)
        win_a, wins_a, losses_a, draws_a = _eval_vs_optimal(
            eval_key,
            policy,
            state.system_state.model_state,
            max_pile,
            model_player=0,
            deterministic=eval_deterministic,
        )
        eval_key, subkey = jrng.split(eval_key)
        win_b, wins_b, losses_b, draws_b = _eval_vs_optimal(
            subkey,
            policy,
            state.system_state.model_state,
            max_pile,
            model_player=1,
            deterministic=eval_deterministic,
        )
        wins = wins_a + wins_b
        losses = losses_a + losses_b
        draws = draws_a + draws_b
        total = max(1, wins + losses + draws)
        win_rate = wins / total
        print(
            f"eval win_rate {win_rate:.3f} "
            f"wins {wins} losses {losses} draws {draws}"
        )
    return state


if __name__ == "__main__":
    run_training(
        epochs=100,
        steps_per_epoch=4,
        rollout_steps=128,
        max_pile=7,
        eval_deterministic=True,
    )
