import jax.random as jrng

from dirt.examples.nom import NomParams, NomAction, nom

from mechagogue.rl.vpg import VPGConfig, vpg

def main(params=NomParams()):
    key = jrng.key(1234)
    train_params = VPGConfig(
        parallel_envs=2,
        rollout_steps=128,
    )
    env_params = NomParams(
        mean_initial_food = 0.,
        mean_food_growth = 0.,
    )
    reset, step = nom(env_params)
    
    key, weight_key, train_key = jrng.split(key, 3)
    
    def policy(weights, obs):
        def action_sampler(key):
            b = obs.view.shape[0]
            forward_key, turn_key = jrng.split(key)
            forward = jrng.randint(forward_key, b, 0, 2)
            turn = jrng.randint(turn_key, b, -1, 2)
            return NomAction(forward, turn)
        
        def action_logp(action):
            return jnp.log(2) + jnp.log(3)
        
        return action_sampler, action_logp
    
    def init_params(key):
        return None
    
    def train_params(key, params, grad):
        return None
    
    #state, obs = reset(key)
    #action = NomAction(forward=True, rotate=0)
    #state, obs, reward, done = step(key, state, action)
    
    vpg_reset, vpg_step = vpg(
        train_key,
        train_params,
        reset,
        step,
        policy,
        init_params,
        train_params,
    )

if __name__ == '__main__':
    main()
