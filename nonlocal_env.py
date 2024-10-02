"""

Env using paradigm from "Experience replay is associated with 
efficient nonlocal learning". Six different route leads to two different goals, with three of which 
corresponding to each goal.

"""

import jax
import jax.numpy as jnp
from functools import partial

def reset(length, n_arm, n_agents, key):
    n_choice = 2
    n_obs = 8
    arm_unnorm_prob = jax.random.uniform(key, (n_arm, ))
    arm_prob = arm_unnorm_prob/jnp.sum(arm_unnorm_prob)
    key, subkey = jax.random.split(key)
    reward_prob = jax.random.uniform(subkey, (n_choice, ))
    reward_prob = jnp.clip(reward_prob, 0.25, 0.75)
    # [n_agents, n_arm]
    key, subkey = jax.random.split(key)
    current_arm = jax.random.choice(subkey, jnp.arange(n_arm, dtype=jnp.int8), 
                    shape=(n_agents, 1), p=arm_prob)
    key, subkey = jax.random.split(key)
    grid = jax.random.randint(subkey, (n_arm, length, n_choice+1, n_obs), 0, 2)
    choice = (-1) * jnp.ones((n_agents, 1), dtype=jnp.int8)
    grid = grid.at[:,:,-1,:].set(grid[:,0:1,-1,:])
    # [n_agents, 1]
    current_pos = jnp.concatenate((current_arm, jnp.zeros((n_agents, 1), dtype=jnp.int8), choice), axis=1)
    # [n_agents, 2]
    obs = grid.at[current_pos[:,0], current_pos[:,1], current_pos[:,2]].get()
    # [n_agents, 8]
    
    env_state = {'grid': grid, 'current_pos': current_pos,
                 'arm_prob': arm_prob, 'reward_prob': reward_prob}
    return obs, env_state

@jax.jit
def step(env_state, action, key):
    arm = env_state['current_pos'].at[:,0:1].get()
    loc = env_state['current_pos'].at[:,1:2].get()
    choice = env_state['current_pos'].at[:,2:3].get()
    new_choice = jnp.where(choice==-1, action, choice)

    current_pos = env_state['current_pos']
    n_arm, length, n_choice = env_state['grid'].shape[0:3]
    n_agents = current_pos.shape[0]
    done = jnp.where(current_pos.at[:,1:2].get() == length-1, 1, 0)
    key, subkey = jax.random.split(key)
    reward_prob = jax.vmap(lambda x: env_state['reward_prob'][x],0,0)(choice).reshape(-1,1)
    reward = jnp.where(jax.random.uniform(subkey, (n_agents, 1)) < reward_prob, 1, 0)
    new_choice = jnp.where(done==1, -1, new_choice)
    new_arm = jnp.where(done==1, jax.random.choice(subkey, jnp.arange(n_arm),
                    shape=(n_agents, 1), p=env_state['arm_prob']), arm)
    new_loc = jnp.where((done==1) | (choice==-1), jnp.zeros((n_agents, 1), dtype=jnp.int8), loc+1)
    # [n_agents, 1]
    new_pos = jnp.concatenate((new_arm, new_loc, new_choice), axis=1)
    env_state = dict(env_state, current_pos=new_pos)
    obs = env_state['grid'].at[new_pos[:,0], new_pos[:,1], new_pos[:,2]].get()
    return obs, reward, done, env_state

@jax.jit
def reset_reward(env_state, done, key, reset_prob):
    key, subkey = jax.random.split(key)
    reset_reward_prob = 0.025 * jax.random.normal(subkey, (env_state['reward_prob'].shape[0], )) + env_state['reward_prob']
    reset_reward_prob = jnp.clip(reset_reward_prob, 0.25, 0.75)
    key, subkey = jax.random.split(key)
    new_reward_prob = jnp.where((jax.random.uniform(subkey, (1, )) < reset_prob) & done.reshape(-1), reset_reward_prob, env_state['reward_prob'])
    env_state = dict(env_state, reward_prob=new_reward_prob)
    return env_state


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    obs, env_state = reset(3, 2, 1, key)
    print(env_state['current_pos'])
    while True:
        key, subkey = jax.random.split(key)

        actions = jnp.array(int(input('input action:')), dtype=jnp.int8).reshape(1,1)
        obs, rewards, done, env_state = step(env_state, actions, subkey)
        env_state = reset_reward(env_state, done, subkey, 0.1)
        print(env_state['current_pos'])
        print(env_state['reward_prob'])

