"""
Env, a 5*5 grid env
obs: (5, 5), 0 for pathway, 1 for obstacle, 2 for reward, 3 for self pos,
    (the obs is set to be zero with 0.8 prob in tasks now)
goal: fixed at (4, 4), when reached the goal, the agent will be send back to the start point
reward: a small reward random appear in the map. When got the reward,
    the reward will be send to another pos with 0.1 prob (call reset_reward)
action: 4 actions, up, down, left, right
"""
import jax
import jax.numpy as jnp
from functools import partial


# fixme: seldom obs, whether reset randomly,
#reset
# possible_reward = jnp.array([[0,1],[0,2],[0,3],[0,4],[1,4],[2,4],[3,4],
#                              [1,0],[2,0],[3,0],[4,0],[4,1],[4,2],[4,3]])
train_reward = jnp.array([[1,1],[1,2],[1,3],
                             [2,1],[2,2],[2,3],
                             [3,1],[3,2],[3,3]])
pseudo_reward_list = [jnp.array([[3,1],[1,3],
                             
                             [2,2],[2,3],
                             [3,2],[3,3]]),
                    jnp.array([[1,3],[3,1]])]
#pseudo_reset
ck0_x = pseudo_reward_list[1][0,0].item()
ck0_y = pseudo_reward_list[1][0,1].item()
ck1_x = pseudo_reward_list[1][1,0].item()
ck1_y = pseudo_reward_list[1][1,1].item()
ck0_x_g_lb = max(0, ck0_x-1)
ck0_y_g_lb = max(0, ck0_y-1)
ck1_x_g_lb = max(0, ck1_x-1)
ck1_y_g_lb = max(0, ck1_y-1)
####WARNING: no obstacle now
@partial(jax.jit, static_argnums=(3))
def add_obstacle(grid, obstx, obsty, n_agents):
    # add obstacle for grid [n, h, w];
    # obstacles: n * n_obstacles * [2,]
    grid = grid.at[jnp.arange(n_agents).reshape(-1,1,1),obstx,obsty].set(0)
    return grid


@partial(jax.jit, static_argnums=(2, 3, 4))
def init_reward(grid, key, n_agents, height, width):
    key, subkey = jax.random.split(key, 2)
    reward_idx = jax.random.randint(subkey, (n_agents,), minval=0, maxval=train_reward.shape[0])
    # print(reward_idx)
    reward_center = train_reward.at[reward_idx].get()
    grid = add_reward(grid, reward_center)
    return grid, reward_center, reward_idx.reshape(-1,1)

@partial(jax.jit, static_argnums=(2, 3, 4))
def pseudo_init_reward(grid, key, n_agents, height, width, pseudo_reward):
    key, subkey = jax.random.split(key, 2)
    reward_idx = jax.random.randint(subkey, (n_agents,), minval=0, maxval=pseudo_reward.shape[0])
    # print(reward_idx)
    reward_center = pseudo_reward.at[reward_idx].get()
    grid = add_reward(grid, reward_center)
    return grid, reward_center, reward_idx.reshape(-1,1)

@jax.jit
def add_reward(grid, reward_center):
    # fixme: one reward for each env
    def set_r(gd, pos):
        # gd[10, 10], pos[2,]
        return gd.at[pos[0], pos[1]].set(2)
    grid = jax.vmap(set_r, (0, 0), 0)(grid, reward_center)  # todo
    # grid = jax.vmap(set_r, (0, 0), 0)(grid, reward_center+jnp.array([1,0]))
    # grid = jax.vmap(set_r, (0, 0), 0)(grid, reward_center-jnp.array([1,0]))
    # grid = jax.vmap(set_r, (0, 0), 0)(grid, reward_center+jnp.array([0,1]))
    # grid = jax.vmap(set_r, (0, 0), 0)(grid, reward_center-jnp.array([0,1]))
    return grid


@jax.jit
def fetch_pos(grid, pos):
    # grid[h, w], pos[2,]
    return grid[pos[0], pos[1]]


@jax.jit
def set_pos(grid, pos, value):
    # grid[h, w], pos[2,]
    return grid.at[pos[0], pos[1]].set(value)


@jax.jit
def prepare_obs(grid, current_pos):
    # obs: can see self.current pos, but cannot see rewards
    obs = jax.vmap(set_pos, (0, 0, None), 0)(grid, current_pos, 3)
    # obs = jnp.where(obs == 2, 0, obs)
    return obs


@jax.jit
def take_action(actions, current_pos, grid, goal_pos, checked, step_count, move_to_start):
    next_pos = jnp.where(actions == 0, current_pos - jnp.array([1, 0], dtype=jnp.int8),
                         jnp.where(actions == 1, current_pos + jnp.array([0, 1], dtype=jnp.int8),
                                   jnp.where(actions == 2, current_pos + jnp.array([1, 0], dtype=jnp.int8),
                                             jnp.where(actions == 3, current_pos - jnp.array([0, 1], dtype=jnp.int8),
                                                       current_pos))))
    # next_pos [n, 2]
    next_pos = jnp.clip(next_pos, 0, jnp.array([grid.shape[1] - 1, grid.shape[2] - 1], dtype=jnp.int8))
    # knocked_wall = jnp.where((next_pos == current_pos).all(axis=-1), -0.4, 0).reshape(-1,1)
    hit = jax.vmap(fetch_pos, (0, 0), 0)(grid, next_pos)
    hit = hit.reshape((-1, 1))
    # blocked = jnp.where(hit == 1, 1, 0)
    next_pos = jnp.where(hit == 1, current_pos, next_pos)
    # rewarded = jnp.where((not checked) & hit == 2, 0.5, 0)
    # jax.debug.breakpoint()
    rewarded = jnp.where(hit == 2, jnp.where(checked, 0, 0.5), 0)
    
    # step_punishment = -jnp.ones((actions.shape[0],1))*0.3
    out_of_time = jnp.where(step_count > 100, 1, 0) 
    # goal_start_reward = jnp.where(jnp.all(next_pos == goal_pos, axis=1, keepdims=True) & checked, 1, 0) | \
    #                         move_to_start
    goal_start_reward = jnp.where(jnp.all(next_pos == goal_pos, axis=1, keepdims=True) & checked, 1, 0)
    rewards = goal_start_reward + rewarded\
                # + out_of_time_punishment# + step_punishment
    done = (jnp.all(next_pos == goal_pos, axis=1, keepdims=True) & checked) | out_of_time
    step_count = jnp.where(done, 0, step_count)
    checked = jnp.where(hit == 2, 1, checked)
    return next_pos, rewards, done, checked, step_count


def reset(width, height, n_agents, key):
    grid = jnp.zeros((n_agents, height, width), dtype=jnp.int8)
    obstx, obsty = jnp.meshgrid(jnp.arange(1, height - 1), jnp.arange(1, width - 1))
    grid = add_obstacle(grid, obstx, obsty, n_agents)
    # fixme: no obstacles now; so add_obstacle is not checked
    # fixme: magic number: 0 for pathway, 1 for obstacle, 2 for reward, 3 for self pos
    grid, reward_center, reward_idx = init_reward(grid, key, n_agents, height, width)
    start_pos = jnp.zeros((n_agents, 2), dtype=jnp.int8)
    goal_pos = jnp.array([[height - 1, width - 1]] * n_agents).astype(jnp.int8)
    current_pos = start_pos
    checked = jnp.zeros((n_agents, 1), dtype=jnp.int8)
    step_count = jnp.zeros((n_agents, 1), dtype=jnp.int8)
    move_to_start = jnp.zeros((n_agents, 1), dtype=jnp.int8)
    return prepare_obs(grid, current_pos), {'grid': grid, 'current_pos': current_pos,
                                            'goal_pos': goal_pos, 'reward_center':reward_center,
                                            'checked': checked, 'step_count':step_count,
                                            'move_to_start':move_to_start}

def pseudo_reset(width, height, n_agents, key, pseudo_reward):
    grid = jnp.zeros((n_agents, height, width), dtype=jnp.int8)
    obstx, obsty = jnp.meshgrid(jnp.arange(1, height - 1), jnp.arange(1, width - 1))
    grid = add_obstacle(grid, obstx, obsty, n_agents)
    # fixme: no obstacles now; so add_obstacle is not checked
    # fixme: magic number: 0 for pathway, 1 for obstacle, 2 for reward, 3 for self pos
    # reward_center_c0 = jnp.repeat(jnp.array([[ck0_x,ck0_y]],dtype=jnp.int8), n_agents, axis=0)
    # reward_center_c1 = jnp.repeat(jnp.array([[ck1_x,ck1_y]],dtype=jnp.int8), n_agents, axis=0)
    # phase = jax.random.randint(key, shape=(n_agents,1), minval=0, maxval=2, dtype=jnp.int8)
    # #####FIXME: phase is not randomly initialized
    # reward_center = jnp.where(phase==0, reward_center_c0, reward_center_c1)
    # grid = add_reward(grid, reward_center)

    grid, reward_center, reward_idx = pseudo_init_reward(grid, key, n_agents, height, width, pseudo_reward)
    start_pos = jnp.zeros((n_agents, 2), dtype=jnp.int8)
    goal_pos = jnp.array([[height - 1, width - 1]] * n_agents).astype(jnp.int8)
    current_pos = start_pos
    checked = jnp.zeros((n_agents, 1), dtype=jnp.int8)
    step_count = jnp.zeros((n_agents, 1), dtype=jnp.int8)
    move_to_start = jnp.zeros((n_agents, 1), dtype=jnp.int8)

    return prepare_obs(grid, current_pos), {'grid': grid, 'current_pos': current_pos,
                                            'goal_pos': goal_pos, 'reward_center':reward_center,
                                            'checked': checked, 'step_count':step_count,
                                            'phase': reward_idx, 'move_to_start':move_to_start}

def Igata_reset(width, height, n_agents, key, pseudo_reward):
    grid = jnp.zeros((n_agents, height, width), dtype=jnp.int8)
    # obstx, obsty = jnp.meshgrid(jnp.arange(1, height - 1), jnp.arange(1, width - 1))
    # grid = add_obstacle(grid, obstx, obsty, n_agents)
    reward_idx = jnp.zeros((n_agents, ), dtype=jnp.int8)
    # print(reward_idx)
    reward_center = pseudo_reward.at[reward_idx].get()
    grid = add_reward(grid, reward_center)
    phase = reward_idx.reshape(-1,1)

    start_pos = jnp.zeros((n_agents, 2), dtype=jnp.int8)
    goal_pos = jnp.array([[height - 1, width - 1]] * n_agents).astype(jnp.int8)
    current_pos = start_pos
    checked = jnp.zeros((n_agents, 1), dtype=jnp.int8)
    step_count = jnp.zeros((n_agents, 1), dtype=jnp.int8)
    move_to_start = jnp.zeros((n_agents, 1), dtype=jnp.int8)
    checked_times = jnp.zeros((n_agents, 1), dtype=jnp.int8)
    total_checked = jnp.zeros((n_agents, 1), dtype=jnp.int8)
    return prepare_obs(grid, current_pos), {'grid': grid, 'current_pos': current_pos,
                                            'goal_pos': goal_pos, 'reward_center':reward_center,
                                            'checked': checked, 'step_count':step_count,
                                            'phase': phase, 'move_to_start':move_to_start,
                                            'checked_times': checked_times, 'total_checked': total_checked}


@jax.jit
def step(env_state, actions):
    env_state['step_count'] = env_state['step_count'].at[:].add(1)
    next_pos, rewards, done, checked, step_count = take_action(actions,
                                                   env_state['current_pos'], env_state['grid'],
                                                   env_state['goal_pos'], env_state['checked'],
                                                   env_state['step_count'], env_state['move_to_start'])
                                                
    # current_pos = jnp.where(blocked == -1, env_state['current_pos'], next_pos)
    current_pos = jnp.where(done, jnp.zeros_like(next_pos), next_pos)
    # current_pos = jnp.where(env_state['move_to_start'], jnp.zeros_like(next_pos), next_pos)
    # env_state['move_to_start'] = jnp.where(done, 1, 0)
    # fixme: reset pos to zero(start point) as soon as goal is reached
    # if checked, hide the reward; if done, show the reward
    # if rewarded, set checked to 1; if done, set checked to 0
    # only after checked can it reach the goal
    # grid = jnp.where(checked.reshape(-1,1,1), 0, env_state['grid'])
    # grid = jnp.where(done.reshape(-1,1,1), add_reward(grid, env_state['reward_center']), grid)
    checked = jnp.where(done, 0, checked)
    env_state = dict(env_state, current_pos=current_pos, checked=checked, step_count=step_count)
    obs = prepare_obs(env_state['grid'], current_pos)
    
    return obs, rewards, done, env_state


@jax.jit
def reset_reward(env_state, rewards, key):
    key, subkey = jax.random.split(key)
    reset_flag = jax.random.uniform(subkey, (rewards.shape[0], 1, 1)) < 0.1
    # print(jax.random.uniform(subkey, (rewards.shape[0], 1, 1)))
    new_grid = jnp.where(jnp.isclose(rewards.reshape((-1, 1, 1)), 0.5) & reset_flag & (env_state['grid']==2), 0, env_state['grid'])
    # fixme: if reward, set grid to 0 (no obstacles)
    key, subkey = jax.random.split(key)
    new_grid, new_center, new_idx = init_reward(new_grid, subkey, *env_state['grid'].shape)
    new_grid = jnp.where(jnp.isclose(rewards.reshape((-1, 1, 1)), 0.5) & reset_flag, new_grid, env_state['grid'])
    new_center = jnp.where(jnp.isclose(rewards.reshape((-1, 1)), 0.5) & reset_flag.reshape(-1,1), new_center, env_state['reward_center'])
    env_state = dict(env_state, grid=new_grid, reward_center=new_center)
    # g = env_state['grid']
    # r = reset_flag
    # jax.debug.breakpoint()
    return env_state
@partial(jax.jit, static_argnums=(3,))
def pseudo_reset_reward(env_state, rewards, key, n_agents, reset_prob, pseudo_reward):
    # ck_0 = jnp.repeat(jnp.array([[ck0_x,ck0_y]],dtype=jnp.int8), n_agents, axis=0)
    # ck_1 = jnp.repeat(jnp.array([[ck1_x,ck1_y]],dtype=jnp.int8), n_agents, axis=0)

    # new_grid = jnp.where(env_state['grid']==2, 0, env_state['grid'])
    # new_center = jnp.where(env_state['phase']==1, ck_0, ck_1)
    # new_grid = add_reward(new_grid, new_center)
    
    key, subkey = jax.random.split(key)
    reset_flag = jax.random.uniform(subkey, (rewards.shape[0], 1, 1)) < reset_prob
    # jax.debug.print('reset_flag={a}', a=reset_flag)
    new_grid = jnp.where(jnp.isclose(rewards.reshape((-1, 1, 1)), 0.5) & reset_flag & (env_state['grid']==2), 0, env_state['grid'])
    key, subkey = jax.random.split(key)
    new_grid, new_center, new_idx = pseudo_init_reward(new_grid, subkey, *env_state['grid'].shape, pseudo_reward)
    
    new_grid = jnp.where(jnp.isclose(rewards.reshape((-1, 1, 1)), 0.5) & reset_flag, new_grid, env_state['grid'])
    new_center = jnp.where(jnp.isclose(rewards.reshape((-1, 1)), 0.5) & reset_flag.reshape(-1,1), new_center, env_state['reward_center'])
    new_phase = jnp.where(jnp.isclose(rewards.reshape((-1, 1)), 0.5) & reset_flag.reshape(-1,1), new_idx, env_state['phase'])
    # phase = jnp.where(jnp.isclose(rewards.reshape((-1, 1)), 0.5) & reset_flag.reshape(-1,1), 1 - env_state['phase'], env_state['phase'])
    # jax.debug.print('reset_flag:{a}, phase:{b}',a=reset_flag, b=phase)
    env_state = dict(env_state, grid=new_grid, reward_center=new_center, phase=new_phase)
    return env_state

def Igata_reset_reward(env_state, rewards, key, n_agents, pseudo_reward):
    key, subkey = jax.random.split(key)
    env_state['checked_times'] = jnp.where(jnp.isclose(rewards.reshape((-1, 1)), 0.5), env_state['checked_times'] + 1, env_state['checked_times'])
    reset_flag = jnp.where((env_state['checked_times'] > 4)&(env_state['phase']==0), 1, 0)
    # print(reset_flag.shape, env_state['grid'].shape, env_state['phase'].shape, env_state['reward_center'].shape)
    new_grid = jnp.where(env_state['grid']==2,
                 0, env_state['grid'])
    new_idx = 1 - env_state['phase']
    new_center = pseudo_reward.at[new_idx.reshape(-1)].get()
    # print(new_center.shape)
    new_grid = add_reward(new_grid, new_center)

    new_grid = jnp.where(jnp.isclose(rewards.reshape((-1, 1, 1)), 0.5) & (reset_flag.reshape(-1,1,1)),
                    new_grid, env_state['grid'])
    new_center = jnp.where(jnp.isclose(rewards.reshape((-1, 1)), 0.5) & (reset_flag.reshape(-1,1)),
                    new_center, env_state['reward_center'])
    new_phase = jnp.where(jnp.isclose(rewards.reshape((-1, 1)), 0.5) & (reset_flag.reshape(-1,1)),
                    new_idx, env_state['phase'])
    new_checked_times = jnp.where(jnp.isclose(rewards.reshape((-1, 1)), 0.5) & (reset_flag.reshape(-1,1)),
                    jnp.zeros_like(env_state['checked_times']), env_state['checked_times'])
    total_checked = jnp.where(jnp.isclose(rewards.reshape((-1, 1)), 0.5),
                    env_state['total_checked'] + 1, env_state['total_checked'])
    env_state = dict(env_state, grid=new_grid, reward_center=new_center, phase=new_phase, 
                    checked_times=new_checked_times, total_checked=total_checked)
    return env_state


##########debug
##########
if __name__ == '__main__':

    key = jax.random.PRNGKey(1)
    obs, env_state = Igata_reset(5, 5, 1, key, pseudo_reward_list[1])
    env_state['checked_times'] = jnp.zeros_like(env_state['checked'])
    print(env_state)
    print(obs, )
    while True:
        key, subkey = jax.random.split(key)

        actions = jnp.array(int(input('input action:')), dtype=jnp.int8).reshape(1,1)
        obs, rewards, done, env_state = step(env_state, actions)
        env_state = Igata_reset_reward(env_state, rewards, subkey, 1, pseudo_reward_list[1])
        
        print(prepare_obs(env_state['grid'], env_state['current_pos']), 
                rewards, done, env_state['step_count'], env_state['checked'],
                env_state['checked_times'])

    # for i in range(4):
    #     actions = jnp.ones((2, 1), dtype=jnp.int8) * 3
    #     obs, rewards, done, env_state = step(env_state, actions)
    #     print(obs, rewards, done, env_state)
    # for i in range(4):
    #     actions = jnp.ones((2, 1), dtype=jnp.int8) * 1
    #     obs, rewards, done, env_state = step(env_state, actions)
    #     print(obs, rewards, done, env_state)
    # for i in range(3):
    #     actions = jnp.ones((2, 1), dtype=jnp.int8) * 2
    #     obs, rewards, done, env_state = step(env_state, actions)
    #     print(obs, rewards, done, env_state)

