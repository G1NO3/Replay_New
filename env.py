import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def step(key, env_state, s, a):
    wall_maze = env_state['wall_maze']
    reward_map = env_state['reward_map']
    goal_s = env_state['goal_s']
    def get_value3(grid, loc):
        return grid.at[loc[0], loc[1], loc[2]].get()
    # find if there is a wall in the direction of action
    actions = jnp.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=jnp.int8)
    neighbor = get_neighbor(s, actions[a.reshape(-1)], reward_map.shape[-1])
    assert neighbor.shape == s.shape
    iswall = jax.vmap(get_value3)(wall_maze, jnp.concatenate([s,a],-1))
    next_s = jnp.where(iswall.reshape(reward_map.shape[0],1), s, neighbor)
    assert next_s.shape == s.shape
    key, subkey = jax.random.split(key)
    # get reward and done
    reward = jax.vmap(get_value)(reward_map, next_s).reshape(reward_map.shape[0],1)
    env_state['done'] = jnp.where(jnp.all(next_s == goal_s, -1, keepdims=True), 1, 0)
    # if done, reset start
    start_s = regenerate_start(subkey, s, reward_map.shape[-1])
    next_s = jnp.where(env_state['done'], start_s, next_s)
    assert next_s.shape == s.shape
    # update start
    env_state['start_s'] = jnp.where(env_state['done'], start_s, env_state['start_s'])
    assert env_state['start_s'].shape == s.shape
    
    # if you want to get done and reset at the next step, swap the above two parts
    
    return next_s, reward, env_state['done'], env_state

def get_obs(env_state, s):
    grid = env_state['empty_grid']
    grid = jax.vmap(partial(set_value, value=1))(grid, s) # set current state to 1
    grid = jax.vmap(partial(set_value, value=2))(grid, env_state['start_s']) # set start state to 2
    grid = grid.reshape(grid.shape[0], grid.shape[1], grid.shape[2], 1)
    reward_map = env_state['reward_map'].reshape(env_state['reward_map'].shape[0], env_state['reward_map'].shape[1], env_state['reward_map'].shape[2], 1)
    obs = jnp.concatenate((env_state['wall_maze'], grid, reward_map), axis=-1)
    return obs

def get_value(reward_map, s):
    return reward_map.at[s[0],s[1]].get()

def set_value(grid, loc, value):
    return grid.at[loc[0], loc[1]].set(value)
# def update_value(grid, old_s, new_s, value):
#     grid = grid.at[old_s[0], old_s[1]].set(0)
#     grid = grid.at[new_s[0], new_s[1]].set(value)
#     # grid = grid.at[start_s[0], start_s[1]].set(1)
#     return grid

def regenerate_reward(key, reward_map, reward_location):
    new_reward_location = jax.random.randint(key, reward_location.shape, 0, reward_map.shape[-1]) # Regenerate reward location
    # new_reward_location = jnp.zeros((reward_location.shape[0], 2), dtype=jnp.int8)
    def set_value(grid, loc, value):
        return grid.at[loc[0], loc[1]].set(value)
    new_reward_map = jax.vmap(partial(set_value, value=0))(reward_map, reward_location)# Remove reward from old location
    new_reward_map = jax.vmap(partial(set_value, value=1))(new_reward_map, new_reward_location)# Add reward to new location
    return new_reward_map, new_reward_location

def regenerate_start(key, s, grid_size):
    new_start_location = jax.random.randint(key, s.shape, 0, grid_size) # Regenerate start location
    return new_start_location

def reset(key, grid_size, n_agents, env_state:dict):
    """Reset the environment"""
    """Return: start_location, env_state"""
    reward_map = jnp.zeros((n_agents, grid_size, grid_size), dtype=jnp.int8)
    goal_location = jnp.zeros((n_agents, 2), dtype=jnp.int8)
    start_location = jnp.zeros((n_agents, 2), dtype=jnp.int8)
    done = jnp.zeros((n_agents, 1), dtype=jnp.int8)
    env_state['done'] = done
    key, subkey = jax.random.split(key)
    reward_map, goal_location = regenerate_reward(subkey, reward_map, goal_location) # Regenerate reward location
    env_state['reward_map'] = reward_map
    env_state['goal_s'] = goal_location
    key, subkey = jax.random.split(key)
    new_start_location = regenerate_start(subkey, start_location, grid_size) # Regenerate start location
    env_state['start_s'] = new_start_location
    env_state['empty_grid'] = jnp.zeros((n_agents, grid_size, grid_size), dtype=jnp.int8) # Reset grid
    return new_start_location, env_state





def get_neighbor(s, a, grid_size):
    "s: (2,), a: (2,)"
    neigh = (s+a) % grid_size
    return neigh

def neighbors_ar(s, grid_size):
    actions = jnp.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=jnp.int8)
    return jax.vmap(get_neighbor, in_axes=(None, 0, None))(s, actions, grid_size)

def walk(key, maze, visited, start_s, grid_size):
    actions = jnp.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=jnp.int8)
    inverse_action_map = jnp.array([1,0,3,2], dtype=jnp.int8)
    neighbors = np.array([get_neighbor(start_s, a, grid_size) for a in actions])
    visited = visited.at[start_s[0], start_s[1]].set(1)
    key, subkey = jax.random.split(key)
    random_order = jax.random.permutation(subkey, len(neighbors))
    for i in random_order:
        neighbor = neighbors[i]
        if visited[neighbor[0], neighbor[1]] == 0:
            maze = maze.at[start_s[0], start_s[1], i].set(0) # remove the wall
            maze = maze.at[neighbor[0], neighbor[1], inverse_action_map[i]].set(0) # remove the inverse wall
            maze, visited = walk(subkey, maze, visited, neighbor, grid_size)
    return maze, visited

def get_maze(key, start_s, grid_size):
    actions = jnp.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=np.int8)
    inverse_action_map = jnp.array([1,0,3,2], dtype=np.int8)
    maze = np.ones((grid_size, grid_size, 4))
    visited = np.zeros((grid_size, grid_size), dtype=np.int8)
    key, subkey = jax.random.split(key)
    maze, visited = walk(subkey, maze, visited, start_s, grid_size)
    extra_remove_n = 3*(grid_size-3)
    wall_loc = np.where(maze == 1)
    key, subkey = jax.random.split(key)
    remove_index = jax.random.randint(subkey, (extra_remove_n,), 0, wall_loc[0].shape[0])
    for i in range(extra_remove_n):
        r, c, a = wall_loc[0][remove_index[i]], wall_loc[1][remove_index[i]], wall_loc[2][remove_index[i]]
        # print(r,c,a)
        neighbor = get_neighbor(np.array([r,c]), actions[a], grid_size)
        maze = maze.at[r, c, a].set(0)
        maze = maze.at[neighbor[0], neighbor[1], inverse_action_map[a]].set(0)
    return maze

def render(env_state, s, grid_size):
    fig, ax = plt.subplots(figsize=(3,3),frameon=True)
    ax.grid(visible=True)
    ax.set_xlim(0,grid_size)
    ax.set_ylim(grid_size,0)
    goal_s = env_state['goal_s']
    wall_maze = env_state['wall_maze']
    rect = mpl.patches.Rectangle((goal_s[1], goal_s[0]), width=1, height=1, facecolor='yellow')
    ax.add_patch(rect)
    circle = mpl.patches.Circle((s[1]+0.5, s[0]+0.5), radius=0.3, facecolor='red')
    ax.add_patch(circle)
    wall_loc = np.where(wall_maze==1)
    for i in range(len(wall_loc[0])):
        r, c, a = wall_loc[0][i], wall_loc[1][i], wall_loc[2][i] # already in matrix coordinates
        if a == 0:
            start_r = np.array([r+1, r+1])
            start_c = np.array([c, c+1])
        elif a == 1:
            start_r = np.array([r, r])
            start_c = np.array([c, c+1])
        elif a == 2:
            start_r = np.array([r, r+1])
            start_c = np.array([c+1, c+1])
        elif a == 3:
            start_r = np.array([r, r+1])
            start_c = np.array([c, c])
        line = mpl.lines.Line2D(start_c, start_r, color='blue', linewidth=5)
        ax.add_line(line)
    ax.xaxis.set_tick_params(bottom=False, top=True, labeltop=True, labelbottom=False)
    plt.show()



if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    grid_size = 4
    n_action = 4
    n_agents = 1
    maze_list = jnp.load('maze_list.npy')
    print('maze', maze_list.shape)
    wall_maze = jnp.repeat(maze_list[0].reshape(1, grid_size, grid_size, n_action), n_agents,
                           axis=0)
    key, subkey = jax.random.split(key)
    start_s, env_state = reset(subkey, grid_size, n_agents, env_state={'wall_maze': wall_maze})
    # print(start_location)
    print(env_state['goal_s'])
    print(env_state['reward_map'])
    # print(env_state['wall_maze'])
    s = env_state['start_s']
    screen, clock = None, None
    while True:
        a = int(input('Action:'))
        a = jnp.array([a for _ in range(n_agents)]).reshape(-1, 1)
        key, subkey = jax.random.split(key)
        next_s, reward, done, env_state = step(subkey, env_state, s, a)
        obs = get_obs(env_state, next_s)
        s = next_s
        print('s r done', next_s, reward, done)
        print('s', obs[:, :, :, -2])
        print('r', obs[:, :, :, -1])
        # render(env_state, next_s, grid_size)
        # print(next_s, reward, done)