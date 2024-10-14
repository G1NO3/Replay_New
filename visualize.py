import argparse
from functools import partial
import jax
from jax import numpy as jnp
import optax
from flax import struct  # Flax dataclasses
from clu import metrics
from tensorboardX import SummaryWriter
from flax.traverse_util import flatten_dict, unflatten_dict
import env
from agent import Encoder, Hippo, Policy
from flax.training import train_state, checkpoints
import path_int
import buffer
import os
import flax.linen as nn
# import cal_plot
import pickle
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import train

def render(ax, goal_s, wall_maze, s, grid_size):
    ax.grid(visible=True)
    ax.set_xlim(0,grid_size)
    ax.set_ylim(grid_size,0)
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


def eval_video(args):
    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key)
    env_state, buffer_state, encoder_state, hippo_state, policy_state = train.init_states(args, subkey)
    writer = SummaryWriter(f"./train_logs/{args.prefix}")
    # Initialize actions, hippo_hidden, and theta ==================
    s = env_state['start_s']
    a = jax.random.randint(subkey, (args.n_agents, 1), 0, args.n_action)
    hippo_hidden = jnp.zeros((args.n_agents, args.hippo_hidden_size))
    theta = jnp.zeros((args.n_agents, args.theta_hidden_size))
    # pc_centers = jnp.array([[i, j] for i in range(args.grid_size) for j in range(args.grid_size)])
    grid_col, grid_row = jnp.meshgrid(jnp.arange(args.grid_size), jnp.arange(args.grid_size), indexing='ij')
    pc_centers = jnp.concatenate((grid_row.reshape(-1, 1), grid_col.reshape(-1, 1)), axis=-1)
    
    if args.model_path[2:] not in os.listdir():
        os.mkdir(args.model_path)
    for ei in range(args.epochs):
        # walk in the env and update buffer (model_step)
        key, subkey = jax.random.split(subkey)
        env_state, hippo_hidden, theta, next_s, next_a, rewards, done, replayed_history \
            = train.eval_step((env_state, encoder_state, hippo_state, policy_state),
                         subkey, s, a, hippo_hidden, theta, temperature=1, replay_steps=args.replay_steps)
        # jax.debug.print('ei:{a},n:{b},rewards:{c},hippo:{d},theta:{e}',
        #                 a=ei, b=2, c=rewards[2], d=hippo_hidden[2], e=theta[2])
        s = next_s
        a = next_a


        

if __name__ == '__main__':
    pass