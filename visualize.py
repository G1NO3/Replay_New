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


def parse_args():
    # args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--hippo_lr', type=float, default=1e-4)
    parser.add_argument('--policy_lr', type=float, default=1e-4)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--train_every', type=int, default=400)
    parser.add_argument('--n_agents', type=int, default=32)
    parser.add_argument('--max_size', type=int, default=10000)  # max_size of buffer
    parser.add_argument('--sample_len', type=int, default=64)  # sample len from buffer: at most max_size - 1
    parser.add_argument('--epochs', type=int, default=int(1e6))

    parser.add_argument('--replay_steps', type=int, default=4)  # todo: tune

    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=1e-2)
    parser.add_argument('--n_train_time', type=int, default=6)
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--n_eval_steps', type=int, default=200)
    # params that should be the same with config.py
    parser.add_argument('--hippo_hidden_size', type=int, default=64)
    parser.add_argument('--theta_hidden_size', type=int, default=32)
    parser.add_argument('--grid_size', type=int, default=4)
    parser.add_argument('--n_action', type=int, default=4)
    # parser.add_argument('--encoder_prefix', type=str, default=None)  # todo: checkpoint
    # parser.add_argument('--hippo_prefix', type=str)
    # parser.add_argument('--policy_prefix', type=str)

    parser.add_argument('--prefix', type=str, required=True)

    # Init params for hippo and policy
    parser.add_argument('--bottleneck_size', type=int, default=4)
    parser.add_argument('--policy_scan_len', type=int, default=20)
    parser.add_argument('--hippo_mem_len', type=int, default=5)
    parser.add_argument('--hippo_pred_len', type=int, default=5)
    parser.add_argument('--pc_sigma', type=float, default=1)

    parser.add_argument('--eval_temperature', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

def render(ax, goal_s, wall_maze, s):
    grid_size = wall_maze.shape[-2]
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
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    model_path = f'./modelzoo/{args.prefix}/{args.seed}'
    env_state, buffer_state, encoder_state, hippo_state, policy_state = train.init_states(args, subkey, model_path)
    # Initialize actions, hippo_hidden, and theta ==================
    s = env_state['start_s']
    a = jax.random.randint(subkey, (args.n_agents, 1), 0, args.n_action)
    hippo_hidden = jnp.zeros((args.n_agents, args.hippo_hidden_size))
    theta = jnp.zeros((args.n_agents, args.theta_hidden_size))

    n_step = 30
    fig, axes = plt.subplots(n_step//10 * 2,10, figsize=(40,20))
    axes = axes.flatten()
    all_rewards = jnp.zeros((n_step,))
    for ei in range(n_step):
        # walk in the env and update buffer (model_step)
        key, subkey = jax.random.split(subkey)
        env_state, hippo_hidden, theta, next_s, next_a, rewards, done, hippo_output, replayed_history \
            = train.eval_step((env_state, encoder_state, hippo_state, policy_state),
                         subkey, s, a, hippo_hidden, theta, temperature=1, replay_steps=args.replay_steps)

        render(axes[2*ei], env_state['goal_s'][0], env_state['wall_maze'][0], next_s[0])
        pc_activation = hippo_output[0,:args.grid_size*args.grid_size]
        pc_activation = jax.nn.softmax(pc_activation).reshape(args.grid_size, args.grid_size)
        axes[2*ei+1].imshow(pc_activation)
        mem_reward = hippo_output[0, args.grid_size*args.grid_size:-1]
        pred_reward = jax.nn.sigmoid(hippo_output[0, -1])
        axes[2*ei+1].set_title(f'{pred_reward:.1f}')
        jax.debug.print('s:{a}, a:{b}, r:{c}, done:{d}, mem_r:{e}'.format(a=next_s[0], b=next_a[0], c=rewards[0], d=done[0], \
                        e=mem_reward))
        s = next_s
        a = next_a
        all_rewards = all_rewards.at[ei].set(rewards.mean().item())
    print(all_rewards.mean())
    [ax.axis('off') for ax in axes]
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig_path = f'figure/{args.prefix}/{args.seed}'
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(f'figure/{args.prefix}/{args.seed}/video.png')
    print(f'figure/{args.prefix}/{args.seed}/video.png')


        

if __name__ == '__main__':
    args = parse_args()
    eval_video(args)