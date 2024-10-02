import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib as mpl
from functools import partial
from agent import Encoder
import path_int
import optax
from agent import Hippo, Policy
import train
import os
from flax.training import checkpoints
import seaborn as sns
import pandas as pd
import scipy
@partial(jax.jit, static_argnums=(1,2))
def calculate_replay(replay_traj, width, height, ck0_x, ck0_y, ck1_x, ck1_y):
    replay_path = jnp.stack([replay_traj//width, replay_traj%width], axis=-1)
    # jax.debug.print(f'replay_path:{replay_path}')
    directionality, forward_degree = calculate_path_criterion(replay_path)
    score, max_segment = calculate_segment(replay_path, width, height, ck0_x, ck0_y, ck1_x, ck1_y)
    # jax.debug.print(f'directionality:{directionality}, forward_degree:{forward_degree}, score:{score}, max_segment:{max_segment}')
    return directionality, forward_degree, score, max_segment

@jax.jit
def calculate_path_criterion(replay_path):
    # [replay_step, 2]
    total_length = jnp.sum(jnp.sqrt(jnp.square(jnp.diff(replay_path[:,0], axis=0))+jnp.square(jnp.diff(replay_path[:,1], axis=0))),axis=0)
    entire_vector = replay_path[-1] - replay_path[0]
    single_vectors = jnp.diff(replay_path, axis=0)
    cos_sim_with_entire = jnp.dot(entire_vector, jnp.transpose(single_vectors))/(jnp.linalg.norm(entire_vector,axis=-1)*jnp.linalg.norm(single_vectors,axis=-1))
    directionality = jnp.nanmean(cos_sim_with_entire)
    # print(single_vectors)
    # print(single_vectors[:-1]*single_vectors[1:])
    # cos_sim_between = jnp.sum(single_vectors[:-1]*single_vectors[1:],axis=-1)/(jnp.linalg.norm(single_vectors[:-1],axis=-1)*jnp.linalg.norm(single_vectors[1:],axis=-1))
    # sequentiality = jnp.nanmean(cos_sim_between)
    forward_degree = jnp.sum(entire_vector)/total_length

    directionality = jnp.where(jnp.isnan(directionality), 0, directionality)
    # sequentiality = jnp.where(jnp.isnan(sequentiality), 0, sequentiality)
    forward_degree = jnp.where(jnp.isnan(forward_degree), 0, forward_degree)

    return directionality, forward_degree

@partial(jax.jit, static_argnums=(1,2))
def calculate_segment(replay_traj, width, height, ck0_x, ck0_y, ck1_x, ck1_y):
    grid_s_c1, grid_s_c2, grid_c1_g, grid_c2_g = \
        jnp.zeros((width, height)), jnp.zeros((width, height)), jnp.zeros((width, height)), jnp.zeros((width, height))
    v1=0.5
    ck0_x_g_lb = jnp.maximum(0, ck0_x-1)
    ck0_y_g_lb = jnp.maximum(0, ck0_y-1)
    ck1_x_g_lb = jnp.maximum(0, ck1_x-1)
    ck1_y_g_lb = jnp.maximum(0, ck1_y-1)
    grid_s_c1 = grid_s_c1.at[:ck0_x+1, :ck0_y+1].set(1)
    grid_s_c1 = grid_s_c1.at[ck0_x+1, :ck0_y+2].set(v1)
    grid_s_c1 = grid_s_c1.at[:ck0_x+2, ck0_y+1].set(v1)
    # grid_s_c1 = grid_s_c1.at[ck0_x+2, :ck0_y+3].set(v2)
    # grid_s_c1 = grid_s_c1.at[:ck0_x+3, ck0_y+2].set(v2)

    grid_s_c2 = grid_s_c2.at[:ck1_x+1, :ck1_y+1].set(1)
    grid_s_c2 = grid_s_c2.at[ck1_x+1, :ck1_y+2].set(v1)
    grid_s_c2 = grid_s_c2.at[:ck1_x+2, ck1_y+1].set(v1)
    # grid_s_c2 = grid_s_c2.at[ck1_x+2, :ck1_y+3].set(v2)
    # grid_s_c2 = grid_s_c2.at[:ck1_x+3, ck1_y+2].set(v2)

    grid_c1_g = grid_c1_g.at[ck0_x_g_lb, ck0_y_g_lb:].set(v1)
    grid_c1_g = grid_c1_g.at[ck0_x_g_lb:, ck0_y_g_lb].set(v1)
    grid_c1_g = grid_c1_g.at[ck0_x:, ck0_y:].set(1)
    # grid_c1_g = grid_c1_g.at[ck0_x-2, ck0_y-2:].set(v2)
    # grid_c1_g = grid_c1_g.at[ck0_x-2:, ck0_y-2].set(v2)


    grid_c2_g = grid_c2_g.at[ck1_x_g_lb, ck1_y_g_lb:].set(v1)
    grid_c2_g = grid_c2_g.at[ck1_x_g_lb:, ck1_y_g_lb].set(v1)
    grid_c2_g = grid_c2_g.at[ck1_x:, ck1_y:].set(1)

    print(grid_s_c1, grid_s_c2, grid_c1_g, grid_c2_g)
    grid_all = jnp.stack([grid_s_c1, grid_s_c2, grid_c1_g, grid_c2_g], axis=0)
    pos = jnp.zeros((width, height))
    pos = pos.at[(replay_traj[:,0],replay_traj[:,1])].set(1)
    score = jnp.sum(grid_all*pos, axis=(1,2))/jnp.sum(grid_all*pos, axis=(0,1,2))
    score = jnp.where(jnp.isnan(score), 0, score)
    max_segment = jnp.argmax(score, axis=0)
    return score, max_segment

def plot_statistics_replay(axes, directionality, forward_degree, score, max_segment, row_text, set_title=False):
    significant = jnp.where((jnp.max(score, axis=-1) > 0.4) & (directionality > 0.4), 1, 0)
    s_c1_f = jnp.sum(jnp.where(significant & (max_segment==0) & (forward_degree>0), 1, 0))
    s_c1_r = jnp.sum(jnp.where(significant & (max_segment==0) & (forward_degree<0), 1, 0))
    s_c2_f = jnp.sum(jnp.where(significant & (max_segment==1) & (forward_degree>0), 1, 0))
    s_c2_r = jnp.sum(jnp.where(significant & (max_segment==1) & (forward_degree<0), 1, 0))
    c1_g_f = jnp.sum(jnp.where(significant & (max_segment==2) & (forward_degree>0), 1, 0))
    c1_g_r = jnp.sum(jnp.where(significant & (max_segment==2) & (forward_degree<0), 1, 0))
    c2_g_f = jnp.sum(jnp.where(significant & (max_segment==3) & (forward_degree>0), 1, 0))
    c2_g_r = jnp.sum(jnp.where(significant & (max_segment==3) & (forward_degree<0), 1, 0))
    axes[0].text(0.5, 0.5, row_text, horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
    axes[0].hist(directionality)
    axes[1].hist(forward_degree)
    axes[2].hist(jnp.max(score, axis=-1))
    data = [s_c1_f, s_c1_r, s_c2_f, s_c2_r, c1_g_f, c1_g_r, c2_g_f, c2_g_r]
    data = [0 if jnp.isnan(x) else x for x in data]
    data = [x/sum(data) for x in data]
    labels = ['s1f', 's1r', 's2f', 's2r', '1gf', '1gr', '2gf', '2gr']
    axes[3].bar(labels, data, color=['darkred', 'red', 'darkorange', 'orange', 'darkblue', 'blue', 'darkgreen', 'green'])
    axes[3].set_ylim([0,1])
    if set_title:
        axes[0].set_title('directionality')
        axes[1].set_title('forward_degree')
        axes[2].set_title('max_score')
        axes[3].set_title('segment_proportion')
    return
import pickle
def draw_backward_proportion():
    prefix = 'nb_nstd_mem5'
    forward_backward_proportion = pickle.load(open('./figures/'+prefix+'/forward_backward_proportion.pkl', 'rb'))
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    xs = jnp.arange(4)
    for ax in axes:
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.set_xticks(xs, ['0','1','2','3'])
        ax.tick_params(axis='both', which='major', labelsize=40, width=3, length=10)
    effective_proportion = forward_backward_proportion[:,0]
    forward_porportion = forward_backward_proportion[:,1]
    backward_proportion = forward_backward_proportion[:,2]
    
    axes[0].plot(xs, effective_proportion, label='effective proportion', linewidth=4)
    axes[1].plot(xs, backward_proportion, label='backward proportion', linewidth=4)

    # axes[0].legend(loc='upper right', shadow=True, fontsize='x-large')
    # axes[1].legend(loc='upper right', shadow=True, fontsize='x-large')
    fig.subplots_adjust(wspace=0.5, bottom=0.35, top=0.9, left=0.2, right=0.9)
    plt.savefig('./figures/'+prefix+'/forward_backward_proportion.png')
    print('save forward backward proportion to ./figures/'+prefix+'/forward_backward_proportion.png')
    return

def draw_hipp_info_decoding_acc():
    num = [0.81195, 1.0]
    figure, axis = plt.subplots()
    axis.bar([0,1],num)
    axis.set_xticks([0,1], ['train','test'], rotation=0)
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['bottom'].set_linewidth(3)
    axis.spines['left'].set_linewidth(3)
    axis.tick_params(axis='both', which='major', labelsize=40, width=3, length=10)
    plt.subplots_adjust(bottom=0.35, top=0.9, left=0.2, right=0.9)
    plt.savefig('./figures/test.png')
    return

if __name__ == '__main__':

    # import pickle
    # prefix = 'nb_nstd_mem5'
    # line_chart_info = pickle.load(open('./figures/'+prefix+'/line_chart.pkl', 'rb'))
    # line_chart_mean = line_chart_info['line_chart_mean']
    # line_chart_std = line_chart_info['line_chart_std']
    # line_chart_shortcut_emerge_mean = line_chart_mean[1,1]
    # line_chart_shortcut_emerge_std = line_chart_std[1,1]
    # t_score = line_chart_shortcut_emerge_mean/(line_chart_shortcut_emerge_std/np.sqrt(10))
    # p = scipy.stats.t.sf(np.abs(t_score), 9)
    # print(t_score, p)
    #to test
    # draw_backward_proportion()
    # biology_line_chart_mean = jnp.load('./figures/biology_line_chart_mean.npy')
    # print(biology_line_chart_mean)
    # prefix = 'hml_926_noise_toHPC'
    # kl_mean, kl_std, step_count_mean, step_count_std = pickle.load(open('./figures/'+prefix+'/Ablation_KL_div.pkl','rb')).values()
    # print(step_count_mean, step_count_std)
    # draw_hipp_info_decoding_acc()
    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 6)
    replay_keys = jnp.concatenate((jnp.zeros_like(subkeys[:1]), subkeys[1:]),0)
    print(len(subkeys))
    print(replay_keys.dtype)
    for i in range(6):
        noise = jax.random.normal(replay_keys[i], (1,2))
        print(jnp.where((replay_keys[i]!=0).all(),noise,jnp.zeros_like(noise)))
    # print(jax.random.normal(replay_keys[0], (2,2)))