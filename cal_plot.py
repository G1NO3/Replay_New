import argparse
from functools import partial
import jax
from jax import numpy as jnp
import optax
from flax import struct  # Flax dataclasses
from clu import metrics
from tensorboardX import SummaryWriter

import seaborn as sns
import env
from agent import Encoder, Hippo, Policy
from flax.training import train_state, checkpoints
import path_int
import buffer
import os 
import matplotlib.pyplot as plt
import train
import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
import pickle
import sklearn
import config
import path_int
# import umap
from sklearn import manifold
from sklearn.cluster import KMeans
import scipy
# import umap

def cal_overlap_score(args, hist_state, mid_ei, goal_ei, mid_traj, goal_traj):
    mid_past_overlap_score = []
    goal_past_overlap_score = []
    mid_distance_score_lists = []
    goal_distance_score_lists = []
    hist_pos = hist_state[:,:,:2]
    backward_steps = 5
    for n in range(args.n_agents):
        for mid_ei_n, mid_replay_n in zip(mid_ei[n], mid_traj[n]):
            mid_replay_double_digit = jnp.stack([mid_replay_n%args.width, mid_replay_n//args.width], axis=1)
            # print(mid_ei_n, mid_replay_n)
            hist_pos_double_digit = hist_pos[mid_ei_n.item()-backward_steps:mid_ei_n.item(),n]
            hist_pos_single_digit = hist_pos_double_digit[...,0]*args.width + hist_pos_double_digit[...,1]
            # print(hist_pos_single_digit.shape)
            # print(mid_replay_n.shape)
            overlap_matrix = mid_replay_n.reshape(-1,1)==hist_pos_single_digit.reshape(1,-1)
            # print(overlap_matrix)
            overlap_score = overlap_matrix.sum(1).mean()
            mid_past_overlap_score.append(overlap_score)

            mid_distance_x_matrix = jnp.square(mid_replay_double_digit[:,0].reshape(-1,1)-hist_pos_double_digit[...,0].reshape(1,-1))
            mid_distance_y_matrix = jnp.square(mid_replay_double_digit[:,1].reshape(-1,1)-hist_pos_double_digit[...,1].reshape(1,-1))
            mid_distance_matrix = jnp.exp(-(mid_distance_x_matrix+mid_distance_y_matrix))
            mid_distance_score = mid_distance_matrix.mean()
            mid_distance_score_lists.append(mid_distance_score)

        for goal_ei_n, goal_replay_n in zip(goal_ei[n], goal_traj[n]):
            # print(goal_ei_n, goal_replay_n)
            goal_replay_double_digit = jnp.stack([goal_replay_n%args.width, goal_replay_n//args.width], axis=1)
            hist_pos_double_digit = hist_pos[goal_ei_n.item()-backward_steps:goal_ei_n.item(),n]
            hist_pos_single_digit = hist_pos_double_digit[...,0]*args.width + hist_pos_double_digit[...,1]
            overlap_matrix = goal_replay_n.reshape(-1,1)==hist_pos_single_digit.reshape(1,-1)
            overlap_score = overlap_matrix.sum(1).mean()
            goal_past_overlap_score.append(overlap_score)

            goal_distance_x_matrix = jnp.square(goal_replay_double_digit[:,0].reshape(-1,1)-hist_pos_double_digit[...,0].reshape(1,-1))
            goal_distance_y_matrix = jnp.square(goal_replay_double_digit[:,1].reshape(-1,1)-hist_pos_double_digit[...,1].reshape(1,-1))
            goal_distance_matrix = jnp.exp(-(goal_distance_x_matrix+goal_distance_y_matrix))
            goal_distance_score = goal_distance_matrix.mean()
            goal_distance_score_lists.append(goal_distance_score)

    return jnp.array(mid_distance_score_lists).mean(), jnp.array(goal_distance_score_lists).mean()
    return jnp.array(mid_past_overlap_score).mean(), jnp.array(goal_past_overlap_score).mean()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def cal_biology_line_chart(args):
    #red yellow blue green
    biology_line_chart_mean = jnp.array([
        [0.14,0.05,0.65,0.16],
        [0.13,0.06,0.12,0.69],
        [0.11,0.23,0.06,0.60],
        [0.04,0.58,0.10,0.28],
        [0.03,0.68,0.03,0.26],
    ])
    color_list = ['r', 'y', 'b', 'g']
    label_list = ['past_cons', 'new_cons', 'past_plan', 'new_plan']
    fig, axis = plt.subplots(1,1,figsize=(6,4))
    x = np.arange(5)
    for i in range(len(color_list)):
        axis.plot(x, biology_line_chart_mean[:,i], color_list[i], label=label_list[i], linewidth=5)
        # axis.fill_between(x, biology_line_chart_mean[:,i], line_upper_bound[:,i], color=color_list[i], alpha=0.2)
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['bottom'].set_linewidth(3)
    axis.spines['left'].set_linewidth(3)
    axis.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    axis.set_yticklabels(['0','','','','0.8','1.0'])
    axis.set_xticks(x, ['Pre', '', 'Learning', '', 'Post'])
    axis.set_ylabel('Proportion', fontdict={'size':20})
    axis.tick_params(axis='both', which='major', labelsize=20, width=3, length=10)
    title = 'Events at open field'
    axis.set_title(title, fontdict={'size':25})
    plt.show()
    jnp.save('./figures/biology_line_chart_mean.npy', biology_line_chart_mean)
    return

def cal_plot_task_setting(args):
    grid = jnp.zeros((5,5))
    width = args.width
    height = args.height
    state_traj = jnp.array([[0,0],[1,0],[1,1],[2,1],[2,2],[3,2],[3,1]])
    
    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    axis[0].grid(linewidth=3,zorder=0)
    axis[0].set_xlim(0-0.5,width-0.5)
    axis[0].set_ylim(0-0.5,height-0.5)
    axis[0].set_xticklabels([])
    axis[0].set_yticklabels([])
    axis[0].tick_params(length=0)
    axis[0].spines['top'].set_linewidth(3)
    axis[0].spines['right'].set_linewidth(3)
    axis[0].spines['bottom'].set_linewidth(3)
    axis[0].spines['left'].set_linewidth(3)
    reward_center = jnp.array([[3,1]])
    axis[0].scatter(reward_center[:,0],reward_center[:,1], c='r', s=500, marker='*', zorder=100)


    axis[1].grid(linewidth=3,zorder=0)
    axis[1].set_xlim(0-0.5,width-0.5)
    axis[1].set_ylim(0-0.5,height-0.5)
    axis[1].plot(state_traj[:,0],state_traj[:,1], linewidth=5, zorder=100)
    axis[1].set_xticklabels([])
    axis[1].set_yticklabels([])
    axis[1].tick_params(length=0)
    axis[1].spines['top'].set_linewidth(3)
    axis[1].spines['right'].set_linewidth(3)
    axis[1].spines['bottom'].set_linewidth(3)
    axis[1].spines['left'].set_linewidth(3)
    reward_center = jnp.array([[3,1]])
    axis[1].scatter(reward_center[:,0],reward_center[:,1], c='r', s=500, marker='*', zorder=200)


    axis[2].grid(linewidth=3,zorder=0)
    axis[2].set_xlim(0-0.5,width-0.5)
    axis[2].set_ylim(0-0.5,height-0.5)
    axis[2].set_xticklabels([])
    axis[2].set_yticklabels([])
    axis[2].tick_params(length=0)
    axis[2].spines['top'].set_linewidth(3)
    axis[2].spines['right'].set_linewidth(3)
    axis[2].spines['bottom'].set_linewidth(3)
    axis[2].spines['left'].set_linewidth(3)
    replay_traj = jnp.array([[4,2],[2,0],[1,0],[0,0]])
    direction_traj = jnp.diff(replay_traj, axis=0)
    # axis[2].plot(replay_traj[:,0],replay_traj[:,1], linewidth=5, c='deeppink')
    # axis[2].scatter(reward_center[:,0],reward_center[:,1], c='r', s=500, marker='*', zorder=100)
    axis[2].scatter(replay_traj[:,0],replay_traj[:,1], c='deeppink', s=400, marker='o', zorder=200)
    axis[2].quiver(replay_traj[:-1,0], replay_traj[:-1,1], 
                   direction_traj[:,0], direction_traj[:,1], width=0.018, scale=5, color='deeppink', headwidth=3,alpha=1, zorder=300)

    fig.tight_layout()

    plt.savefig('./figures/state_traj.png')


def cal_plot_manifold_stability(args, hist_hippo, hist_theta, reward_n,
                    replay_infos, hist_phase, hist_state, titles):
    mid_hippo, goal_hippo, mid_theta, goal_theta = replay_infos
    print(hist_hippo.shape, hist_theta.shape)
    switch_mid_idx = [0]+[len(x) for x in mid_hippo]
    switch_mid_idx = jnp.cumsum(jnp.array(switch_mid_idx))
    print('switch_mid_idx',switch_mid_idx)
    # all_hippo, theta
    # print('hist_state')
    # print(hist_state[:,0])
    # print('reward_n[0]')
    # print(reward_n[0])
    hist_pos = hist_state[:,:,:2]
    mid_hippo = jnp.concatenate(mid_hippo)
    goal_hippo = jnp.concatenate(goal_hippo)
    mid_theta = jnp.concatenate(mid_theta)
    goal_theta = jnp.concatenate(goal_theta)
    # print(mid_hippo.shape, goal_hippo.shape, mid_theta.shape, goal_theta.shape)
    hist_hippo_start_mid = [[] for i in range(len(titles))]
    hist_theta_start_mid = [[] for i in range(len(titles))]
    hist_hippo_mid_goal = [[] for i in range(len(titles))]
    hist_theta_mid_goal = [[] for i in range(len(titles))]
    for n in range(args.n_agents):
        count = 0
        for i in range(len(reward_n[n])-1):
            # print(i)
            if reward_n[n][i][-1]==0 and reward_n[n][i+1][-1]==1:
                # mid->goal
                s, e = reward_n[n][i][0], reward_n[n][i+1][0]
                # c_idx = c_idx.at[s:e, n].set(jnp.linspace(args.path_mid_value,args.path_goal_value,e-s))
                context_switch = (hist_phase[s,n]!=hist_phase[e,n])
                if count <= 3:
                    hist_hippo_mid_goal[count].append(hist_hippo[s:e,n])
                    hist_theta_mid_goal[count].append(hist_theta[s:e,n])
                # print('context_switch:',context_switch)
            if reward_n[n][i][-1]==1 and reward_n[n][i+1][-1]==0:
                # start->mid
                s, e = reward_n[n][i][0], reward_n[n][i+1][0]
                # c_idx = c_idx.at[s:e, n].set(jnp.linspace(args.path_start_value,args.path_mid_value,e-s))
                if (not context_switch) and (count==0):
                    hist_hippo_start_mid[0].append(hist_hippo[s:e,n])
                    hist_theta_start_mid[0].append(hist_theta[s:e,n])
                elif context_switch:
                    # for ei in range(s,e):
                    #     if jnp.all(hist_pos[ei,n]==env.pseudo_reward_list[args.pseudo_reward_idx][0]):
                    #         exploration_start = ei
                    #         break
                    # c_idx = c_idx.at[exploration_start:e,n].set(jnp.linspace(args.path_start_value,args.path_mid_value,e-exploration_start))
                    # c_idx = c_idx.at[s:exploration_start,n].set(jnp.linspace(args.path_start_value,args.path_mid_value,exploration_start-s))
                    hist_hippo_start_mid[1].append(hist_hippo[s:e,n])
                    hist_theta_start_mid[1].append(hist_theta[s:e,n])
                    count += 1
                elif count == 1:
                    hist_hippo_start_mid[2].append(hist_hippo[s:e,n])
                    hist_theta_start_mid[2].append(hist_theta[s:e,n])
                    count += 1
                elif count == 2:
                    hist_hippo_start_mid[3].append(hist_hippo[s:e,n])
                    hist_theta_start_mid[3].append(hist_theta[s:e,n])
                    count += 1

    hist_hippo_start_mid = [jnp.concatenate(x) for x in hist_hippo_start_mid]
    hist_theta_start_mid = [jnp.concatenate(x) for x in hist_theta_start_mid]
    hist_hippo_mid_goal = [jnp.concatenate(x) for x in hist_hippo_mid_goal]
    hist_theta_mid_goal = [jnp.concatenate(x) for x in hist_theta_mid_goal]
    hist_hippo_all = [jnp.concatenate((hist_hippo_start_mid[i], hist_hippo_mid_goal[i])) for i in range(len(hist_hippo_start_mid))]
    hist_theta_all = [jnp.concatenate((hist_theta_start_mid[i], hist_theta_mid_goal[i])) for i in range(len(hist_theta_start_mid))]

    variance_hippo_start_mid = []
    variance_theta_start_mid = []
    variance_hippo_mid_goal = []
    variance_theta_mid_goal = []
    variance_hippo = []
    variance_theta = []
    n_cluster = 20
    for i in range(len(titles)):
        # kmeans = KMeans(n_cluster).fit(hist_hippo_start_mid[i])
        # cluster_center = kmeans.cluster_centers_[kmeans.labels_]
        # variance_hippo_start_mid.append(jnp.mean((hist_hippo_start_mid[i] - cluster_center)**2,1))
        # print('all data and cluster center')
        # print(hist_hippo_start_mid[i].shape, cluster_center.shape)
        # kmeans = KMeans(n_cluster).fit(hist_theta_start_mid[i])
        # cluster_center = kmeans.cluster_centers_[kmeans.labels_]
        # variance_theta_start_mid.append(jnp.mean((hist_theta_start_mid[i] - cluster_center)**2,1))
        # kmeans = KMeans(n_cluster).fit(hist_hippo_mid_goal[i])
        # cluster_center = kmeans.cluster_centers_[kmeans.labels_]
        # variance_hippo_mid_goal.append(jnp.mean((hist_hippo_mid_goal[i] - cluster_center)**2,1))
        # kmeans = KMeans(n_cluster).fit(hist_theta_mid_goal[i])
        # cluster_center = kmeans.cluster_centers_[kmeans.labels_]
        # variance_theta_mid_goal.append(jnp.mean((hist_theta_mid_goal[i] - cluster_center)**2,1))
        kmeans = KMeans(n_cluster).fit(hist_hippo_all[i])
        cluster_center = kmeans.cluster_centers_[kmeans.labels_]
        variance_hippo.append(jnp.mean((hist_hippo_all[i] - cluster_center)**2,1))
        kmeans = KMeans(n_cluster).fit(hist_theta_all[i])   
        cluster_center = kmeans.cluster_centers_[kmeans.labels_]
        variance_theta.append(jnp.mean((hist_theta_all[i] - cluster_center)**2,1))

    # variance_hippo_start_mid_mean = jnp.array([x.mean() for x in variance_hippo_start_mid])
    # variance_theta_start_mid_mean = jnp.array([x.mean() for x in variance_theta_start_mid])
    # variance_hippo_mid_goal_mean = jnp.array([x.mean() for x in variance_hippo_mid_goal])
    # variance_theta_mid_goal_mean = jnp.array([x.mean() for x in variance_theta_mid_goal])

    fig, axes = plt.subplots(2,2)
    # axes[0][0].plot(variance_hippo_start_mid_mean)
    # axes[0][0].set_title('variance_hippo_start_mid')
    # axes[0][1].plot(variance_theta_start_mid_mean)
    # axes[0][1].set_title('variance_theta_start_mid')
    # axes[1][0].plot(variance_hippo_mid_goal_mean)
    # axes[1][0].set_title('variance_hippo_mid_goal')
    # axes[1][1].plot(variance_theta_mid_goal_mean)
    # axes[1][1].set_title('variance_theta_mid_goal')


    fig.tight_layout()
    if not os.path.exists('figures'):
        os.makedirs('figures')
    save_figure = './figures/'+args.prefix+'/trajectory_stability_'+args.suffix+'.png'
    print('save trajectory_stability to', save_figure)
    plt.savefig(save_figure)
    # plt.show()
    return variance_hippo_start_mid, variance_theta_start_mid, variance_hippo_mid_goal, variance_theta_mid_goal, \
            variance_hippo, variance_theta

def cal_plot_subspace_dimension(args, hist_hippo, hist_theta,
                    replay_infos, hist_phase, hist_state, titles):
    n_component = 20
    dim_threshold = 0.7

    mid_hippo, goal_hippo, mid_theta, goal_theta = replay_infos
    # print(hist_hippo.shape, hist_theta.shape)
    switch_mid_idx = [0]+[len(x) for x in mid_hippo]
    switch_mid_idx = jnp.cumsum(jnp.array(switch_mid_idx))
    print('switch_mid_idx',switch_mid_idx)
    # all_hippo, theta
    mid_hippo = jnp.concatenate(mid_hippo)
    goal_hippo = jnp.concatenate(goal_hippo)
    mid_theta = jnp.concatenate(mid_theta)
    goal_theta = jnp.concatenate(goal_theta)
    print(mid_hippo.shape, goal_hippo.shape, mid_theta.shape, goal_theta.shape)
    fig, axes = plt.subplots(4,5,figsize=(20,12))
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            axes[i][j].set_ylim(0,1)
            axes[i][j].plot(jnp.ones(n_component,)*dim_threshold, c='orange', linestyle='--')

    key = jax.random.PRNGKey(args.initkey)
    n_hist_show = 200
    hist_idx = jax.random.randint(key, (n_hist_show,), 0, hist_hippo.shape[0]*hist_hippo.shape[1])
    hist_hippo_to_show = hist_hippo.reshape(-1, args.hidden_size)[hist_idx]
    hist_theta_to_show = hist_theta.reshape(-1, args.theta_hidden_size)[hist_idx]
    reducer = PCA(n_components=n_component)
    reducer.fit(hist_hippo_to_show)
    evr_hist_hippo = jnp.cumsum(reducer.explained_variance_ratio_)
    axes[0][0].plot(evr_hist_hippo)
    dim = jnp.min(jnp.where(evr_hist_hippo>dim_threshold)[0])
    axes[0][0].set_title(dim+1)

    reducer = PCA(n_components=n_component)
    reducer.fit(hist_theta_to_show)
    evr_hist_theta = jnp.cumsum(reducer.explained_variance_ratio_)
    axes[2][0].plot(evr_hist_theta)
    dim = jnp.min(jnp.where(evr_hist_theta>dim_threshold)[0])
    axes[2][0].set_title(dim+1)
    
    theta_dims = jnp.zeros((2, len(titles)))
    for i in range(len(titles)):
        mid_idx_of_interest = jnp.arange(switch_mid_idx[i], switch_mid_idx[i+1])
        mid_hippo_of_interest = mid_hippo[mid_idx_of_interest].reshape(-1,args.hidden_size)
        mid_theta_of_interest = mid_theta[mid_idx_of_interest].reshape(-1,args.theta_hidden_size)
        # balance_idx_hist = jnp.
        reducer = PCA(n_components=50)
        reducer.fit(mid_hippo_of_interest)
        evr_mid_hippo = jnp.cumsum(reducer.explained_variance_ratio_)
        axes[0][i+1].plot(evr_mid_hippo)
        # print(jnp.where(evr_mid_hippo>dim_threshold)[0])
        dim = jnp.min(jnp.where(evr_mid_hippo>dim_threshold)[0])
        axes[0][i+1].set_title(dim+1)

        reducer = PCA(n_components=50)
        reducer.fit(jnp.concatenate((hist_hippo_to_show,mid_hippo_of_interest)))
        evr_hist_mid_hippo = jnp.cumsum(reducer.explained_variance_ratio_)
        axes[1][i+1].plot(evr_hist_mid_hippo)
        dim = jnp.min(jnp.where(evr_hist_mid_hippo>dim_threshold)[0])
        axes[1][i+1].set_title(dim+1)

        reducer = PCA(n_components=n_component)
        reducer.fit(mid_theta_of_interest)
        evr_mid_theta = jnp.cumsum(reducer.explained_variance_ratio_)
        axes[2][i+1].plot(evr_mid_theta)
        dim = jnp.min(jnp.where(evr_mid_theta>dim_threshold)[0])
        axes[2][i+1].set_title(dim+1)
        theta_dims = theta_dims.at[0,i].set(dim+1)

        reducer = PCA(n_components=n_component)
        reducer.fit(jnp.concatenate((hist_theta_to_show,mid_theta_of_interest)))
        evr_hist_mid_theta = jnp.cumsum(reducer.explained_variance_ratio_)
        axes[3][i+1].plot(evr_hist_mid_theta)
        dim = jnp.min(jnp.where(evr_hist_mid_theta>dim_threshold)[0])
        axes[3][i+1].set_title(dim+1)
        theta_dims = theta_dims.at[1,i].set(dim+1)

        # print(evr_mid_hippo)
        # print(evr_hist_mid_hippo)
        # print(evr_mid_theta)
        # print(evr_hist_mid_theta)
    fig.tight_layout()
    if not os.path.exists('figures'):
        os.makedirs('figures')
    save_figure = './figures/'+args.prefix+'/subspace_dimension_'+args.suffix+'.png'
    print('save subspace_dimension to', save_figure)
    plt.savefig(save_figure)
    plt.close()


    fig, axes = plt.subplots(1,3,figsize=(18,6))
    axes = axes.flatten()
    for axis in axes:
        axis.spines['top'].set_color('none')
        axis.spines['right'].set_color('none')
        axis.spines['bottom'].set_linewidth(3)
        axis.spines['left'].set_linewidth(3)
        axis.tick_params(axis='both', which='major', labelsize=20, width=3, length=10)
    mid_idx_of_interest = jnp.arange(switch_mid_idx[1], switch_mid_idx[2])
    mid_theta_of_interest = mid_theta[mid_idx_of_interest].reshape(-1,args.theta_hidden_size)
    reducer = PCA(n_components=12)
    # reducer.fit(jnp.concatenate((hist_theta_to_show,mid_theta_of_interest)))
    reducer.fit(mid_theta_of_interest)
    evr_hist_theta = jnp.cumsum(reducer.explained_variance_ratio_)
    axes[0].plot(evr_hist_theta, color='black', linewidth=3)
    axes[0].scatter(jnp.arange(12), evr_hist_theta, color='black', s=200, marker='o')
    axes[0].set_xticks([0,1,2,9],[1,2,3,10],fontdict={'fontsize':40})
    axes[0].set_yticks([0.4,0.7,1],[0.4,0.7,1],fontdict={'fontsize':40})
    # axes[0].set_xlabel('Dimension', fontsize=15)
    # axes[0].set_ylabel('Accumulated Explained Variance', fontsize=15)
    axes[0].axhline(y=dim_threshold, color='grey', linestyle='dashed', linewidth=3)
    axes[0].axhline(y=1, color='grey', linestyle='dashed', linewidth=3)
    # dim = jnp.min(jnp.where(evr_hist_theta>dim_threshold)[0])
    # axes[0].set_title(dim+1)
    axes[1].bar(jnp.arange(len(titles)), theta_dims[0], color='seagreen')
    axes[1].set_xticks(jnp.arange(len(titles)), jnp.arange(len(titles)), fontdict={'fontsize':40})
    # axes[1].set_xlabel('Times of meeting checkpoint 2 repeatedly', fontsize=15)
    axes[1].set_yticks(jnp.arange(3), jnp.arange(3), fontdict={'fontsize':40})
    # axes[1].set_ylabel('Dimension of neural subspace', fontsize=15)
    # axes[1].set_title('Only replay', fontsize=20)
    axes[2].bar(jnp.arange(len(titles)), theta_dims[1], color='darkturquoise')
    axes[2].set_xticks(jnp.arange(len(titles)), jnp.arange(len(titles)), fontdict={'fontsize':40})
    # axes[2].set_xlabel('Times of meeting checkpoint 2 repeatedly', fontsize=15)
    axes[2].set_yticks(jnp.arange(4), jnp.arange(4), fontdict={'fontsize':40})
    # axes[2].set_ylabel('Dimension of neural subspace', fontsize=15)
    # axes[2].set_title('Replay + Real experience', fontsize=20)
    fig.tight_layout()
    plt.savefig('./figures/'+args.prefix+'/subspace_dimension_bar_'+args.suffix+'.png')
    print('save subspace_dimension_bar to', './figures/'+args.prefix+'/subspace_dimension_bar_'+args.suffix+'.png')
    # plt.show()
    return evr_hist_theta, theta_dims

def cal_plot_manifold(args, hist_hippo, hist_theta, reward_n, replay_infos, hist_phase, hist_state, subtitles):
    n_component = 2
    reducer = PCA(n_components=n_component)
    # reducer = umap.UMAP(n_components=n_component, random_state=42)
    # reducer = manifold.TSNE(n_components=n_component, init='pca', random_state=501)
    point_number = 500
    point_size = 10

    hist_pos = hist_state[:,:,:2]
    mid_hippo, goal_hippo, mid_theta, goal_theta = replay_infos
    switch_mid_idx = [0]+[len(x) for x in mid_hippo]
    switch_mid_idx = jnp.cumsum(jnp.array(switch_mid_idx))
    mid_hippo = jnp.concatenate(mid_hippo)
    goal_hippo = jnp.concatenate(goal_hippo)
    mid_theta = jnp.concatenate(mid_theta)
    goal_theta = jnp.concatenate(goal_theta)
    print(len(mid_hippo))
    # dimension reduction
    reduced_hippo = reducer.fit_transform(jnp.concatenate( (
        hist_hippo.reshape(-1, args.hidden_size),
        mid_hippo.reshape(-1, args.hidden_size)) ))
    reduced_theta = reducer.fit_transform(jnp.concatenate( (
        hist_theta.reshape(-1, args.theta_hidden_size),
        mid_theta.reshape(-1, args.theta_hidden_size)) ))
    
    hist_hippo_reduced = reduced_hippo[:hist_hippo.shape[0]*args.n_agents].reshape(hist_hippo.shape[0], args.n_agents, n_component)
    mid_hippo_reduced = reduced_hippo[hist_hippo.shape[0]*args.n_agents : hist_hippo.shape[0]*args.n_agents+mid_hippo.shape[0]*args.replay_steps].reshape(mid_hippo.shape[0], args.replay_steps, n_component)

    hist_theta_reduced = reduced_theta[:hist_theta.shape[0]*args.n_agents].reshape(hist_theta.shape[0], args.n_agents, n_component)
    mid_theta_reduced = reduced_theta[hist_theta.shape[0]*args.n_agents:hist_theta.shape[0]*args.n_agents+mid_theta.shape[0]*args.replay_steps].reshape(mid_theta.shape[0], args.replay_steps, n_component)

    hist_hippo_start_mid_centroid, hist_theta_start_mid_centroid, hist_hippo_mid_goal_centroid, hist_theta_mid_goal_centroid, \
        hist_hippo_before_switch, hist_theta_before_switch, hist_hippo_exploring, hist_theta_exploring, \
        hist_hippo_after_switch, hist_theta_after_switch, hist_c_before_switch, hist_c_exploring, hist_c_after_switch = \
            cal_hist_centroid_scatter(args, hist_state, hist_hippo_reduced, hist_theta_reduced, hist_phase, reward_n, hist_pos)

    # 3*4*3 # before switch, exploration, after switch

    fig = plt.figure(figsize=(20,10))

    goal_idx_of_interest = jnp.arange(len(goal_hippo))

    mid_c = (jnp.arange(args.replay_steps)/2+1).reshape(1,-1).repeat(mid_theta.shape[0],0)
    goal_c = (jnp.arange(args.replay_steps-1, -1, -1)/2+1).reshape(1,-1).repeat(goal_theta.shape[0],0)

    if args.zero_goal_idx:
        goal_idx_of_interest = 0
        goal_c = jnp.zeros_like(goal_c)
    if args.zero_traj_idx:
        traj_idx_of_interest = 0

    ###00, 1, 2, 3
    # cal replay centroid and scatter
    for i in range(len(switch_mid_idx)-1):
        mid_idx_of_interest = jnp.arange(switch_mid_idx[i], switch_mid_idx[i+1])

        if args.zero_mid_idx: 
            mid_idx_of_interest = 0
            mid_c = jnp.zeros_like(mid_c)
        if not args.replay_interest:
            mid_idx_of_interest = jnp.arange(len(mid_hippo))

        mid_hippo_of_interest = mid_hippo_reduced[mid_idx_of_interest].reshape(-1,n_component)
        mid_hippo_centroid = mid_hippo_reduced[mid_idx_of_interest].mean(0).reshape(-1,n_component)
        # 4*3

        mid_theta_of_interest = mid_theta_reduced[mid_idx_of_interest].reshape(-1,n_component)
        mid_theta_centroid = mid_theta_reduced[mid_idx_of_interest].mean(0).reshape(-1,n_component)

        mid_c_of_interest = mid_c[mid_idx_of_interest].reshape(-1)
        mid_c_of_interest = mid_c_of_interest.at[0].set(0)

        # print(mid_c_of_interest)
        #plot 3d scatter
        if n_component == 3:
            ax_hippo = fig.add_subplot(2,4,1+i*2, projection='3d')
        elif n_component == 2:
            ax_hippo = fig.add_subplot(2,4,1+i*2)
        plot_scatter(ax_hippo, hist_hippo_before_switch, hist_hippo_after_switch, mid_hippo_of_interest, 
                     hist_c_before_switch, hist_c_after_switch, mid_c_of_interest, point_number, point_size, subtitles[i])
        if i==0:
            latent_hippo_trajectory_00 = jnp.concatenate((hist_hippo_start_mid_centroid[0], mid_hippo_centroid, hist_hippo_mid_goal_centroid[0]))
        elif i==1:
            latent_hippo_trajectory_01 = jnp.concatenate((hist_hippo_start_mid_centroid[0], mid_hippo_centroid, hist_hippo_mid_goal_centroid[2]))
        elif i>=2:
            latent_hippo_trajectory_11 = jnp.concatenate((hist_hippo_start_mid_centroid[2], mid_hippo_centroid, hist_hippo_mid_goal_centroid[2]))


        if n_component == 3:
            ax_theta = fig.add_subplot(2,4,i*2+2, projection='3d') 
        elif n_component == 2:
            ax_theta = fig.add_subplot(2,4,i*2+2)
        plot_scatter(ax_theta, hist_theta_before_switch, hist_theta_after_switch, mid_theta_of_interest, 
                     hist_c_before_switch, hist_c_after_switch, mid_c_of_interest, point_number, point_size, subtitles[i])
        if i==0:
            latent_theta_trajectory_00 = jnp.concatenate((hist_theta_start_mid_centroid[0], mid_theta_centroid, hist_theta_mid_goal_centroid[0]))
        elif i==1:
            latent_theta_trajectory_01 = jnp.concatenate((hist_theta_start_mid_centroid[0], mid_theta_centroid, hist_theta_mid_goal_centroid[2]))
        elif i>=2:
            latent_theta_trajectory_11 = jnp.concatenate((hist_theta_start_mid_centroid[2], mid_theta_centroid, hist_theta_mid_goal_centroid[2]))
    axes = fig.get_axes()
    print(len(axes))
    # plot centroid of the trajectories
    centroid_traj_colors = ['salmon','magenta','deepskyblue','deepskyblue']
    dye_index = [0,1,2,2]
    for i in range(len(subtitles)):
        plot_centroid(axes[i*2], axes[i*2+1], \
            [latent_hippo_trajectory_00, latent_hippo_trajectory_01, latent_hippo_trajectory_11], \
            [latent_theta_trajectory_00, latent_theta_trajectory_01, latent_theta_trajectory_11], \
                dye_index[i], centroid_traj_colors[i])
    fig.tight_layout()
    plt.suptitle(args.prefix)
    if args.zero_mid_idx:
        args.suffix='only_traj'
    if not os.path.exists('figures'):
        os.makedirs('figures')
    save_figure = './figures/'+args.prefix+'/manifold_'+args.suffix+'.png'
    print('save manifold to', save_figure)
    plt.savefig(save_figure)
    plt.show()
    return

def cal_hist_centroid_scatter(args, hist_state, hist_hippo_reduced, hist_theta_reduced, hist_phase,
                           reward_n, hist_pos):
    c_idx = jnp.zeros((args.total_eval_steps, args.n_agents))
    hist_hippo_sample_start_mid = [[],[],[]]
    # before switch, exploration, after switch
    hist_theta_sample_start_mid = [[],[],[]]
    hist_hippo_sample_mid_goal = [[],[],[]]
    # before switch, exploration, after switch
    hist_theta_sample_mid_goal = [[],[],[]]
    e_list = []
    s_list = []
    exploration_start_list = []
    exploration_end_list = []
    print('hist_state')
    print(hist_state[:,0])
    print('reward_n[0]')
    print(reward_n[0])
    past_length = 0
    for n in range(args.n_agents):
        s = 0
        e = 0
        context_switched = False
        # print(reward_n[n])
        # if n==73:
        #     print(reward_n[n])
        for i in range(len(reward_n[n])-1):
            # print(reward_n[n][i],reward_n[n][i+1])
            if reward_n[n][i][-1]==0 and reward_n[n][i+1][-1]==1:
                # mid->goal
                # print(reward_n[n][i],reward_n[n][i+1])
                s, e = reward_n[n][i][0], reward_n[n][i+1][0]
                c_idx = c_idx.at[s:e, n].set(jnp.linspace(args.path_mid_value,args.path_goal_value,e-s))
                # c_idx = c_idx.at[s:e, n].set(jnp.linspace(1,3,e-s))
                if len(s_list)==n:
                    s_list.append(s)
                context_switch = (hist_phase[s,n]!=hist_phase[e,n])
                if context_switch:
                    hist_hippo_sample_mid_goal[1].append(hist_hippo_reduced[s:s+4,n])
                    hist_theta_sample_mid_goal[1].append(hist_theta_reduced[s:s+4,n])
                elif hist_phase[s,n]==0:
                    hist_hippo_sample_mid_goal[0].append(hist_hippo_reduced[s:s+4,n])
                    hist_theta_sample_mid_goal[0].append(hist_theta_reduced[s:s+4,n])
                elif hist_phase[s,n]==1:
                    hist_hippo_sample_mid_goal[2].append(hist_hippo_reduced[s:s+4,n])
                    hist_theta_sample_mid_goal[2].append(hist_theta_reduced[s:s+4,n])
                # print('context_switch:',context_switch)
            if reward_n[n][i][-1]==1 and reward_n[n][i+1][-1]==0:
                # start->mid
                s, e = reward_n[n][i][0], reward_n[n][i+1][0]
                c_idx = c_idx.at[s:e, n].set(jnp.linspace(args.path_start_value,args.path_mid_value,e-s))
                if context_switch:
                    for ei in range(s,e):
                        # print(hist_pos[ei,n])
                        if jnp.all(hist_pos[ei,n]==env.pseudo_reward_list[args.pseudo_reward_idx][0]):
                            exploration_start = ei
                            break
                    c_idx = c_idx.at[exploration_start:e,n].set(jnp.linspace(args.path_start_value,args.path_mid_value,e-exploration_start))
                    c_idx = c_idx.at[s:exploration_start,n].set(jnp.linspace(args.path_start_value,args.path_mid_value,exploration_start-s))
                    exploration_start_list.append(exploration_start)
                    exploration_end_list.append(e)

                    hist_hippo_sample_start_mid[1].append(hist_hippo_reduced[exploration_start:exploration_start+4,n])
                    hist_theta_sample_start_mid[1].append(hist_theta_reduced[exploration_start:exploration_start+4,n])
                    hist_hippo_sample_start_mid[0].append(hist_hippo_reduced[s:s+4,n])
                    hist_theta_sample_start_mid[0].append(hist_theta_reduced[s:s+4,n])

                    context_switched = True

                elif hist_phase[s,n]==0:
                    hist_hippo_sample_start_mid[0].append(hist_hippo_reduced[s:s+4,n])
                    hist_theta_sample_start_mid[0].append(hist_theta_reduced[s:s+4,n])
                elif hist_phase[s,n]==1:
                    hist_hippo_sample_start_mid[2].append(hist_hippo_reduced[s:s+4,n])
                    hist_theta_sample_start_mid[2].append(hist_theta_reduced[s:s+4,n])

        if s==0:
            s_list.append(s)
        e_list.append(e)
        # print(n, 's_list:',s_list)
        if len(s_list) <= past_length:
            break
        past_length = len(s_list)
        if not context_switched:
            exploration_start_list.append(e)
            exploration_end_list.append(e)
    # print(len(s_list),len(e_list),len(exploration_start_list),len(exploration_end_list))
    # print('s,e,start,end')
    # print(s_list[0],e_list[0],exploration_start_list[0],exploration_end_list[0])
    # print('c_idx')
    # print(c_idx[:,0])
    # centroid for manifold
    hist_hippo_sample_start_mid = [jnp.stack(x,0) for x in hist_hippo_sample_start_mid]
    hist_theta_sample_start_mid = [jnp.stack(x,0) for x in hist_theta_sample_start_mid]
    hist_hippo_sample_mid_goal = [jnp.stack(x,0) for x in hist_hippo_sample_mid_goal]
    hist_theta_sample_mid_goal = [jnp.stack(x,0) for x in hist_theta_sample_mid_goal]

    # hist_hippo_sample = [jnp.concatenate([hist_hippo_sample_start_mid[i], hist_hippo_sample_mid_goal[i]]) for i in range(len(hist_hippo_sample_start_mid))]
    # hist_theta_sample = [jnp.concatenate([hist_theta_sample_start_mid[i], hist_theta_sample_mid_goal[i]]) for i in range(len(hist_theta_sample_start_mid)]
    # hist_hippo_sample_start_mid_mean = jnp.array([x.mean(0) for x in hist_hippo_sample_start_mid])
    # hist_theta_sample_start_mid_mean = jnp.array([x.mean(0) for x in hist_theta_sample_start_mid])
    # hist_hippo_sample_mid_goal_mean = jnp.array([x.mean(0) for x in hist_hippo_sample_mid_goal])
    # hist_theta_sample_mid_goal_mean = jnp.array([x.mean(0) for x in hist_theta_sample_mid_goal])
    # cal scatter
    # hist_hippo_before_switch = []
    # hist_theta_before_switch = []
    # hist_hippo_after_switch = []
    # hist_theta_after_switch = []
    # hist_hippo_exploring = []
    # hist_theta_exploring = []
    # hist_c_before_switch = []
    # hist_c_exploring = []
    # hist_c_after_switch = []
    # # print(len(s_list),len(e_list),args.n_agents)
    # for n in range(args.n_agents):
    #     # print(s_list[n],e_list[n],hist_hippo_reduced.shape)
    #     hist_hippo_before_switch.append(hist_hippo_reduced[s_list[n]:exploration_start_list[n],n])
    #     hist_theta_before_switch.append(hist_theta_reduced[s_list[n]:exploration_start_list[n],n])
    #     hist_hippo_exploring.append(hist_hippo_reduced[exploration_start_list[n]:exploration_end_list[n],n])
    #     hist_theta_exploring.append(hist_theta_reduced[exploration_start_list[n]:exploration_end_list[n],n])
    #     hist_hippo_after_switch.append(hist_hippo_reduced[exploration_end_list[n]:e_list[n],n])
    #     hist_theta_after_switch.append(hist_theta_reduced[exploration_end_list[n]:e_list[n],n])
    #     hist_c_before_switch.append(c_idx[s_list[n]:exploration_start_list[n],n])
    #     hist_c_exploring.append(c_idx[exploration_start_list[n]:exploration_end_list[n],n])
    #     hist_c_after_switch.append(c_idx[exploration_end_list[n]:e_list[n],n])
    #     #TODO:Simplify the procedure here to jnp.arange()
    # hist_hippo_before_switch = jnp.concatenate(hist_hippo_before_switch)
    # hist_theta_before_switch = jnp.concatenate(hist_theta_before_switch)
    # hist_hippo_exploring = jnp.concatenate(hist_hippo_exploring)
    # hist_theta_exploring = jnp.concatenate(hist_theta_exploring)
    # hist_hippo_after_switch = jnp.concatenate(hist_hippo_after_switch)
    # hist_theta_after_switch = jnp.concatenate(hist_theta_after_switch)
    # hist_c_before_switch = jnp.concatenate(hist_c_before_switch)
    # hist_c_exploring = jnp.concatenate(hist_c_exploring)
    # hist_c_after_switch = jnp.concatenate(hist_c_after_switch)
    # print(hist_hippo_before_switch.shape, hist_hippo_exploring.shape, hist_hippo_after_switch.shape)
    # print(hist_theta_before_switch.shape, hist_theta_exploring.shape, hist_theta_after_switch.shape)
    # print(hist_c_before_switch.shape, hist_c_exploring.shape, hist_c_after_switch.shape)

    return hist_hippo_sample_start_mid, hist_theta_sample_start_mid, hist_hippo_sample_mid_goal, hist_theta_sample_mid_goal


def plot_scatter(axis, hist_before_switch, hist_after_switch, mid_of_interest, 
                    hist_c_before_switch, hist_c_after_switch, mid_c_of_interest, 
                    point_number, point_size, title):
    axis.scatter(*(hist_before_switch[:point_number].transpose(1,0)), \
                        s=point_size, c=hist_c_before_switch[:point_number], cmap='Reds')
    # ax_hippo.scatter(*(hist_hippo_exploring[:point_number].transpose(1,0)), \
    #                  s=point_size, c=hist_c_exploring[:point_number], cmap='Purples')
    axis.scatter(*(hist_after_switch[:point_number].transpose(1,0)), \
                        s=point_size, c=hist_c_after_switch[:point_number], cmap='Blues')
    axis.scatter(*(mid_of_interest[:point_number].transpose(1,0)), \
                        s=point_size, c=mid_c_of_interest[:point_number], cmap='Greens')
    axis.set_title(title)

def plot_centroid(ax_hippo, ax_theta, latent_centroid_hippo, latent_centroid_theta, dye_index, dye_color):
    for j in range(len(latent_centroid_hippo)):
        color = 'grey'
        if j == dye_index:
            color = dye_color
        ax_hippo.plot(*(latent_centroid_hippo[j].transpose(1,0)),c=color,linewidth=5)
        ax_theta.plot(*(latent_centroid_theta[j].transpose(1,0)),c=color,linewidth=5)



def cal_plot_policy_map(args, policy_map_switch, policy_logit_map_switch):
    X,Y = jnp.meshgrid(jnp.arange(args.width), jnp.arange(args.height))
    policy_map_all_mean = [[jnp.nanmean(policy_map_switch[j][i],axis=0) for i in range(len(policy_map_switch[j]))] for j in range(len(policy_map_switch))]
    fig, axes = plt.subplots(len(policy_map_switch),len(policy_map_switch[0]), figsize=(40,6))
    for i in range(len(policy_map_switch)):
        # print(items[i])
        for j in range(len(policy_map_switch[0])):
            # print(jitems[j])
            # print(policy_map_all_mean[i][j][...,0])
            # print(policy_map_all_mean[i][j][...,1])
            # print(policy_map_all_mean[i][j][...,2])
            # print(policy_map_all_mean[i][j][...,3])
            # policy_map = jnp.argmax(policy_map_all_mean[i][j],-1,keepdims=True)
            # print(policy_map[...,0])
            # vector_policy_map = jnp.where(policy_map==0, jnp.array([-1,0]),
            #                         jnp.where(policy_map==1, jnp.array([0,1]),
            #                                 jnp.where(policy_map==2, jnp.array([1,0]),
            #                                         jnp.where(policy_map==3, jnp.array([0,-1]),jnp.array([0,0])))))
            # print('policy_map_all_mean[i][j]',policy_map_all_mean[i][j].transpose(2,0,1))
            policy_map_to_plot = policy_map_all_mean[i][j].transpose(1,0,2)
            # print('policy_map_to_plot',policy_map_to_plot.transpose(2,0,1))
            modulus = np.sqrt(policy_map_to_plot[...,0]**2+policy_map_to_plot[...,1]**2)
            angle_map = policy_map_to_plot/modulus.reshape(args.height,args.width,1)/2
            axes[j].quiver(X,Y,angle_map[...,0],angle_map[...,1], modulus, scale=0.01, scale_units='dots',
                           headwidth=10,headlength=10,headaxislength=5,width=0.01,pivot='mid',
                           )
    plt.title('policy map')
    plt.savefig('./figures/'+args.prefix+'/policy_map.png')
    print('save policy map to '+'./figures/'+args.prefix+'/policy_map.png')
    plt.close()
    fig, axes = plt.subplots(len(policy_logit_map_switch),len(policy_logit_map_switch[0]), figsize=(40,6))
    M0 = jnp.zeros((args.width, args.height))
    M1 = jnp.ones((args.width, args.height))*0.5
    for i in range(len(policy_logit_map_switch)):
        for j in range(len(policy_logit_map_switch[0])):
            logit_hist = jnp.sum(policy_logit_map_switch[i][j].transpose(3,0,2,1),axis=(1))
            print(policy_logit_map_switch[i][j])
            axes[j].quiver(X,Y,-M1,M0, logit_hist[0], scale=0.02, scale_units='dots',pivot='tail')
            axes[j].quiver(X,Y,M0,M1, logit_hist[1], scale=0.02, scale_units='dots',pivot='tail')
            axes[j].quiver(X,Y,M1,M0, logit_hist[2], scale=0.02, scale_units='dots',pivot='tail')
            axes[j].quiver(X,Y,M0,-M1, logit_hist[3], scale=0.02, scale_units='dots',pivot='tail')
            axes[j].set_xlim(-0.5, args.width-0.5)
            axes[j].set_ylim(-0.5, args.height-0.5)
            # axes[j].set_xticks(jnp.arange(len(logit_hist)))
            # axes[j].set_xticklabels(['0','1','2','3'])
    plt.title('policy logit map')
    plt.savefig('./figures/'+args.prefix+'/policy_logit_map.png')
    print('save policy logit map to '+'./figures/'+args.prefix+'/policy_logit_map.png')
    plt.close()

    ck0_filter = jnp.zeros((args.width, args.height, args.n_action))
    ck1_filter = jnp.zeros((args.width, args.height, args.n_action))
    ck0_x = env.pseudo_reward_list[args.pseudo_reward_idx][0][0]
    ck0_y = env.pseudo_reward_list[args.pseudo_reward_idx][0][1]
    ck1_x = env.pseudo_reward_list[args.pseudo_reward_idx][1][0]
    ck1_y = env.pseudo_reward_list[args.pseudo_reward_idx][1][1]
    ck0_filter = ck0_filter.at[:ck0_x,:,2].set(1)
    ck0_filter = ck0_filter.at[ck0_x+1:,:,0].set(1)
    ck0_filter = ck0_filter.at[:,:ck0_y,1].set(1)
    ck0_filter = ck0_filter.at[:,ck0_y+1:,3].set(1)
    ck1_filter = ck1_filter.at[:ck1_x,:,2].set(1)
    ck1_filter = ck1_filter.at[ck1_x+1:,:,0].set(1)
    ck1_filter = ck1_filter.at[:,:ck1_y,1].set(1)
    ck1_filter = ck1_filter.at[:,ck1_y+1:,3].set(1)
    angle0_filter = jnp.zeros((args.width, args.height, 2))
    angle0_filter = jnp.where(ck0_filter[...,0:1],angle0_filter+jnp.array([-1,0]),angle0_filter)
    angle0_filter = jnp.where(ck0_filter[...,1:2],angle0_filter+jnp.array([0,1]),angle0_filter)
    angle0_filter = jnp.where(ck0_filter[...,2:3],angle0_filter+jnp.array([1,0]),angle0_filter)
    angle0_filter = jnp.where(ck0_filter[...,3:4],angle0_filter+jnp.array([0,-1]),angle0_filter)
    angle1_filter = jnp.zeros((args.width, args.height, 2))
    angle1_filter = jnp.where(ck1_filter[...,0:1],angle1_filter+jnp.array([-1,0]),angle1_filter)
    angle1_filter = jnp.where(ck1_filter[...,1:2],angle1_filter+jnp.array([0,1]),angle1_filter)
    angle1_filter = jnp.where(ck1_filter[...,2:3],angle1_filter+jnp.array([1,0]),angle1_filter)
    angle1_filter = jnp.where(ck1_filter[...,3:4],angle1_filter+jnp.array([0,-1]),angle1_filter)
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    X,Y = jnp.meshgrid(jnp.arange(args.width), jnp.arange(args.height))
    modulus = jnp.ones((args.width, args.height))*0.5
    axes[0].quiver(X,Y,angle0_filter[...,0].transpose(),angle0_filter[...,1].transpose(), modulus, scale=0.02, scale_units='dots',
                    headwidth=10,headlength=10,headaxislength=5,width=0.02,pivot='mid',
                    )
    axes[1].quiver(X,Y,angle1_filter[...,0].transpose(),angle1_filter[...,1].transpose(), modulus, scale=0.02, scale_units='dots',
                    headwidth=10,headlength=10,headaxislength=5,width=0.02,pivot='mid',
                    )
    plt.savefig('./figures/angle_map.png')
    plt.close()
    fig, axes = plt.subplots(len(policy_logit_map_switch),len(policy_logit_map_switch[0]), figsize=(40,6))
    for i in range(len(policy_logit_map_switch)):
        for j in range(len(policy_logit_map_switch[0])):
            # print(jitems[j])
            print('policy_logit_map_switch[i][j]')
            print(policy_logit_map_switch[i][j].mean(0))
            print(ck0_filter)
            ck0_idx = jnp.sum(policy_map_all_mean[i][j]*angle0_filter)
            ck1_idx = jnp.sum(policy_map_all_mean[i][j]*angle1_filter)
            print('ck0_idx',ck0_idx)
            print('ck1_idx',ck1_idx)
            axes[j].bar(jnp.arange(2),[ck0_idx,ck1_idx],tick_label=['ck0','ck1'])
    plt.title('policy centrality map')
    plt.savefig('./figures/'+args.prefix+'/policy_centrality_map.png')
    print('save policy centrality map to '+'./figures/'+args.prefix+'/policy_centrality_map.png')
    plt.close()
    #TODO:filter for policy map
    return 

def cal_plot_value_map(args, value_map_switch):

    # value_map_all_mean = [[jnp.nanmean(value_map_switch[j][i],axis=0) for i in range(len(value_map_switch[j]))] for j in range(len(value_map_switch))]
    value_map_all_mean = 0
    # items = ['mid','goal']
    # jitems = ['00','1','2','3','4','5']
    # for replay_loc in range(len(items)):
    #     print(items[replay_loc])
    #     for i in range(len(policy_map_switch[replay_loc])):
    #         print(jitems[i])
    #         print(policy_map_switch[replay_loc][i].transpose(0,3,1,2))
    #         print(policy_map_all_mean[replay_loc][i].transpose(2,0,1))
    #     print()
    # theta_cor_all_mean = [[jnp.nanmean(jnp.array(theta_cor_all[j][i]),axis=0) for i in range(len(theta_cor_all[j]))] for j in range(len(theta_cor_all))]
    items = ['mid']
    jitems = ['0','1','2','3','4']
    ck0_idx, ck1_idx, value_advantage, delta_value_advantage, advantage_list = cal_plot_value_advantage_and_delta(args, value_map_switch, items, jitems)

    # fig, axes = plt.subplots(len(value_map_switch),len(value_map_switch[0]), figsize=(20,20))
    # for i in range(len(value_map_switch)):
    #     # print(items[i])
    #     for j in range(len(value_map_switch[0])):
    #         # print(jitems[j])
    #         axes[j].imshow(value_map_all_mean[i][j][::-1],cmap='Oranges')
    #         # axes[j].set_title('ck0:{:.2f}'.format(ck0_idx[i][j])+' ck1:{:.2f}'.format(ck1_idx[i][j]))
    #         axes[j].set_xticks([])
    #         axes[j].set_yticks([])

            # print('value_map_all_mean:',value_map_all_mean[i][j])
            # print(jnp.nansum(value_map_all_mean[i][j]*ck0_filter))

    # print(value_map_all_mean[0])
    # print(value_map_all_mean[1])
    # plt.colorbar()
    # plt.show()
    # fig.suptitle('value map')
    # plt.savefig('./figures/'+args.prefix+'/value_map.png')
    # print('save value map to '+'./figures/'+args.prefix+'/value_map.png')
    # plt.close()

    # delta advantage vs outer difference
    # mid_value_map = value_map_all_mean[0]
    # jnp.save('./figures/'+args.prefix+'/mid_value_map.npy',mid_value_map)
    outer_diff_mean, outer_diff_std = cal_plot_outer_difference(args, jitems, advantage_list, value_map_switch)
    cal_plot_path_value(args, value_map_switch, jitems)

    # print('ck0_idx',ck0_idx)
    # print('ck1_idx',ck1_idx)

    # fig, axes = plt.subplots(2,2)
    # axes[0][0].imshow(theta_cor_all_mean[0][0][::-1],cmap='viridis')
    # axes[0][0].set_title('0-0')
    # axes[0][1].imshow(theta_cor_all_mean[0][1][::-1],cmap='viridis')
    # axes[0][1].set_title('0-1')
    # axes[1][0].imshow(theta_cor_all_mean[1][0][::-1],cmap='viridis')
    # axes[1][0].set_title('1-0')
    # axes[1][1].imshow(theta_cor_all_mean[1][1][::-1],cmap='viridis')
    # axes[1][1].set_title('1-1')
    # print(theta_cor_all_mean[0])
    # print(theta_cor_all_mean[1])
    # fig.suptitle('theta correlation')
    # plt.savefig('./figures/'+args.prefix+'/theta_cor.png')
    # plt.show()

    return value_map_all_mean, value_advantage, delta_value_advantage, \
            outer_diff_mean, outer_diff_std


def cal_plot_value_advantage_and_delta(args, value_map_switch, items, jitems):

    # x, y = jnp.meshgrid(jnp.arange(args.width), jnp.arange(args.height))
    # xy = jnp.stack([x, y], axis=-1).reshape(args.width*args.height, 2)
    # ck0_filter = path_int.generate_place_cell(xy,config.sigma,env.pseudo_reward_list[args.pseudo_reward_idx][0].reshape(1,2)).reshape(args.height,args.width).transpose(1,0)
    # ck1_filter = path_int.generate_place_cell(xy,config.sigma,env.pseudo_reward_list[args.pseudo_reward_idx][1].reshape(1,2)).reshape(args.height,args.width).transpose(1,0)
    # # show ck0 and ck1 filter
    # fig, axis = plt.subplots(1,2)
    # # axis[0][0].imshow(value_map_all_mean[0][0],cmap='Oranges')
    # # axis[0][1].imshow(value_map_all_mean[0][1],cmap='Oranges')
    # axis[0].imshow(ck0_filter,cmap='Oranges')
    # axis[1].imshow(ck1_filter,cmap='Oranges')
    # plt.savefig('./figures/value_map.png')
    # plt.close()
    # # print(ck0_filter)
    # # print(ck1_filter)
    # # ck0_idx = [[jnp.sum(value_map_all_mean[j][i]*ck0_filter) for i in range(len(value_map_all_mean[j]))] for j in range(len(value_map_all_mean))]
    # # ck1_idx = [[jnp.sum(value_map_all_mean[j][i]*ck1_filter) for i in range(len(value_map_all_mean[j]))] for j in range(len(value_map_all_mean))]
    # ck0_idx_mean = jnp.zeros((len(value_map_switch),len(value_map_switch[0])))
    # ck1_idx_mean = jnp.zeros((len(value_map_switch),len(value_map_switch[0])))
    # ck0_idx_std = jnp.zeros((len(value_map_switch),len(value_map_switch[0])))
    # ck1_idx_std = jnp.zeros((len(value_map_switch),len(value_map_switch[0])))
    # advantage_list = [[] for i in range(len(value_map_switch))]
    # for i in range(len(value_map_switch)):
    #     # print(items[i])
    #     for j in range(len(value_map_switch[0])):
    #         # print(jitems[j])
    #         # print(j,len(value_map_switch[i][j]))
    #         ck0_idx = jnp.nansum(value_map_switch[i][j]*ck0_filter,axis=(1,2))
    #         ck1_idx = jnp.nansum(value_map_switch[i][j]*ck1_filter,axis=(1,2))
    #         advantage_list[i].append(ck1_idx-ck0_idx)
    #         ck0_idx_mean = ck0_idx_mean.at[i,j].set(ck0_idx.mean())
    #         ck1_idx_mean = ck1_idx_mean.at[i,j].set(ck1_idx.mean())
    #         ck0_idx_std = ck0_idx_std.at[i,j].set(ck0_idx.std())
    #         ck1_idx_std = ck1_idx_std.at[i,j].set(ck1_idx.std())

    # value_advantage_mean = (jnp.array(ck1_idx_mean) - jnp.array(ck0_idx_mean))
    # value_advantage_std = (jnp.array(ck0_idx_std) + jnp.array(ck1_idx_std))/jnp.sqrt(len(advantage_list[0][0]))
    # stablility_ANOVA_result = ar_ANOVA_analysis(jnp.array(advantage_list[0][1:]))
    # print('stability ANOVA analysis',stablility_ANOVA_result)
    # change_ANOVA_result = ar_ANOVA_analysis(jnp.array(advantage_list[0][:2]))
    # print('change ANOVA analysis',change_ANOVA_result)
    # 1[mid]*5[01,011,0111,01111,00]
    scale = 1e3
    ck0_idx_mean, ck1_idx_mean, value_advantage_mean, value_advantage_std, advantage_list = pickle.load(open('./figures/'+args.prefix+'/value_advantage.pkl', 'rb'))
    fig, axis = plt.subplots(1,len(items), figsize=(6,5))
    axis.errorbar(jnp.arange(len(value_map_switch[0])),value_advantage_mean[0] * scale,
        yerr=value_advantage_std[0] * scale, c='turquoise', marker='o', markersize=10, linewidth=4, capsize=8, capthick=4, fmt='-')
    # axis.scatter(jnp.arange(len(value_map_switch[0])),value_advantage_mean[i], marker='o', s=200, c='dodgerblue')
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['bottom'].set_linewidth(3)
    axis.spines['left'].set_linewidth(3)
    axis.set_xticks(jnp.arange(len(jitems)),jitems,fontdict={'fontsize':40})
    axis.set_yticks([-2.5,0,2.5,5],[-2.5,0,2.5,5],fontdict={'fontsize':40})
    axis.tick_params(axis='both', which='major', labelsize=40, width=3, length=10)
    # axis.set_ylabel('Advantage', fontsize=20)
    axis.axhline(y=0, color='cyan', linestyle='dashed', linewidth=3)
    # plt.show()
    # fig.suptitle('value advantage')
    fig.subplots_adjust(top=0.9,bottom=0.3,left=0.3,right=0.9)
    plt.savefig('./figures/'+args.prefix+'/value_advantage.png')
    print('save value advantage to '+'./figures/'+args.prefix+'/value_advantage.png')
    # jnp.save('./figures/'+args.prefix+'/value_advantage_0.npy',value_advantage)
    pickle.dump([ck0_idx_mean, ck1_idx_mean, value_advantage_mean, value_advantage_std, advantage_list], open('./figures/'+args.prefix+'/value_advantage.pkl', 'wb'))
    plt.close()

    delta_value_advantage = jnp.diff(value_advantage_mean,axis=1)
    fig, axis = plt.subplots(1,len(items), figsize=(6,5))
    for i in range(len(delta_value_advantage)):
        axis.bar(jnp.arange(delta_value_advantage.shape[1]),delta_value_advantage[i],
        tick_label=['1','2','3','4'])
    plt.title('delta value advantage')
    plt.savefig('./figures/'+args.prefix+'/delta_value_advantage.png')
    print('save delta value advantage to '+'./figures/'+args.prefix+'/delta_value_advantage.png')
    plt.close()

    return ck0_idx_mean, ck1_idx_mean, value_advantage_mean, delta_value_advantage, advantage_list

def list_ANOVA_analysis(advantage_list):
    all_data = jnp.concatenate(advantage_list,0)
    group_mean = jnp.array([x.mean() for x in advantage_list])
    group_std = jnp.array([x.std() for x in advantage_list])
    total_mean = all_data.mean()
    group_length = jnp.array([len(x) for x in advantage_list])
    print('n:',group_length)
    S_T = jnp.sum((all_data-total_mean)**2)
    print("S_T",S_T)
    S_A = jnp.sum(group_length*jnp.square(group_mean-total_mean))
    print("S_A",S_A)
    S_E = jnp.sum(jnp.array([advantage_list[i].var()*(len(advantage_list[i])) for i in range(len(advantage_list))]))
    print("S_E",S_E)
    df_A = len(group_mean)-1
    print("df_A",df_A)
    df_E = len(all_data)-len(group_mean)
    print("df_E",df_E)
    MS_A = S_A/df_A
    MS_E = S_E/df_E
    F = MS_A/MS_E
    print(F)
    print(scipy.stats.f.sf(F, df_A, df_E))
    reject_null_hypothesis = F>scipy.stats.f.isf(0.001, df_A, df_E)
    return reject_null_hypothesis

def ar_ANOVA_analysis(advantage_ar):
    all_data = advantage_ar.reshape(-1)
    print('n:',advantage_ar.shape[1])
    group_mean = jnp.array([x.mean() for x in advantage_ar])
    # print('group_mean',group_mean)
    group_std = jnp.array([x.std() for x in advantage_ar])
    # print('group_std',group_std)
    total_mean = all_data.mean()
    # print('total_mean',total_mean)
    group_length = jnp.array([len(x) for x in advantage_ar])
    # print('group_length',group_length)
    S_T = jnp.sum((all_data-total_mean)**2)
    print("S_T",S_T)
    S_A = jnp.sum(group_length*jnp.square(group_mean-total_mean))
    print("S_A",S_A)
    # advantage_list = advantage_list[0][1:]
    S_E = jnp.sum(jnp.array([advantage_ar[i].var()*(len(advantage_ar[i])) for i in range(len(advantage_ar))]))
    print("S_E",S_E)
    df_A = len(group_mean)-1
    print("df_A",df_A)
    df_E = len(all_data)-len(group_mean)
    print("df_E",df_E)
    MS_A = S_A/df_A
    MS_E = S_E/df_E
    F = MS_A/MS_E
    print(F)
    print(scipy.stats.f.sf(F, df_A, df_E))
    reject_null_hypothesis = F>scipy.stats.f.isf(0.001, df_A, df_E)
    return reject_null_hypothesis

def cal_plot_outer_difference(args, jitems, advantage_list, value_map_switch):
    scale = 1e3
    n = len(advantage_list[0][0])
    n_sqrt = jnp.sqrt(len(advantage_list[0][0]))

    # x, y = jnp.meshgrid(jnp.arange(args.width), jnp.arange(args.height))
    # xy = jnp.stack([x, y], axis=-1).reshape(args.width*args.height, 2)
    # all_filters = jnp.array([path_int.generate_place_cell(xy,config.sigma,center.reshape(1,2)).reshape(args.height,args.width).transpose(1,0) for center in xy])

    # advantage_ar = jnp.array(advantage_list[0])
    # delta_advantage_ar = jnp.diff(advantage_ar,axis=0)
    # delta_advantage_mean = jnp.mean(delta_advantage_ar,1)
    # delta_advantage_std = jnp.std(delta_advantage_ar,1)/n_sqrt

   
    # convoluted_value = jnp.nansum(jnp.array(value_map_switch[0]).reshape(5, len(advantage_list[0][0]), 1, args.width, args.height)*all_filters, axis=(3,4))
    # # 5 * 118 * 25
    # out_diff = []
    # outer_diff_mean = jnp.zeros((len(jitems)-1,))
    # outer_diff_std = jnp.zeros((len(jitems)-1,))
    # for j in range(len(jitems)-1):
    #     value_map_vector_outer_difference = convoluted_value[j+1].reshape(n,-1,1)-convoluted_value[j].reshape(n,1,-1)
    #     out_diff.append(value_map_vector_outer_difference.reshape(-1))
    #     outer_diff_mean = outer_diff_mean.at[j].set(jnp.mean(value_map_vector_outer_difference))
    #     outer_diff_std = outer_diff_std.at[j].set(jnp.std(value_map_vector_outer_difference))

    delta_advantage_mean, delta_advantage_std, outer_diff_mean, outer_diff_std = \
        pickle.load(open('./figures/'+args.prefix+'/delta_advantage_vs_outer_diff.pkl', 'rb'))
    fig, axes = plt.subplots(1,1, figsize=(6,5))
    

    
    # outer_diff_lower_bound = outer_diff_mean-3*outer_diff_std/n_sqrt
    # outer_diff_upper_bound = outer_diff_mean+3*outer_diff_std/n_sqrt
    # axes.fill_between(jnp.arange(len(jitems)-1), outer_diff_lower_bound, outer_diff_upper_bound, color='lightgray', alpha=1)
    outer_diff_lower_bound = outer_diff_mean-outer_diff_std/n_sqrt
    outer_diff_upper_bound = outer_diff_mean+outer_diff_std/n_sqrt
    axes.fill_between(jnp.arange(len(jitems)-1), outer_diff_lower_bound*scale, outer_diff_upper_bound*scale, color='lightblue', alpha=0.5)
    # outlier_ANOVA_result = outlier_ANOVA_analysis(out_diff, delta_advantage_ar, n=args.width*args.height)
    print(delta_advantage_mean.shape)
    axes.errorbar(jnp.arange(4), delta_advantage_mean*scale, delta_advantage_std*scale, c='limegreen', 
                  marker='o', markersize=10, linewidth=4, capsize=8, capthick=4, fmt='-')
    axes.plot(outer_diff_mean*scale, c='deepskyblue', marker='o', markersize=10, linewidth=3)
    axes.set_xticks(jnp.arange(len(jitems)-1),jitems[1:],fontdict={'fontsize':40})
    axes.set_yticks([0,2,4,6,8],[0,2,4,6,8],fontdict={'fontsize':40})
    # axes.set_ylabel('Delta advantage', fontsize=20)
    # axes.set_xlabel('Times of meeting checkpoint 2 repeatedly', fontsize=20)
    axes.spines['top'].set_color('none')
    axes.spines['right'].set_color('none')
    axes.spines['bottom'].set_linewidth(3)
    axes.spines['left'].set_linewidth(3)
    axes.tick_params(axis='both', which='major', labelsize=40, width=3, length=10)
    fig.subplots_adjust(top=0.9,bottom=0.3,left=0.2,right=0.8)
    # plt.title('delta_advantage_vs_outer_diff')
    plt.savefig('./figures/'+args.prefix+'/delta_advantage_vs_outer_diff.png')
    pickle.dump([delta_advantage_mean, delta_advantage_std, outer_diff_mean, outer_diff_std], open('./figures/'+args.prefix+'/delta_advantage_vs_outer_diff.pkl', 'wb'))
    return outer_diff_mean, outer_diff_std

def outlier_ANOVA_analysis(out_diff, delta_advantage_ar, n):
    reject_null_hypothesis = jnp.zeros((len(out_diff),))
    for j in range(len(out_diff)):
        print('n_out_diff:',len(out_diff[j]),'n_delta_advantage_ar:',len(delta_advantage_ar[j]))
        all_data = jnp.concatenate([out_diff[j], delta_advantage_ar[j]],0)
        group_mean = jnp.array([out_diff[j].mean(), delta_advantage_ar[j].mean()])
        total_mean = all_data.mean()
        group_length = jnp.array([len(out_diff[j]), len(delta_advantage_ar[j])])
        S_T = jnp.sum((all_data-total_mean)**2)
        S_A = jnp.sum(group_length*jnp.square(group_mean-total_mean))
        S_E = S_T - S_A
        df_A = len(group_mean)-1
        df_E = len(all_data)-len(group_mean)
        MS_A = S_A/df_A
        MS_E = S_E/df_E
        F = MS_A/MS_E
        print(F)
        print(scipy.stats.f.sf(F, df_A, df_E))
        reject_null_hypothesis = reject_null_hypothesis.at[j].set(F>scipy.stats.f.isf(0.01, df_A, df_E))
    print('outlier ANOVA analysis',reject_null_hypothesis)


    # t = (delta_value_advantage-outer_diff_mean)/(outer_diff_std/jnp.sqrt(n))
    # print('t',t)
    # p = scipy.stats.t.sf(t, n-1)
    # print('p',p)
    # reject_null_hypothesis = (t>scipy.stats.t.isf(0.01, n-1)) | (t<scipy.stats.t.isf(0.99, n-1))
    # print('outlier ANOVA analysis',reject_null_hypothesis)
    return reject_null_hypothesis


def cal_plot_path_value(args, value_map_switch, jitems):
    # path value
    scale = 1e2
    # segment_list = ['s0','s1', '0g', '1g']
    # grid_s_c0, grid_s_c1, grid_c0_g, grid_c1_g = \
    #     [jnp.zeros((args.width, args.height)) for _ in range(4)]
    # ck0_x = env.pseudo_reward_list[args.pseudo_reward_idx][0][0]
    # ck0_y = env.pseudo_reward_list[args.pseudo_reward_idx][0][1]
    # ck1_x = env.pseudo_reward_list[args.pseudo_reward_idx][1][0]
    # ck1_y = env.pseudo_reward_list[args.pseudo_reward_idx][1][1]
    # ck0_x_g_lb = ck0_x-1
    # ck0_y_g_lb = ck0_y-1
    # ck1_x_g_lb = ck1_x-1
    # ck1_y_g_lb = ck1_y-1
    # v1 = 0
    # grid_s_c0 = grid_s_c0.at[:ck0_x+1, :ck0_y+1].set(1)
    # grid_s_c0 = grid_s_c0.at[ck0_x+1, :ck0_y+2].set(v1)
    # grid_s_c0 = grid_s_c0.at[:ck0_x+2, ck0_y+1].set(v1)

    # grid_s_c1 = grid_s_c1.at[:ck1_x+1, :ck1_y+1].set(1)
    # grid_s_c1 = grid_s_c1.at[ck1_x+1, :ck1_y+2].set(v1)
    # grid_s_c1 = grid_s_c1.at[:ck1_x+2, ck1_y+1].set(v1)

    # grid_c0_g = grid_c0_g.at[ck0_x_g_lb, ck0_y_g_lb:].set(v1)
    # grid_c0_g = grid_c0_g.at[ck0_x_g_lb:, ck0_y_g_lb].set(v1)
    # grid_c0_g = grid_c0_g.at[ck0_x:, ck0_y:].set(1)

    # grid_c1_g = grid_c1_g.at[ck1_x_g_lb, ck1_y_g_lb:].set(v1)
    # grid_c1_g = grid_c1_g.at[ck1_x_g_lb:, ck1_y_g_lb].set(v1)
    # grid_c1_g = grid_c1_g.at[ck1_x:, ck1_y:].set(1)

    # grid_all = jnp.stack([grid_s_c0, grid_s_c1, grid_c0_g, grid_c1_g], axis=0).reshape(1,1,len(segment_list),args.width,args.height)
    # grid_all = grid_all.at[...,0,0].set(0)
    # mid_value_map_all = jnp.array(value_map_switch).reshape(len(jitems),-1,1,args.width,args.height)
    # mid_value_map_all = jnp.where(jnp.isnan(mid_value_map_all), 0, mid_value_map_all)
    # # mid_value_map_all = mid_value_map_all.at[...,0,0].set(0)
    # # print(mid_value_map_all)
    # path_value = jnp.sum(grid_all*mid_value_map_all, axis=(3,4))/jnp.sum(grid_all*mid_value_map_all, axis=(2,3,4)).reshape(len(jitems),-1,1)
    # # [5, 118, 4]
    # # print(path_value)
    # cons_advantage = path_value[:,:,1] - path_value[:,:,0]
    # cons_advantage_mean = cons_advantage.mean(1)
    # cons_advantage_std = cons_advantage.std(1)/jnp.sqrt(len(cons_advantage[1]))
    # stablility_ANOVA_result = ar_ANOVA_analysis(cons_advantage[1:])
    # print('cons advantage stability ANOVA analysis',stablility_ANOVA_result)
    # change_ANOVA_result = ar_ANOVA_analysis(cons_advantage[:2])
    # print('plan advantage change ANOVA analysis',change_ANOVA_result)

    # plan_advantage = path_value[:,:,3] - path_value[:,:,2]
    # plan_advantage_mean = plan_advantage.mean(1)
    # plan_advantage_std = plan_advantage.std(1)/jnp.sqrt(len(plan_advantage[1]))
    # stablility_ANOVA_result = ar_ANOVA_analysis(plan_advantage[1:])
    # print('plan advantage stability ANOVA analysis',stablility_ANOVA_result)
    # change_ANOVA_result = ar_ANOVA_analysis(plan_advantage[:2])
    # print('plan advantage change ANOVA analysis',change_ANOVA_result)


    cons_advantage_mean, cons_advantage_std, plan_advantage_mean, plan_advantage_std = \
        pickle.load(open('./figures/'+args.prefix+'/path_value.pkl', 'rb'))
    # path_value = jnp.where(jnp.isnan(path_value), 0, path_value)
    fig, axes = plt.subplots(1,2, figsize=(16,6))
    # axes.plot(path_value[:,0], c='red')
    # axes.plot(path_value[:,1], c='yellow')
    # axes.plot(path_value[:,2], c='blue')
    # axes.plot(path_value[:,3], c='green')
    axes[0].errorbar(jnp.arange(len(jitems)), cons_advantage_mean*scale, cons_advantage_std*scale, c='orange',
        marker='o', markersize=10, linewidth=4, capsize=8, capthick=4, fmt='-')
    axes[0].set_xticks(jnp.arange(len(jitems)),jitems,fontdict={'fontsize':40})
    axes[0].set_yticks(jnp.arange(0,4),jnp.arange(0,4),fontdict={'fontsize':40})
    # axes[0].set_xlabel('Times of meeting checkpoint 2 repeatedly')
    # axes[0].set_ylabel('Consolidation Advantage', fontsize=20)
    axes[0].spines['top'].set_color('none')
    axes[0].spines['right'].set_color('none')
    axes[0].spines['bottom'].set_linewidth(3)
    axes[0].spines['left'].set_linewidth(3)
    axes[0].tick_params(axis='both', which='major', labelsize=40, width=3, length=10)
    axes[0].axhline(y=0, color='moccasin', linestyle='dashed', linewidth=3)
    axes[1].errorbar(jnp.arange(len(jitems)), plan_advantage_mean*scale, plan_advantage_std*scale, c='forestgreen',
        marker='o', markersize=10, linewidth=4, capsize=8, capthick=4, fmt='-')
    axes[1].set_xticks(jnp.arange(len(jitems)),jitems,fontdict={'fontsize':40})
    axes[1].set_yticks(jnp.arange(-3,3),jnp.arange(-3,3),fontdict={'fontsize':40})
    # axes[1].set_xlabel('Times of meeting checkpoint 2 repeatedly')
    # axes[1].set_ylabel('Planning Advantage', fontsize=20)
    axes[1].spines['top'].set_color('none')
    axes[1].spines['right'].set_color('none')
    axes[1].spines['bottom'].set_linewidth(3)
    axes[1].spines['left'].set_linewidth(3)
    axes[1].tick_params(axis='both', which='major', labelsize=40, width=3, length=10)
    axes[1].axhline(y=0, color='palegreen', linestyle='dashed', linewidth=3)
    fig.subplots_adjust(wspace=0.7,hspace=0.7,bottom=0.2)
    # fig.tight_layout()
    
    # plt.title('path_value')
    plt.savefig('./figures/'+args.prefix+'/path_value.png')
    print('save path value to '+'./figures/'+args.prefix+'/path_value_advantage.png')
    pickle.dump([cons_advantage_mean, cons_advantage_std, plan_advantage_mean, plan_advantage_std], open('./figures/'+args.prefix+'/path_value.pkl', 'wb'))
    plt.close()
    return

def scan_value_map(args, n, origin_env_state, origin_buffer_state, origin_running_encoder_state, origin_running_hippo_state, 
                         origin_running_hippo_std_state, origin_running_policy_state,
                         subkey, origin_hippo_hidden, origin_theta):
    env_state = origin_env_state.copy()
    buffer_state = origin_buffer_state
    running_encoder_state = origin_running_encoder_state
    running_hippo_state = origin_running_hippo_state
    running_hippo_std_state = origin_running_hippo_std_state
    running_policy_state = origin_running_policy_state
    hippo_hidden = origin_hippo_hidden.copy()
    theta = origin_theta.copy()
    # action_dict = {0:jnp.array([-1,0]),1:jnp.array([0,1]),2:jnp.array([1,0]),3:jnp.array([0,-1])}
    action_dict = jnp.array([[-1,0],[0,1],[1,0],[0,-1]])
    """
    Calculate the state map of the environment at the current step.
    """
    # {'grid': grid, 'current_pos': current_pos,
    # 'goal_pos': goal_pos, 'reward_center':reward_center,
    # 'checked': checked, 'step_count':step_count,
    # 'move_to_start':move_to_start}
    # next_pos = jnp.where(actions == 0, current_pos - jnp.array([1, 0], dtype=jnp.int8),
        #       jnp.where(actions == 1, current_pos + jnp.array([0, 1], dtype=jnp.int8),
    #            jnp.where(actions == 2, current_pos + jnp.array([1, 0], dtype=jnp.int8),
#                      jnp.where(actions == 3, current_pos - jnp.array([0, 1], dtype=jnp.int8),
#                                current_pos))))
    action_list = jax.random.randint(subkey, (100,1), 0, args.n_action, dtype=jnp.int8)             
    action_list = jnp.concatenate([jnp.ones((3,1),dtype=jnp.int8), jnp.ones((3,1),dtype=jnp.int8)*2, action_list],0)
    action_list = jnp.array(action_list).reshape(-1,1,1).repeat(args.n_agents, axis=1)
    # print(action_list.shape)
    # Cancel all rewards

    value_map = jnp.zeros((args.height, args.width))
    count_map = jnp.zeros((args.height, args.width))
    theta_map = jnp.zeros((args.height, args.width, args.theta_hidden_size))
    policy_map = jnp.zeros((args.height, args.width, 2))
    policy_logit_map = jnp.zeros((args.height, args.width, 4))
    # total_checked = env_state['total_checked']
    for a in action_list:
        if jnp.all(env_state['current_pos'][n] == jnp.zeros_like(env_state['current_pos'][n]), -1):
            env_state['checked'] = 1
        # env_state['move_to_start'] = 0
        # print(env_state['current_pos'][n])
        env_state, buffer_state, actions, hippo_hidden, theta, rewards, done, replayed_hippo_theta_output, first_hippo_theta_output, hipp_info, value, policy \
            = train.Igata_step(env_state, buffer_state, running_encoder_state, running_hippo_state, 
                         running_hippo_std_state, running_policy_state,
                         subkey, a, hippo_hidden, theta,
                         args.n_agents, args.bottleneck_size, args.hidden_size, args.replay_steps, args.height, args.width,
                         args.visual_prob, args.eval_temperature, args.reset_prob, args.noise_scale, args.pseudo_reward,
                         args.block_idx)
        replayed_hippo_history, replayed_theta_history, output_history, hipp_info_history = replayed_hippo_theta_output 
        # 先把所有奖励取消掉，先不算起点终点的奖励
        # print(env_state['current_pos'][n], rewards[n], done[n], env_state['phase'][n], value[n], policy[n])
        # print(env_state['current_pos'][n][0], env_state['current_pos'][n][1])
        value_map = value_map.at[env_state['current_pos'][n][0].item(), 
                                env_state['current_pos'][n][1].item()].add(value[n].item())
        # print(env_state['current_pos'][n],policy[n])
        policy_map = policy_map.at[env_state['current_pos'][n][0].item(),
                                   env_state['current_pos'][n][1].item()].add(action_dict[jnp.argmax(policy[n])].reshape(2))
        # print(policy_map.transpose(2,0,1))
        policy_logit_map = policy_logit_map.at[env_state['current_pos'][n][0].item(),
                                      env_state['current_pos'][n][1].item(),jnp.argmax(policy[n])].add(1)
        # print(policy_logit_map.transpose(2,0,1))
        count_map = count_map.at[env_state['current_pos'][n][0].item(),
                                env_state['current_pos'][n][1].item()].add(1)
        theta_map = theta_map.at[env_state['current_pos'][n][0].item(),
                                env_state['current_pos'][n][1].item()].add(theta[n].reshape(args.theta_hidden_size))
        # n * 1
    # print(value_map)
    # value_list = jnp.array(value_list)
    # print(value_list.shape)
    # hw * 1
    # value_list = jnp.concatenate([value_list[-1:], value_list[:-1]])
    # print(value_list.shape)
    value_map = value_map/(count_map)
    value_map = jnp.where(jnp.isnan(value_map), 0, value_map)
    value_map = value_map/jnp.nansum(value_map)
    policy_map = policy_map/((count_map).reshape(args.height, args.width, 1))
    policy_map = jnp.where(jnp.isnan(policy_map), 0, policy_map)
    # print('policy_map',policy_map.transpose(2,0,1))
    policy_logit_map = policy_logit_map/((count_map).reshape(args.height, args.width, 1))
    policy_logit_map = jnp.where(jnp.isnan(policy_logit_map), 0, policy_logit_map)
    # print('policy_logit_map',policy_logit_map.transpose(2,0,1))
    theta_map = theta_map/((count_map).reshape(args.height, args.width, 1))
    # print('vmap:',value_map)
    # col_to_be_inversed = jnp.array([0,1,0,1,0]).reshape(-1,1).repeat(args.width,1)
    # value_map = jnp.where(col_to_be_inversed == 1, value_map[:,::-1], value_map)
    return value_map, policy_map, theta_map, policy_logit_map


def cal_energy(infos):
    """
    Calculate the normalized energy based in each array of infos.

    Args: `infos` same as the returns of `select_switch_transition`

    Returns: 
        Four energy lists corresponding to aforementioned infos `(00,01,011,0111)` or `(cons_0,cons_1,plan_0,plan_1)`. 
        Every list includes four arrays corresponding to accuracy of hippo, theta, theta_slow and hipp_info.
        Every array is of shape [args.replay_steps+2,] corresponding to different energy at different steps.

    """
    hippos, thetas, theta_slows, hipp_infos, phases = infos

    hippo_activity = []
    theta_activity = []
    theta_slow_activity = []
    hipp_info_activity = []
    for i in range(len(phases)):
        hippo_energy = jnp.linalg.norm(hippos[i], axis=-1).mean(0)
        normalized_hippo_energy = (hippo_energy - hippo_energy.min())/(hippo_energy.max()-hippo_energy.min())
        hippo_activity.append(normalized_hippo_energy)
        theta_energy = jnp.linalg.norm(thetas[i], axis=-1).mean(0)
        normalized_theta_energy = (theta_energy - theta_energy.min())/(theta_energy.max()-theta_energy.min())
        theta_activity.append(normalized_theta_energy)
        theta_slow_energy = jnp.linalg.norm(theta_slows[i], axis=-1).mean(0)
        normalized_theta_slow_energy = (theta_slow_energy - theta_slow_energy.min())/(theta_slow_energy.max()-theta_slow_energy.min())
        theta_slow_activity.append(normalized_theta_slow_energy)
        hipp_info_energy = jnp.linalg.norm(hipp_infos[i], axis=-1).mean(0)
        normalized_hipp_info_energy = (hipp_info_energy - hipp_info_energy.min())/(hipp_info_energy.max()-hipp_info_energy.min())
        hipp_info_activity.append(normalized_hipp_info_energy)
    
    energy = list(zip(hippo_activity, theta_activity, theta_slow_activity, hipp_info_activity))

    return energy

### TODO: to be modified
def cal_plot_hist_phase(args, clf_hippo, clf_theta, hist_hippo, hist_theta, hist_state, reward_n):
    mid_goal_decoding_hippo = [[],[]]
    mid_goal_decoding_theta = [[],[]]
    start_mid_decoding_hippo = [[],[]]
    start_mid_decoding_theta = [[],[]]
    step_threshold = 4
    for n in range(args.n_agents):
        # print(reward_n[n])
        for i in range(len(reward_n[n])-2):
            if reward_n[n][i][-1]==0 and reward_n[n][i+1][-1]==1:
                # mid->goal
                s, e = reward_n[n][i][0], reward_n[n][i+1][0]
                for ph in range(2):
                    if e-s<=step_threshold and hist_state[s,n,5]==hist_state[e,n,5]==ph:
                        pred_phase_hippo = clf_hippo.predict(hist_hippo[s:e,n]).reshape(-1)
                        pred_phase_theta = clf_theta.predict(hist_theta[s:e,n]).reshape(-1)
                        # print('mid-goal', ph, '\n', pred_phase_hippo, '\n', pred_phase_theta)
                        mid_goal_decoding_hippo[ph].append(jnp.where(pred_phase_hippo==ph,1,0))
                        mid_goal_decoding_theta[ph].append(jnp.where(pred_phase_theta==ph,1,0))
            if reward_n[n][i][-1]==1 and reward_n[n][i+1][-1]==0:
                # start->mid
                s, e = reward_n[n][i][0], reward_n[n][i+1][0]
                for ph in range(2):
                    if e-s<=step_threshold and hist_state[s,n,5]==hist_state[e,n,5]==ph:
                        pred_phase_hippo = clf_hippo.predict(hist_hippo[s:e,n]).reshape(-1)
                        pred_phase_theta = clf_theta.predict(hist_theta[s:e,n]).reshape(-1)
                        # print('start-mid', ph, '\n', pred_phase_hippo, '\n', pred_phase_theta)
                        start_mid_decoding_hippo[ph].append(jnp.where(pred_phase_hippo==ph,1,0))
                        start_mid_decoding_theta[ph].append(jnp.where(pred_phase_theta==ph,1,0))
            
    mid_goal_decoding_acc_hippo = [jnp.mean(jnp.array(mid_goal_decoding_hippo[i]),axis=0) for i in range(2)]
    mid_goal_decoding_acc_theta = [jnp.mean(jnp.array(mid_goal_decoding_theta[i]),axis=0) for i in range(2)]
    start_mid_decoding_acc_hippo = [jnp.mean(jnp.array(start_mid_decoding_hippo[i]),axis=0) for i in range(2)]
    start_mid_decoding_acc_theta = [jnp.mean(jnp.array(start_mid_decoding_theta[i]),axis=0) for i in range(2)]
    acc_hippo = [mid_goal_decoding_acc_hippo, start_mid_decoding_acc_hippo]
    acc_theta = [mid_goal_decoding_acc_theta, start_mid_decoding_acc_theta]
    acc = list(zip(acc_hippo, acc_theta))
    titles = ['mid_goal', 'start_mid']
    plot_replay_curve(acc, titles, args.prefix, args.suffix, args.info_type, args.replay_steps, baseline=1/args.pseudo_reward.shape[0])

    return

def cal_plot_replay_curve(args, clfs, hippos, theta_fasts, theta_slows, hipp_infos, phases, trajs, items_to_be_recorded, name_suffix=''):
    """
    The function may be hard to understand. Here is a calling example:
        CUDA_VISIBLE_DEVICES='' python record.py --option decoding_info --prefix stay_env --theta_hidden_size 32 \
            --hidden_size 64 --theta_fast_size 4 --policy_scan_len 397 --wd 1e-3 --n_agents 64 --total_eval_steps 300\
            --bottleneck_size 4 --noise_scale 0.5 --hippo_mem_len 5 --suffix cons_plan --info_type replay_phase \
            --pseudo_reward_idx 1 

        Notice how the info_type and the suffix works.

    Different suffix indicates different information we are focusing on. For example, 'cons_plan' means we are selecting replays serving \
        consolidation function or planning function, and 'switch' means we are selecting replays before and after the reward location changes.
    All information would be recorded as variable 'infos'. 

    Then we call function cal_decoding_acc or cal_energy based on different info_type. Variable 'quantities' include the results. 

    Then we call plot_replay_curve to plot the results. 

    Args: 
        clfs: lists of four original classifiers corresponding to hippo, theta, theta_slow and hipp_info
        hippos: lists of three phases of hippo(start, mid, goal). Notice that every single lists consists of args.n_agents arrays corresponding to every agent.
        thetas: same as above
        theta_slows: same as above
        hipp_infos: same as above
        phases: same as above
        trajs: same as above
        items_to_be_recorded: same as above. It should be the items you want to decode (for example, phase, action...)
    
    Returns: No return
    """
    cons_plan_titles = ['cons_0','cons_1','plan_0','plan_1']
    switch_titles = ['00','01','011','0111']
    titles_list = [switch_titles, cons_plan_titles]

    suffix_list = ['switch','cons_plan']
    suffix_input = args.suffix.split('-')
    
    for input_name in suffix_input:
        if input_name in suffix_list:
            i = suffix_list.index(input_name)
            print('plot',input_name)
            items = ['mid', 'goal']
            quantities = []
            for j in range(len(items)):
                print(items[j])
                # choose the suitable infos
                if input_name=='cons_plan':
                    infos = select_cons_plan_infos(args, hippos[j], theta_fasts[j], 
                                                        theta_slows[j], hipp_infos[j], phases[j], trajs[j],
                                                        items_to_be_recorded[j])
                    
                elif input_name=='switch':  
                    if j==0:
                        switch_infos = select_switch_mid_infos(args.n_agents, hippos[j], theta_fasts[j], \
                                                                    theta_slows[j], hipp_infos[j], phases[j], items_to_be_recorded[j])
                    else:
                        print(len(hippos[j]), len(theta_fasts[j]))
                        print(len(theta_slows[j]))
                        print(len(hipp_infos[j]))
                        print(len(phases[j]), len(trajs[j]))
                        switch_infos = select_switch_start_goal_infos(args.n_agents, hippos[j], theta_fasts[j], \
                                                                    theta_slows[j], hipp_infos[j], phases[j], items_to_be_recorded[j])
                    infos = switch_infos
                

                # 3(start-mid-goal) * 4(00-01-011-0111) * 4(hippo,theta,theta_slow,hipp_info)
                if args.info_type == 'replay_phase':
                    print('len:',len(clfs),len(clfs[0]))
                    quantities.append(cal_decoding_acc(args, clfs, infos))
                    baseline = 1/args.pseudo_reward.shape[0]
                elif args.info_type == 'energy':
                    quantities.append(cal_energy(infos))
                    baseline=0
                elif args.info_type == 'replay_action':
                    quantities.append(cal_decoding_error_action(args, clfs, infos))
                    baseline = 1/args.n_action
                if j==0:
                    print(items[j])
                    print(quantities[-1])
            plot_replay_curve(quantities, titles_list[i], args.prefix, input_name+'_'+name_suffix, args.info_type, args.replay_steps, baseline)
    if args.info_type == 'replay_phase':
        for st in range(args.replay_steps):
            print(clfs[0][st].predict(switch_infos[0][0][:100,st,:]))
            print(clfs[1][st].predict(switch_infos[1][0][:100,st,:]))
            print(switch_infos[-1][0][:100,st])
    return quantities

def train_phase_decoder(args, clfs, hippos, thetas, theta_slows, hipp_infos, phases):
    """
    Train the decoder for the phase. If you want to train a decoder for another purpose (action) please resort to another function.
    
    Args:
        clfs: lists of four original classifiers corresponding to hippo, theta, theta_slow and hipp_info
        hippos: lists of three phases of hippo(start, mid, goal). Notice that every single lists consists of args.n_agents arrays corresponding to every agent.
        thetas: same as above
        theta_slows: same as above
        hipp_infos: same as above
        phases: same as above

    Returns: 
    Four trained classifiers as in clfs. They will be saved in vivo.
    """
    clf_hippo_all_steps, clf_theta_all_steps, clf_theta_slow_all_steps, clf_hipp_info_all_steps = clfs
    key = jax.random.PRNGKey(0)
    for st in range(args.replay_steps+4):
        clf_hippo = clf_hippo_all_steps[st]
        clf_theta = clf_theta_all_steps[st]
        clf_theta_slow = clf_theta_slow_all_steps[st]
        clf_hipp_info = clf_hipp_info_all_steps[st]
        replay_length = jnp.concatenate([x[:,st,:].reshape(-1,args.hidden_size) for x in hippos[1:]],0).shape[0]
        key, subkey = jax.random.split(key)
        shuffle_idx_for_hipp_info = jax.random.permutation(subkey, replay_length)
        shuffle_idx_for_theta = jax.random.permutation(subkey, replay_length+hippos[0].shape[0]*hippos[0].shape[1])
        hippo_dataset = jnp.concatenate([hippos[0].reshape(-1,args.hidden_size)]+[x[:,st,:].reshape(-1,args.hidden_size) for x in hippos[1:]],0).at[shuffle_idx_for_theta].get()
        print(replay_length, hippo_dataset.shape[0])
        theta_dataset = jnp.concatenate([thetas[0].reshape(-1,args.theta_fast_size)]+[x[:,st,:].reshape(-1,args.theta_fast_size) for x in thetas[1:]],0).at[shuffle_idx_for_theta].get()
        # theta_slow_dataset = jnp.concatenate([theta_slows[0].reshape(-1,args.theta_hidden_size-args.theta_fast_size)[:30000]]+
        #     [x[:,st,:].reshape(-1,args.theta_hidden_size-args.theta_fast_size) for x in theta_slows[1:]],0).at[shuffle_idx_for_theta].get()
        hipp_info_dataset = jnp.concatenate([x[:,st,:].reshape(-1,args.bottleneck_size) for x in hipp_infos],0).at[shuffle_idx_for_hipp_info].get()

        hipp_info_dataset = jnp.concatenate([hipp_infos[0].reshape(-1,args.bottleneck_size)])

        phase_dataset = jnp.concatenate([x.reshape(-1) for x in phases],0).at[shuffle_idx_for_hipp_info].get()
        phase_dataset_for_hipp_info = jnp.concatenate([x[:,st].reshape(-1) for x in phases[1:]],0).at[shuffle_idx_for_hipp_info].get()

        phase_dataset_for_hipp_info = jnp.concatenate([phases[0].reshape(-1)])

        phase_dataset_for_theta = jnp.concatenate([phases[0].reshape(-1)]+
            [x[:,st].reshape(-1) for x in phases[1:]],0).at[shuffle_idx_for_theta].get()
        print('training decoder, dataset:')
        print(jnp.concatenate([hippos[0].reshape(-1,args.hidden_size)]+[x[:,st,:].reshape(-1,args.hidden_size) for x in hippos[1:]],0).shape)
        print(shuffle_idx_for_theta.shape)
        print(hippos[0].shape[0],hippos[0].shape[1])
        print(len(hippo_dataset), len(theta_dataset), len(hipp_info_dataset), len(phase_dataset_for_theta), len(phase_dataset_for_hipp_info))
        print(phase_dataset_for_theta[:100])
        # for i in range(2):
        #     print(i)
        #     print(hippo_dataset[i].shape, theta_dataset[i].shape, theta_slow_dataset[i].shape, phase_dataset[i].shape)
            # if i>=1:
            #     print(hipp_info_dataset[i-1].shape, phase_dataset_for_hipp_info[i-1].shape)
        
        # hippo_train = jnp.concatenate([x.at[:int(x.shape[0]*0.8)].get() for x in hippo_dataset],0)
        # theta_train = jnp.concatenate([x.at[:int(x.shape[0]*0.8)].get() for x in theta_dataset],0)
        # theta_slow_train = jnp.concatenate([x.at[:int(x.shape[0]*0.8)].get() for x in theta_slow_dataset],0)
        # hipp_info_train = jnp.concatenate([x.at[:int(x.shape[0]*0.8)].get() for x in hipp_info_dataset],0)
        # phase_train = jnp.concatenate([x.at[:int(x.shape[0]*0.8)].get() for x in phase_dataset],0)
        # phase_train_for_hipp_info = jnp.concatenate([x.at[:int(x.shape[0]*0.8)].get() for x in phase_dataset_for_hipp_info],0)
        # phase_train_for_theta = jnp.concatenate([x.at[:int(x.shape[0]*0.8)].get() for x in phase_dataset_for_theta],0)
        hippo_train = hippo_dataset[:int(hippo_dataset.shape[0]*0.8)]
        theta_train = theta_dataset[:int(theta_dataset.shape[0]*0.8)]
        # theta_slow_train = theta_slow_dataset[:int(theta_slow_dataset.shape[0]*0.8)]
        hipp_info_train = hipp_info_dataset[:int(hipp_info_dataset.shape[0]*0.8)]
        phase_train = phase_dataset[:int(phase_dataset.shape[0]*0.8)]
        phase_train_for_hipp_info = phase_dataset_for_hipp_info[:int(phase_dataset_for_hipp_info.shape[0]*0.8)]
        phase_train_for_theta = phase_dataset_for_theta[:int(phase_dataset_for_theta.shape[0]*0.8)]
        # hippo_test = jnp.concatenate([x.at[int(x.shape[0]*0.8):].get() for x in hippo_dataset],0)
        # theta_test = jnp.concatenate([x.at[int(x.shape[0]*0.8):].get() for x in theta_dataset],0)
        # theta_slow_test = jnp.concatenate([x.at[int(x.shape[0]*0.8):].get() for x in theta_slow_dataset],0)
        # hipp_info_test = jnp.concatenate([x.at[int(x.shape[0]*0.8):].get() for x in hipp_info_dataset],0)
        # phase_test = jnp.concatenate([x.at[int(x.shape[0]*0.8):].get() for x in phase_dataset],0)
        # phase_test_for_hipp_info = jnp.concatenate([x.at[int(x.shape[0]*0.8):].get() for x in phase_dataset_for_hipp_info],0)
        # phase_test_for_theta = jnp.concatenate([x.at[int(x.shape[0]*0.8):].get() for x in phase_dataset_for_theta],0)
        hippo_test = hippo_dataset[int(hippo_dataset.shape[0]*0.8):]
        theta_test = theta_dataset[int(theta_dataset.shape[0]*0.8):]
        # theta_slow_test = theta_slow_dataset[int(theta_slow_dataset.shape[0]*0.8):]
        hipp_info_test = hipp_info_dataset[int(hipp_info_dataset.shape[0]*0.8):]
        phase_test = phase_dataset[int(phase_dataset.shape[0]*0.8):]
        phase_test_for_hipp_info = phase_dataset_for_hipp_info[int(phase_dataset_for_hipp_info.shape[0]*0.8):]
        phase_test_for_theta = phase_dataset_for_theta[int(phase_dataset_for_theta.shape[0]*0.8):]

        print('hippo_train:',hippo_train.shape)
        print('theta_train:',theta_train.shape)
        # print('theta_slow_train:',theta_slow_train.shape)
        print('hipp_info_train:',hipp_info_train.shape)
        # print('phase_train:',phase_train.shape)

        shuffle_phase = jax.random.permutation(jax.random.PRNGKey(0), phase_train_for_theta)
        clf_hippo = clf_hippo.fit(hippo_train, phase_train_for_theta)
        clf_theta = clf_theta.fit(theta_train, phase_train_for_theta)
        # clf_theta_slow = clf_theta_slow.fit(theta_slow_train, phase_train_for_theta)
        clf_hipp_info = clf_hipp_info.fit(hipp_info_train, phase_train_for_hipp_info)
        print('acc on train')
        print('hippo',clf_hippo.score(hippo_train, phase_train_for_theta))
        print('theta',clf_theta.score(theta_train, phase_train_for_theta))
        # print('theta_slow',clf_theta_slow.score(theta_slow_train, phase_train_for_theta))
        print('hipp_info',clf_hipp_info.score(hipp_info_train, phase_train_for_hipp_info))
        print('acc on test')
        print('hippo',clf_hippo.score(hippo_test, phase_test_for_theta))
        print('theta',clf_theta.score(theta_test, phase_test_for_theta))
        # print('theta_slow',clf_theta_slow.score(theta_slow_test, phase_test_for_theta))
        print('hipp_info',clf_hipp_info.score(hipp_info_test, phase_test_for_hipp_info))

    clfs = (clf_hippo_all_steps, clf_theta_all_steps, clf_theta_slow_all_steps, clf_hipp_info_all_steps)
    if 'decoder' not in os.listdir():
        os.mkdir('decoder')
    save_path = './decoder/'+args.prefix+'_phase_decoder_'+str(args.pseudo_reward_idx)
    with open(save_path,'wb') as f:
        pickle.dump(clfs, f)
    print('save decoder to:'+save_path)

    return clfs

def train_action_decoder(args, clfs, hippos, thetas, theta_slows, hipp_infos, actions, action_step):
    """
    Train the decoder for the action. If you want to train a decoder for another purpose (phase) please resort to another function.
    
    Args:
        clfs: lists of four original classifiers corresponding to hippo, theta, theta_slow and hipp_info
        hippos: lists of three phases of hippo(start, mid, goal). Notice that every single lists consists of args.n_agents arrays corresponding to every agent.
        thetas: same as above
        theta_slows: same as above
        hipp_infos: same as above
        actions: same as above, notice they have shape[-1] = 1 and can only predict one action.

    Returns: 
        Four trained classifiers as in clfs. They will be saved in vivo.
    """
    clf_hippo_all_steps, clf_theta_all_steps, clf_theta_slow_all_steps, clf_hipp_info_all_steps = clfs
    key = jax.random.PRNGKey(0)
    for st in range(args.replay_steps+4):
        clf_hippo = clf_hippo_all_steps[st]
        clf_theta = clf_theta_all_steps[st]
        # clf_theta_slow = clf_theta_slow_all_steps[st]
        clf_hipp_info = clf_hipp_info_all_steps[st]
        length = jnp.concatenate([x[:,st,:].reshape(-1,args.hidden_size) for x in hippos[1:]],0).shape[0]
        key, subkey = jax.random.split(key)
        shuffle_idx = jax.random.permutation(subkey, length)
        hippo_dataset = jnp.concatenate([x[:,st,:].reshape(-1,args.hidden_size) for x in hippos[1:]],0).at[shuffle_idx].get()
        theta_dataset = jnp.concatenate([x[:,st,:].reshape(-1,args.theta_fast_size) for x in thetas[1:]],0).at[shuffle_idx].get()
        # theta_slow_dataset = jnp.concatenate([x[:,st,:].reshape(-1,args.theta_hidden_size-args.theta_fast_size) for x in theta_slows],0).at[shuffle_idx].get()
        hipp_info_dataset = jnp.concatenate([x[:,st,:].reshape(-1,args.bottleneck_size) for x in hipp_infos],0).at[shuffle_idx].get()
        action_dataset_for_hipp_info = jnp.concatenate([x[:,st].reshape(-1,2) for x in actions[1:]],0).at[shuffle_idx].get()
        action_dataset_for_theta = jnp.concatenate(
            [x[:,st].reshape(-1,2) for x in actions[1:]],0).at[shuffle_idx].get()
        

        print('training decoder, dataset:')
        # for i in range(2):
        #     print(hippo_dataset[i].shape, theta_dataset[i].shape, theta_slow_dataset[i].shape, 
        #                 hipp_info_dataset[i].shape, action_dataset[i].shape)

        hippo_train = hippo_dataset[:int(hippo_dataset.shape[0]*0.8)]
        theta_train = theta_dataset[:int(theta_dataset.shape[0]*0.8)]
        # theta_slow_train = theta_slow_dataset[:int(theta_slow_dataset.shape[0]*0.8)]
        hipp_info_train = hipp_info_dataset[:int(hipp_info_dataset.shape[0]*0.8)]
        action_train_for_hipp_info = action_dataset_for_hipp_info[:int(action_dataset_for_hipp_info.shape[0]*0.8)]
        action_train_for_theta = action_dataset_for_theta[:int(action_dataset_for_theta.shape[0]*0.8)]

        hippo_test = hippo_dataset[int(hippo_dataset.shape[0]*0.8):]
        theta_test = theta_dataset[int(theta_dataset.shape[0]*0.8):]
        # theta_slow_test = theta_slow_dataset[int(theta_slow_dataset.shape[0]*0.8):]
        hipp_info_test = hipp_info_dataset[int(hipp_info_dataset.shape[0]*0.8):]
        action_test_for_hipp_info = action_dataset_for_hipp_info[int(action_dataset_for_hipp_info.shape[0]*0.8):]
        action_test_for_theta = action_dataset_for_theta[int(action_dataset_for_theta.shape[0]*0.8):]
    
        print('hippo_train:',hippo_train.shape)
        print('theta_train:',theta_train.shape)
        # print('theta_slow_train:',theta_slow_train.shape)
        print('hipp_info_train:',hipp_info_train.shape)
        print('action_train_for_theta:',action_train_for_theta.shape)
        print('action_train_for_hipp_info:',action_train_for_hipp_info.shape)

        clf_hippo = clf_hippo.fit(hippo_train, action_train_for_theta)
        clf_theta = clf_theta.fit(theta_train, action_train_for_theta)
        # clf_theta_slow = clf_theta_slow.fit(theta_slow_train, action_train)
        clf_hipp_info = clf_hipp_info.fit(hipp_info_train, action_train_for_hipp_info)
        print('acc train')
        print('hippo',clf_hippo.score(hippo_train, action_train_for_theta))
        print('theta',clf_theta.score(theta_train, action_train_for_theta))
        print('hipp_info',clf_hipp_info.score(hipp_info_train, action_train_for_hipp_info))
        print('acc test')
        print('hippo',clf_hippo.score(hippo_test, action_test_for_theta))
        print('theta',clf_theta.score(theta_test, action_test_for_theta))
        print(jnp.concatenate([clf_hippo.predict(hippo_test[:100]), clf_theta.predict(theta_test[:100]), action_test_for_theta[:100]],1))
        # print('theta_slow',clf_theta_slow.score(theta_slow_test, action_test))
        print('hipp_info',clf_hipp_info.score(hipp_info_test, action_test_for_hipp_info))

    clfs = (clf_hippo_all_steps, clf_theta_all_steps, clf_theta_slow_all_steps, clf_hipp_info_all_steps)

    if 'decoder' not in os.listdir():
        os.mkdir('decoder')
    save_path = './decoder/'+args.prefix+'_action_decoder_'+str(action_step)
    # with open(save_path, 'wb') as f:
    #     pickle.dump(clfs, f)
    # print('save decoder to:' + save_path)

    return clfs



# @partial(jax.jit, static_argnums=(0,1,2))
def cal_decoding_acc(args, clfs, infos):
    """
    Calculate the decoding accuracy using clfs and infos.

    Args:
        clfs: lists of four original classifier ensemble corresponding to hippo, theta, theta_slow and hipp_info
            each classifier ensemble consists of 4 classifiers corresponding to different replay steps.  
        infos: the same format as the returns of `select_switch_transition`. It consists of five lists corresponding to\
            hippo, theta, theta_slow, hipp_info, phase(or other predicting target).
        
    Returns:
        Four accuracy lists corresponding to aforementioned infos `(00,01,011,0111)` or `(cons_0,cons_1,plan_0,plan_1)`. Every list includes four arrays corresponding to accuracy of hippo, theta, theta_slow and hipp_info.
        Every array is of shape [args.replay_steps+2,] corresponding to different accuracy at different steps.
    
    """
    clf_hippo_all_steps, clf_theta_all_steps, clf_theta_slow_all_steps, clf_hipp_info_all_steps = clfs
    origin_hippo, origin_theta, origin_theta_slow, origin_hipp_info, origin_phase = infos
    # every info is split into 4 lists containing needed vectors
    acc_hippo = jnp.zeros((len(origin_hippo),args.replay_steps+4))
    acc_theta = jnp.zeros((len(origin_theta),args.replay_steps+4))
    acc_theta_slow = jnp.zeros((len(origin_theta_slow),args.replay_steps+4))
    acc_hipp_info = jnp.zeros((len(origin_hipp_info),args.replay_steps+4))
    print('decoding...')

    for st in range(args.replay_steps+4):
        clf_hippo = clf_hippo_all_steps[st]
        clf_theta = clf_theta_all_steps[st]
        clf_theta_slow = clf_theta_slow_all_steps[st]
        clf_hipp_info = clf_hipp_info_all_steps[st]

        decoding_hippo = [clf_hippo.predict(x[:,st,:].reshape(-1,args.hidden_size)) for x in origin_hippo]
        decoding_theta = [clf_theta.predict(x[:,st,:].reshape(-1,args.theta_fast_size)) for x in origin_theta]
        # decoding_theta_slow = [clf_theta_slow.predict(x[:,st,:].reshape(-1,args.theta_hidden_size-args.theta_fast_size)) for x in origin_theta_slow]
        decoding_hipp_info = [clf_hipp_info.predict(x[:,st,:].reshape(-1,args.bottleneck_size)) for x in origin_hipp_info]
        decoding_phase = [x.reshape(-1) for x in origin_phase]
        # for i in range(3):
        #     print(decoding_hippo[i].shape, decoding_theta[i].shape, decoding_theta_slow[i].shape, decoding_hipp_info[i].shape, decoding_phase[i].shape)

        for i in range(len(decoding_phase)):
            acc_hippo = acc_hippo.at[i,st].set(jnp.where(
                    jnp.isclose(decoding_hippo[i],decoding_phase[i]),1,0
                    ).mean(0)
                )
            acc_theta = acc_theta.at[i,st].set(jnp.where(
                    jnp.isclose(decoding_theta[i],decoding_phase[i]),1,0
                    ).mean(0)
                )
            # acc_theta_slow = acc_theta_slow.at[i,st].set(jnp.where(
            #         jnp.isclose(decoding_theta_slow[i],decoding_phase[i]),1,0
            #         ).mean(0)
            #     )
            acc_hipp_info = acc_hipp_info.at[i,st].set(jnp.where(
                    jnp.isclose(decoding_hipp_info[i],decoding_phase[i]),1,0
                    ).mean(0)
                )

    acc = jnp.array([acc_hippo, acc_theta, acc_theta_slow, acc_hipp_info])
    # 4(hippo,theta,theta_slow,hipp_info) * 4(00,01,011,0111) * (replay_steps+2)
    return acc.transpose(1,0,2)
    # 4(00,01,011,0111) * 4(hippo,theta,theta_slow,hipp_info) * (replay_steps+2)



# @partial(jax.jit, static_argnums=(0,1,2))
def cal_decoding_error_action(args, clfs, infos):
    """
    Calculate the decoding accuracy of action plan using clfs and infos.

    Args:
        clfs: lists of four original classifier ensemble corresponding to hippo, theta, theta_slow and hipp_info
            each classifier ensemble consists of 4 classifiers corresponding to different replay steps.  
        infos: the same format as the returns of `select_switch_transition`. It consists of five lists corresponding to\
            hippo, theta, theta_slow, hipp_info, phase(or other predicting target).
        
    Returns:
        Four accuracy lists corresponding to aforementioned infos `(00,01,011,0111)` or `(cons_0,cons_1,plan_0,plan_1)`. Every list includes four arrays corresponding to accuracy of hippo, theta, theta_slow and hipp_info.
        Every array is of shape [args.replay_steps+2,] corresponding to different accuracy at different steps.
    
    """
    clf_hippo_all_steps, clf_theta_all_steps, clf_theta_slow_all_steps, clf_hipp_info_all_steps = clfs
    origin_hippo, origin_theta, origin_theta_slow, origin_hipp_info, origin_action = infos
    # every info is split into 4 lists containing needed vectors
    acc_hippo = jnp.zeros((len(origin_hippo),args.replay_steps+4))
    acc_theta = jnp.zeros((len(origin_theta),args.replay_steps+4))
    acc_theta_slow = jnp.zeros((len(origin_theta_slow),args.replay_steps+4))
    acc_hipp_info = jnp.zeros((len(origin_hipp_info),args.replay_steps+4))
    print('decoding...')

    for st in range(args.replay_steps+4):
        clf_hippo = clf_hippo_all_steps[st]
        clf_theta = clf_theta_all_steps[st]
        clf_theta_slow = clf_theta_slow_all_steps[st]
        clf_hipp_info = clf_hipp_info_all_steps[st]

        decoding_hippo = [clf_hippo.predict(x[:,st,:].reshape(-1,args.hidden_size)) for x in origin_hippo]
        decoding_theta = [clf_theta.predict(x[:,st,:].reshape(-1,args.theta_fast_size)) for x in origin_theta]
        # decoding_theta_slow = [clf_theta_slow.predict(x[:,st,:].reshape(-1,args.theta_hidden_size-args.theta_fast_size)) for x in origin_theta_slow]
        decoding_hipp_info = [clf_hipp_info.predict(x[:,st,:].reshape(-1,args.bottleneck_size)) for x in origin_hipp_info]
        decoding_action = [x.reshape(-1,2) for x in origin_action]
        # for i in range(3):
        #     print(decoding_hippo[i].shape, decoding_theta[i].shape, decoding_theta_slow[i].shape, decoding_hipp_info[i].shape, decoding_phase[i].shape)
        # print(decoding_hippo[0][:50],decoding_theta[0][:50],decoding_action[0][:50])
        for i in range(len(decoding_action)):
            acc_hippo = acc_hippo.at[i,st].set(
                    jnp.linalg.norm(decoding_hippo[i]-decoding_action[i],axis=-1).mean(0)
                )
            acc_theta = acc_theta.at[i,st].set(
                    jnp.linalg.norm(decoding_theta[i]-decoding_action[i],axis=-1).mean(0)
                )
            # acc_theta_slow = acc_theta_slow.at[i,st].set(jnp.where(
            #         jnp.isclose(decoding_theta_slow[i],decoding_phase[i]),1,0
            #         ).mean(0)
            #     )
            acc_hipp_info = acc_hipp_info.at[i,st].set(
                    jnp.linalg.norm(decoding_hipp_info[i]-decoding_action[i],axis=-1).mean(0)
                )

    acc = jnp.array([acc_hippo, acc_theta, acc_theta_slow, acc_hipp_info])
    # 4(hippo,theta,theta_slow,hipp_info) * 4(00,01,011,0111) * (replay_steps+2)
    return acc.transpose(1,0,2)
    # 4(00,01,011,0111) * 4(hippo,theta,theta_slow,hipp_info) * (replay_steps+2)

def plot_replay_curve_single(axis, quantity, title):
    # print(acc)
    if len(quantity[0])==2:
        axis.plot(quantity[0][0], label='hippo_0')
        axis.plot(quantity[0][1], label='hippo_1')
        axis.plot(quantity[1][0], label='theta_0')
        axis.plot(quantity[1][1], label='theta_1')
    else:
        axis.plot(quantity[0], label='hippo')
        axis.plot(quantity[1], label='theta')
        axis.plot(quantity[2], label='theta_slow')
        axis.plot(quantity[3], label='hipp_info')
    axis.set_title(title)
    axis.set_ylim(-0.2,1.2)
    axis.set_xticks([0,1,2,3,4,5],['init','0','1','2','3','output'])


def plot_replay_curve(quantities, titles, prefix, suffix, theme, replay_steps, baseline):
    """
    Plot the replay curve.

    Args:
        quantities: the same format as the return of `cal_decoding_acc` or `cal_energy`
        titles: four titles corresponding to `(00,01,011,0111)` or `(cons_0,cons_1,plan_0,plan_1)`
        prefix: `args.prefix`
        suffix: `args.suffix` (`switch` or `cons_plan`)
        theme: `args.info_type`
        replay_steps: `args.replay_steps`
        baseline: the chance_level (0 for energy, 1/pseudo_reward.shape[0] for decoding_acc)

    Returns: No return

    """
    # print(quantities[0][:,1,:])
    fig, axes = plt.subplots(len(quantities),len(titles),figsize=(5*len(titles),3*len(quantities)))
    
    for replay_loc in range(len(quantities)):
        for i in range(len(titles)):
            # print(titles[i],len(quantities[replay_loc][i]))
            plot_replay_curve_single(axes[replay_loc][i], quantities[replay_loc][i], titles[i])
            axes[replay_loc][i].plot(jnp.arange(replay_steps+4),jnp.ones(replay_steps+4)*baseline,'--',color='grey')
    save_path = './figures/'+prefix+'/'+theme+'_'+suffix+'.png'
    fig.tight_layout()
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper right', fontsize=16)
    fig.suptitle(prefix+'_'+theme+'_'+suffix, fontsize=16)
    if theme not in os.listdir('./'):
        os.mkdir('./'+theme)
    print('save '+theme+' to:',save_path)
    plt.savefig(save_path)
    return

def select_cons_plan_infos(args, hippo, theta, theta_slow, hipp_info, phase, traj, item_to_be_recorded):
    """
    Choose the appropriate info (hippo_hidden/theta/theta_slow/hipp_info) based on segment represented by the trajectory.
    
    Args:
        width: `args.width`
        height: `args.height`
        traj: array of replay trajectory (not trajs consisting of traj of `(start-mid-goal)`)
        phase: same as above

    Returns:
        All infos that have been classified based on cons/plan and ck0/ck1. 

    """
    cons_plan_condition = select_consolidation_plannning(args.width, args.height, 
                                    jnp.concatenate(traj), jnp.concatenate(phase))
    cons_plan_idx = [jnp.array(jnp.where(cons_plan_condition[i])[0]) for i in range(4)]
    cons_plan_infos = []
    replay_hippo = []
    replay_theta = []
    replay_theta_slow = []
    replay_hipp_info = []
    replay_item_to_be_recorded = []
    print('collecting cons_plan infos...')

    for ki in range(len(cons_plan_idx)):  
        k = cons_plan_idx[ki]
        if k.shape[0]==0:
            k = jnp.zeros(1, dtype=jnp.int32)
        replay_hippo.append(jnp.concatenate(hippo)[k])
        replay_theta.append(jnp.concatenate(theta)[k])
        replay_theta_slow.append(jnp.concatenate(theta_slow)[k])
        replay_hipp_info.append(jnp.concatenate(hipp_info)[k])
        replay_item_to_be_recorded.append(jnp.concatenate(item_to_be_recorded)[k])

    #     traj_to_be_recorded = jnp.concatenate(traj)[k]
    #     print(traj_to_be_recorded.shape)
    #     plt.subplot(2,2,ki+1)
    #     plt.xlim(0-0.5,args.width-0.5)
    #     plt.ylim(0-0.5,args.height-0.5)
    #     plt.grid()
    #     plot_replay(traj_to_be_recorded[:3], 'blue', args)
    #     calculate_replay_to_scan = partial(calculate_replay, width=args.width, height=args.height, ck0_x=env.ck0_x, ck0_y=env.ck0_y, ck1_x=env.ck1_x, ck1_y=env.ck1_y,
    #                         ck0_x_g_lb=env.ck0_x_g_lb, ck0_y_g_lb=env.ck0_y_g_lb, ck1_x_g_lb=env.ck1_x_g_lb, ck1_y_g_lb=env.ck1_y_g_lb)
    #     directionality, forward_degree, score, max_segment = \
    #         jax.vmap(calculate_replay_to_scan,0,0)(traj_to_be_recorded[:3])
    #     print('directionality:',directionality)
    #     print('forward_degree:',forward_degree)
    #     print('score:',score)
    #     print('max_segment:',max_segment)
    # plt.show()
    cons_plan_infos = [replay_hippo, replay_theta, replay_theta_slow, replay_hipp_info, replay_item_to_be_recorded]
    return cons_plan_infos

def select_switch_mid_infos(n_agents, hippo, theta, theta_slow, hipp_info, phase, item_to_be_recorded):
    """
    Choose the appropriate #mid-replay# info (hippo_hidden/theta/theta_slow/hipp_info) based on transition of the phase (corresponding to a phase list\
         consisting of args.n_agents arrays instead of a list of three lists(start-mid-goal) which we quote as 'phases')
    
    Args:
        hippo: one of three lists of `hippos`(start, mid, goal), consisting of `args.n_agents` arrays corresponding to every agent.
        theta: same as above
        theta_slow: same as above
        hipp_info: same as above
        phase: same as above
        item_to_be_recorded: same as above, you can add any items you want to record. See the argument in `cal_plot_replay_curve` and how it's called.
    
    Returns:
        All infos that have been classified based on transition of the phase. 
        One list consists of four arrays corresponding to `(00,01,011,0111)` which refers to control group, the first time \
            of meeting the replaced checkpoint, the second time to do so, the third time.
    """
    
    items = ['00','01','011','0111']
    replay_hippo = [[],[],[],[]]
    replay_theta = [[],[],[],[]]
    replay_theta_slow = [[],[],[],[]]
    replay_hipp_info = [[],[],[],[]]
    replay_item_to_be_recorded = [[],[],[],[]]

    print('collecting switching hippo and theta replay')
    for n in range(n_agents):
        print(len(item_to_be_recorded[n]), len(phase[n]))
        if len(phase[n])>=2:
            for i in range(len(phase[n])-1):
                if phase[n][i] == phase[n][i+1] == 0:
                    replay_hippo[0].append(hippo[n][i+1])
                    replay_theta[0].append(theta[n][i+1])
                    replay_theta_slow[0].append(theta_slow[n][i+1])
                    replay_hipp_info[0].append(hipp_info[n][i+1])
                    replay_item_to_be_recorded[0].append(item_to_be_recorded[n][i+1])

                elif phase[n][i] != phase[n][i+1]:
                    replay_hippo[1].append(hippo[n][i+1])
                    replay_theta[1].append(theta[n][i+1])
                    replay_theta_slow[1].append(theta_slow[n][i+1])
                    replay_hipp_info[1].append(hipp_info[n][i+1])
                    replay_item_to_be_recorded[1].append(item_to_be_recorded[n][i+1])

                    if i+2<len(phase[n]) and phase[n][i+1] == phase[n][i+2]:
                        replay_hippo[2].append(hippo[n][i+2])
                        replay_theta[2].append(theta[n][i+2])
                        replay_theta_slow[2].append(theta_slow[n][i+2])
                        replay_hipp_info[2].append(hipp_info[n][i+2])
                        replay_item_to_be_recorded[2].append(item_to_be_recorded[n][i+2])

                        if i+3<len(phase[n]) and phase[n][i+2] == phase[n][i+3]:
                            replay_hippo[3].append(hippo[n][i+3])
                            replay_theta[3].append(theta[n][i+3])
                            replay_theta_slow[3].append(theta_slow[n][i+3])
                            replay_hipp_info[3].append(hipp_info[n][i+3])
                            replay_item_to_be_recorded[3].append(item_to_be_recorded[n][i+3])


    replay_hippo = [jnp.array(x) for x in replay_hippo]
    replay_theta = [jnp.array(x) for x in replay_theta]
    replay_theta_slow = [jnp.array(x) for x in replay_theta_slow]
    replay_hipp_info = [jnp.array(x) for x in replay_hipp_info]
    replay_item_to_be_recorded = [jnp.array(x) for x in replay_item_to_be_recorded]
    print('replay_hippo:',[x.shape for x in replay_hippo])
    print('replay_theta:',[x.shape for x in replay_theta])
    print('replay_theta_slow:',[x.shape for x in replay_theta_slow])
    print('replay_hipp_info:',[x.shape for x in replay_hipp_info])
    print('replay_phase:',[x.shape for x in replay_item_to_be_recorded])

    return replay_hippo, replay_theta, replay_theta_slow, replay_hipp_info, replay_item_to_be_recorded


def select_switch_start_goal_infos(n_agents, hippo, theta, theta_slow, hipp_info, phase, item_to_be_recorded):
    """
    Choose the appropriate #start/goal# info (hippo_hidden/theta/theta_slow/hipp_info) based on transition of the phase (corresponding to a phase list\
         consisting of args.n_agents arrays instead of a list of three lists(start-mid-goal) which we quote as 'phases')
    
    Args:
        hippo: one of three lists of `hippos`(start, mid, goal), consisting of `args.n_agents` arrays corresponding to every agent.
        theta: same as above
        theta_slow: same as above
        hipp_info: same as above
        phase: same as above
        item_to_be_recorded: same as above, you can add any items you want to record. See the argument in `cal_plot_replay_curve` and how it's called.
    
    Returns:
        All infos that have been classified based on transition of the phase. 
        One list consists of four arrays corresponding to `(00,01,011,0111)` which refers to control group, the first time \
            of meeting the replaced checkpoint, the second time to do so, the third time.
    

    """
    items = ['00','01','011','0111']
    replay_hippo = [[],[],[],[]]
    replay_theta = [[],[],[],[]]
    replay_theta_slow = [[],[],[],[]]
    replay_hipp_info = [[],[],[],[]]
    replay_phase = [[],[],[],[]]

    print('collecting switching hippo and theta replay')
    for n in range(n_agents):
        if len(phase[n])>=3:
            for i in range(len(phase[n])-2):
                if phase[n][i] == phase[n][i+1] == 0:
                    replay_hippo[0].append(hippo[n][i+2])
                    replay_theta[0].append(theta[n][i+2])
                    replay_theta_slow[0].append(theta_slow[n][i+2])
                    replay_hipp_info[0].append(hipp_info[n][i+2])
                    replay_phase[0].append(item_to_be_recorded[n][i+1])

                elif phase[n][i] != phase[n][i+1]:
                    replay_hippo[1].append(hippo[n][i+2])
                    replay_theta[1].append(theta[n][i+2])
                    replay_theta_slow[1].append(theta_slow[n][i+2])
                    replay_hipp_info[1].append(hipp_info[n][i+2])
                    replay_phase[1].append(item_to_be_recorded[n][i+1])

                    if i+3<len(phase[n]) and phase[n][i+1] == phase[n][i+2]:
                        replay_hippo[2].append(hippo[n][i+3])
                        replay_theta[2].append(theta[n][i+3])
                        replay_theta_slow[2].append(theta_slow[n][i+3])
                        replay_hipp_info[2].append(hipp_info[n][i+3])
                        replay_phase[2].append(item_to_be_recorded[n][i+2])

                        if i+4<len(phase[n]) and phase[n][i+2] == phase[n][i+3]:
                            replay_hippo[3].append(hippo[n][i+4])
                            replay_theta[3].append(theta[n][i+4])
                            replay_theta_slow[3].append(theta_slow[n][i+4])
                            replay_hipp_info[3].append(hipp_info[n][i+4])
                            replay_phase[3].append(item_to_be_recorded[n][i+3])


    replay_hippo = [jnp.array(x) for x in replay_hippo]
    replay_theta = [jnp.array(x) for x in replay_theta]
    replay_theta_slow = [jnp.array(x) for x in replay_theta_slow]
    replay_hipp_info = [jnp.array(x) for x in replay_hipp_info]
    replay_phase = [jnp.array(x) for x in replay_phase]
    print('replay_hippo:',[len(x) for x in replay_hippo])
    print('replay_theta:',[len(x) for x in replay_theta])
    print('replay_theta_slow:',[len(x) for x in replay_theta_slow])
    print('replay_hipp_info:',[len(x) for x in replay_hipp_info])
    print('replay_phase:',[len(x) for x in replay_phase])

    return replay_hippo, replay_theta, replay_theta_slow, replay_hipp_info, replay_phase






def plot_replay(replay_traj, color, args):
    for replay_output in replay_traj:
        # print(list(zip((replay_output//args.width).tolist(), (replay_output%args.width).tolist())))
        xy = jnp.stack([replay_output//args.width, replay_output%args.width], axis=-1)
        uv = jnp.diff(xy, axis=0)
        plt.quiver(xy[:-1,0], xy[:-1,1], uv[:,0], uv[:,1], scale_units='xy', angles='xy', width=0.018, scale=1, color=color, headwidth=3,alpha=1, zorder=300)
        # plt.plot(replay_output//args.width, replay_output%args.width, c=color, marker='.', linewidth=3)
    for replay_output in replay_traj:
        plt.scatter(replay_output//args.width, replay_output%args.width, c=np.arange(args.replay_steps), cmap='Wistia', s=200)

def plot_trajectory(whole_traj:dict,args):
    agent_th, state_traj, replay_traj, reward_pos_traj, pred_r, first_output = whole_traj.values()
    # print(pred_r,pred_r.shape)
    first_pc = jnp.argmax(first_output[:args.width*args.height],axis=-1)
    first_pc = [first_pc.item()//args.width, first_pc.item()%args.width]
    first_pr = first_output[-1]
    pr = ' '.join(list(map(lambda x: format(x, '.2f'), pred_r)))
    # plt.title(f'step:{state_traj.shape[0]-1},\n{pr}')
    # plt.title(f'step{state_traj.shape[0]-1},{first_pc},{first_pr:.2f}\n{pr}')
    plt.grid()
    plt.xlim(0-0.5,args.width-0.5)
    plt.xticks(np.arange(0,args.width,1),np.arange(0,args.width,1)+1)
    plt.ylim(0-0.5,args.height-0.5)
    plt.yticks(np.arange(0,args.height,1),np.arange(0,args.height,1)+1)
    plt.plot(state_traj[:,0],state_traj[:,1])
    plt.scatter(state_traj[-1,0],state_traj[-1,1], marker='o', s=100,c='darkblue')
    # plot_replay(replay_traj[:1], 'red', args)
    # plot_replay(replay_traj[1:], 'blue', args)
    plot_replay(replay_traj, 'red', args)
    if reward_pos_traj is not None:
        if len(reward_pos_traj)>4:
            plt.scatter(reward_pos_traj[:-5,0],reward_pos_traj[:-5,1], marker='*', s=100, c='blue')
        if len(reward_pos_traj)>3:
            plt.scatter(reward_pos_traj[-4,0],reward_pos_traj[-4,1], marker='*', s=100, c='green')
        if len(reward_pos_traj)>2:
            plt.scatter(reward_pos_traj[-3,0],reward_pos_traj[-3,1], marker='*', s=100, c='yellow')
        if len(reward_pos_traj)>1:
            plt.scatter(reward_pos_traj[-2,0],reward_pos_traj[-2,1], marker='*', s=100, c='orange')
        plt.scatter(reward_pos_traj[-1,0],reward_pos_traj[-1,1], marker='*', s=100, c='r')












@partial(jax.jit, static_argnums=(1,2,3,4,5,6,7,8,9,10))
def calculate_replay(replay_traj, width, height, ck0_x, ck0_y, ck1_x, ck1_y,
                    ck0_x_g_lb, ck0_y_g_lb, ck1_x_g_lb, ck1_y_g_lb):
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
    @partial(jax.jit, static_argnums=(1,2,3,4,5,6,7,8,9,10))
    def calculate_segment_attribution(replay_traj, width, height, ck0_x, ck0_y, ck1_x, ck1_y,
                                        ck0_x_g_lb, ck0_y_g_lb, ck1_x_g_lb, ck1_y_g_lb):
        grid_s_c0, grid_s_c1, grid_c0_g, grid_c1_g = \
            jnp.zeros((width, height)), jnp.zeros((width, height)), jnp.zeros((width, height)), jnp.zeros((width, height))
        v1 = 0.5
        
        grid_s_c0 = grid_s_c0.at[:ck0_x+1, :ck0_y+1].set(1)
        grid_s_c0 = grid_s_c0.at[ck0_x+1, :ck0_y+2].set(v1)
        grid_s_c0 = grid_s_c0.at[:ck0_x+2, ck0_y+1].set(v1)
        # grid_s_c1 = grid_s_c1.at[ck0_x+2, :ck0_y+3].set(v2)
        # grid_s_c1 = grid_s_c1.at[:ck0_x+3, ck0_y+2].set(v2)

        grid_s_c1 = grid_s_c1.at[:ck1_x+1, :ck1_y+1].set(1)
        grid_s_c1 = grid_s_c1.at[ck1_x+1, :ck1_y+2].set(v1)
        grid_s_c1 = grid_s_c1.at[:ck1_x+2, ck1_y+1].set(v1)
        # grid_s_c2 = grid_s_c2.at[ck1_x+2, :ck1_y+3].set(v2)
        # grid_s_c2 = grid_s_c2.at[:ck1_x+3, ck1_y+2].set(v2)

        grid_c0_g = grid_c0_g.at[ck0_x_g_lb, ck0_y_g_lb:].set(v1)
        grid_c0_g = grid_c0_g.at[ck0_x_g_lb:, ck0_y_g_lb].set(v1)
        grid_c0_g = grid_c0_g.at[ck0_x:, ck0_y:].set(1)
        # grid_c1_g = grid_c1_g.at[ck0_x-2, ck0_y-2:].set(v2)
        # grid_c1_g = grid_c1_g.at[ck0_x-2:, ck0_y-2].set(v2)


        grid_c1_g = grid_c1_g.at[ck1_x_g_lb, ck1_y_g_lb:].set(v1)
        grid_c1_g = grid_c1_g.at[ck1_x_g_lb:, ck1_y_g_lb].set(v1)
        grid_c1_g = grid_c1_g.at[ck1_x:, ck1_y:].set(1)
        # grid_c2_g = grid_c2_g.at[ck1_x-2, ck1_y-2:].set(v2)
        # grid_c2_g = grid_c2_g.at[ck1_x-2:, ck1_y-2].set(v2)
        grid_all = jnp.stack([grid_s_c0, grid_s_c1, grid_c0_g, grid_c1_g], axis=0)
        pos = jnp.zeros((width, height))
        pos = pos.at[(replay_traj[:,0],replay_traj[:,1])].set(1)
        score = jnp.sum(grid_all*pos, axis=(1,2))/jnp.sum(grid_all*pos, axis=(0,1,2))
        score = jnp.where(jnp.isnan(score), 0, score)
        max_segment = jnp.argmax(score, axis=0)
        return score, max_segment
    
    replay_path = jnp.stack([replay_traj//width, replay_traj%width], axis=-1)
    directionality, forward_degree = calculate_path_criterion(replay_path)
    score, max_segment = calculate_segment_attribution(replay_path, width, height, ck0_x, ck0_y, ck1_x, ck1_y,
                                                    ck0_x_g_lb, ck0_y_g_lb, ck1_x_g_lb, ck1_y_g_lb)
    return directionality, forward_degree, score, max_segment

def calculate_segment_proportion(directionality, forward_degree, score, max_segment):
    dominant_score = jax.lax.top_k(score, 2)[0]
    significant = jnp.where((jnp.max(score, axis=-1) > 0.4) & (directionality > 0.4) \
                & (jnp.abs(forward_degree) > 0.4) & (dominant_score[:,0] > dominant_score[:,1]), 1, 0)
    s_c0_f = jnp.sum(jnp.where(significant & (max_segment==0) & (forward_degree>0), 1, 0))
    s_c0_r = jnp.sum(jnp.where(significant & (max_segment==0) & (forward_degree<0), 1, 0))
    s_c1_f = jnp.sum(jnp.where(significant & (max_segment==1) & (forward_degree>0), 1, 0))
    s_c1_r = jnp.sum(jnp.where(significant & (max_segment==1) & (forward_degree<0), 1, 0))
    c0_g_f = jnp.sum(jnp.where(significant & (max_segment==2) & (forward_degree>0), 1, 0))
    c0_g_r = jnp.sum(jnp.where(significant & (max_segment==2) & (forward_degree<0), 1, 0))
    c1_g_f = jnp.sum(jnp.where(significant & (max_segment==3) & (forward_degree>0), 1, 0))
    c1_g_r = jnp.sum(jnp.where(significant & (max_segment==3) & (forward_degree<0), 1, 0))
    data = [s_c0_f, s_c0_r, s_c1_f, s_c1_r, c0_g_f, c0_g_r, c1_g_f, c1_g_r]
    data = [0 if jnp.isnan(x) else x for x in data]
    data = [x/sum(data) for x in data]
    return jnp.array(data), jnp.mean(significant)

def plot_segment_proportion(axes, statistics_hippo_theta, replay_steps, row_text, set_title=False):
    proportion, hippo, theta = statistics_hippo_theta
    ###FIXME
    labels = ['s0f', 's0r', 's1f', 's1r', '0gf', '0gr', '1gf', '1gr']
    axes[0].text(0.5, 0.5, row_text, horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
    axes[0].bar(labels, proportion, color=['darkred', 'red', 'darkorange', 'orange', 'darkblue', 'blue', 'darkgreen', 'green'])
    axes[0].set_ylim([0,1])
    
    c_idx = np.arange(replay_steps).reshape(1,-1).repeat(hippo.shape[0]//replay_steps,0).reshape((-1,))
    axes[1].scatter(hippo[:,0],hippo[:,1],c=c_idx,cmap='Set1')
    axes[2].scatter(theta[:,0],theta[:,1],c=c_idx,cmap='Set1')

    if set_title:
        axes[0].set_title('segment_proportion')
        axes[1].set_title('hippo')
        axes[2].set_title('theta')
    return 

def plot_all_proportion_single(axis, proportion, row_text):
    labels = ['s0f', 's0r', 's1f', 's1r', '0gf', '0gr', '1gf', '1gr']
    axis.text(0.5, 0.5, row_text, horizontalalignment='center', verticalalignment='center', transform=axis.transAxes, fontsize=20)
    axis.bar(labels, proportion, color=['darkred', 'red', 'darkorange', 'orange', 'darkblue', 'blue', 'darkgreen', 'green'])
    axis.set_ylim([0,1])
    return

#####TODO:明天重点改一下这个，因为现有的环境并没有记录00是什么状态
def Igata_plot_line_chart(axis, proportion, title=None):
    # 5[01,011,0111,01111,0] * 8[s0f, s0r, s1f, s1r, 0gf, 0gr, 1gf, 1gr]
    print(proportion)
    proportion = jnp.where(proportion is None, 0, proportion)
    # 10000 01 011 0111
    ck0_1 = proportion
    # 01111 10 100 1000
    # ck1_0 = proportion.at[[3,4,5,6],:].get()
    # past_cons, new_cons, past_plan, new_plan
    ck0_1_line = jnp.stack([ck0_1[:,0:2].sum(-1), ck0_1[:,2:4].sum(-1),\
                ck0_1[:,4:6].sum(-1), ck0_1[:,6:8].sum(-1)], axis=1)
    # ck1_0_line = jnp.stack([ck1_0[:,2:4].sum(-1), ck1_0[:,0:2].sum(-1),\
    #             ck1_0[:,6:8].sum(-1), ck1_0[:,4:6].sum(-1)], axis=1)
    # print(ck0_1_line.shape, ck1_0_line.shape)
    # line = (ck0_1_line+ck1_0_line)/2
    line = ck0_1_line
    x = jnp.arange(proportion.shape[0])
    axis.plot(x, line[:,0], 'r', label='past_cons')
    axis.plot(x, line[:,1], 'y', label='new_cons')
    axis.plot(x, line[:,2], 'b', label='past_plan')
    axis.plot(x, line[:,3], 'g', label='new_plan')
    axis.set_xticks(x, ['pre_learn', 'learning_1', 'learning_2', 'learning_3', 'post_learn'])
    axis.set_title(title)
    return line

def plot_line_chart(axis, proportion, title=None):
    # 8[01,011,0111,01111,10,100,1000,10000] * 8[s0f, s0r, s1f, s1r, 0gf, 0gr, 1gf, 1gr]
    print(proportion)
    proportion = jnp.where(proportion is None, 0, proportion)
    # 10000 01 011 0111
    ck0_1 = proportion.at[[-1,0,1,2],:].get()
    # 01111 10 100 1000
    ck1_0 = proportion.at[[3,4,5,6],:].get()
    # past_cons, new_cons, past_plan, new_plan
    ck0_1_line = jnp.stack([ck0_1[:,0:2].sum(-1), ck0_1[:,2:4].sum(-1),\
                ck0_1[:,4:6].sum(-1), ck0_1[:,6:8].sum(-1)], axis=1)
    ck1_0_line = jnp.stack([ck1_0[:,2:4].sum(-1), ck1_0[:,0:2].sum(-1),\
                ck1_0[:,6:8].sum(-1), ck1_0[:,4:6].sum(-1)], axis=1)
    print(ck0_1_line.shape, ck1_0_line.shape)
    line = (ck0_1_line+ck1_0_line)/2
    x = jnp.arange(4)
    axis.plot(x, line[:,0], 'r', label='past_cons')
    axis.plot(x, line[:,1], 'y', label='new_cons')
    axis.plot(x, line[:,2], 'b', label='past_plan')
    axis.plot(x, line[:,3], 'g', label='new_plan')
    axis.set_xticks(x, ['pre_learn', 'learning_1', 'learning_2', 'post_learn'])
    axis.set_title(title)
    return


# def cal_plot_segment(args, step_count_of_interest, replay_of_interest, hippo_of_interest, theta_of_interest):
#     step_save_percent = [1-jnp.mean(jnp.concatenate(x)).item() for x in step_count_of_interest]
#     calculate_replay_to_scan = partial(calculate_replay, width=args.width, height=args.height, ck0_x=env.ck0_x, ck0_y=env.ck0_y, ck1_x=env.ck1_x, ck1_y=env.ck1_y,
#                                         ck0_x_g_lb=env.ck0_x_g_lb, ck0_y_g_lb=env.ck0_y_g_lb, ck1_x_g_lb=env.ck1_x_g_lb, ck1_y_g_lb=env.ck1_y_g_lb)
#     fig, axes = plt.subplots(4,6,figsize=(18,10))
#     axes_ = [axes[0][:3], axes[0][3:], axes[1][:3], axes[1][3:], 
#             axes[2][:3], axes[2][3:], axes[3][:3], axes[3][3:]]
#     super_title = ' '.join(['{:.2f}'.format(x) for x in step_save_percent])
    
#     row_text = ['0->1', '0-1->1', '0-1-1->1', '0-1-1-1->1',
#                 '1->0', '1-0->0', '1-0-0->0', '1-0-0-0->0']
#     set_title = [True, True, False, False, False, False, False, False]
#     pca = PCA(n_components=2)
#     all_statistics = []
#     for i in range(8):
#         if replay_of_interest[i]:
#             print('---')
#             print(jnp.stack(replay_of_interest[i],0).shape)
#             directionality, forward_degree, score, max_segment = \
#                 jax.vmap(calculate_replay_to_scan, 0, 0)(jnp.stack(replay_of_interest[i],0))
#             statistics, effective_proportion = calculate_segment_proportion(directionality, forward_degree, score, max_segment)
#             reduced_hippo = pca.fit_transform(jnp.concatenate(hippo_of_interest[i]))
#             reduced_theta = pca.fit_transform(jnp.concatenate(theta_of_interest[i]))
#             plot_segment_proportion(axes_[i], (statistics, reduced_hippo, reduced_theta),
#                                     args.replay_steps, row_text[i], set_title[i])
#             all_statistics.append(statistics)
#         else:
#             all_statistics.append(jnp.zeros((8,)))
    
#     fig.suptitle(args.prefix+'_'+args.replay_type + ' step save percent: '+super_title\
#         +'\neffective:{:.2f}'.format(effective_proportion))

#     fig.tight_layout()
#     print(step_save_percent)
#     if not os.path.exists('proportion'):
#         os.makedirs('proportion')
#     save_figure = './proportion/'+args.prefix+'_'+args.replay_type+'_replay.png'
#     print('save proportion to', save_figure)
#     plt.savefig(save_figure)

#     fig,axis = plt.subplots(1,1,figsize=(8,6))
#     plot_line_chart(axis, jnp.stack(all_statistics,0))
#     axis.legend()
#     axis.set_title(args.prefix+'_'+args.replay_type)
#     save_figure = './proportion/'+args.prefix+'_'+args.replay_type+'_line_chart.png'
#     print('save line_chart to', save_figure)
#     plt.savefig(save_figure)
    
#     return

def cal_plot_all_proportion(args, trajs, step_save_percent, row_text, items):
    # 3(start-mid-goal) * 8(01-011-0111-01111/10-100-1000-10000)
    calculate_replay_to_scan = partial(calculate_replay, width=args.width, height=args.height, ck0_x=env.ck0_x, ck0_y=env.ck0_y, ck1_x=env.ck1_x, ck1_y=env.ck1_y,
                                        ck0_x_g_lb=env.ck0_x_g_lb, ck0_y_g_lb=env.ck0_y_g_lb, ck1_x_g_lb=env.ck1_x_g_lb, ck1_y_g_lb=env.ck1_y_g_lb)
    plt.close()
    fig, axes = plt.subplots(len(trajs),len(row_text),figsize=(4*len(row_text),4*len(trajs)))
    all_proportion = [[] for _ in range(len(trajs))]
    all_effective_percent = [[] for _ in range(len(trajs))]
    all_step_save_percent = [items[replay_loc]+':'+' '.join(['{:.2f}'.format(x) for x in step_save_percent[replay_loc]]) for replay_loc in range(len(trajs))]
    for replay_loc in range(len(trajs)):
        for i in range(len(row_text)):
            if trajs[replay_loc][i] is not None:
                directionality, forward_degree, score, max_segment = \
                    jax.vmap(calculate_replay_to_scan, 0, 0)(trajs[replay_loc][i])
                proportion, effective_percent = calculate_segment_proportion(directionality, forward_degree, score, max_segment)
                plot_all_proportion_single(axes[replay_loc][i], proportion, row_text[i])
                all_proportion[replay_loc].append(proportion)
                all_effective_percent[replay_loc].append(effective_percent)
                significant = jnp.where((jnp.max(score, axis=-1) > 0.5) & (directionality > 0.5) & (jnp.abs(forward_degree) > 0.5))[0]
                print(items[replay_loc], row_text[i])
                replay_trajectory = jnp.stack([trajs[replay_loc][i].at[significant].get()//args.width, trajs[replay_loc][i].at[significant].get()%args.width],-1).reshape(-1,2*args.replay_steps)
                print(replay_trajectory)
                print('directionality:',directionality.at[significant].get())
                print('forward_degree:',forward_degree.at[significant].get())
                print('score:',score.at[significant].get())
            else:
                all_proportion[replay_loc].append(jnp.zeros((8,)))
                all_effective_percent[replay_loc].append(0)
    
    effective_percent = ['{:.2f}'.format(jnp.mean(jnp.array(all_effective_percent[replay_loc]))) for replay_loc in range(len(trajs))]
    effective_percent = ' '.join(effective_percent)
    fig.suptitle(args.prefix+'_'+'\n'.join(all_step_save_percent)
        +'\neffective:'+effective_percent)

    fig.tight_layout()
    # print('step_save_percent',step_save_percent)
    if not os.path.exists('proportion'):
        os.makedirs('proportion')
    save_figure = './figures/'+args.prefix+'/proportion.png'
    print('save proportion to', save_figure)
    plt.savefig(save_figure)
    plt.close()
    fig,axes = plt.subplots(1,len(items),figsize=(6*len(items),6))
    line_chart_mid_goal = []
    for replay_loc in range(len(items)):
        print(items[replay_loc])
        if len(row_text) == 8:
            plot_line_chart(axes[replay_loc], jnp.stack(all_proportion[replay_loc],0), items[replay_loc])
        else:
            line_chart =  Igata_plot_line_chart(axes[replay_loc], jnp.stack(all_proportion[replay_loc],0), items[replay_loc])
            line_chart_mid_goal.append(line_chart)
        # 5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
        # jnp.save('./figures/'+args.prefix+'/'+items[replay_loc]+'_line_chart_0.npy', line_chart)

    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper right', fontsize=16)
    fig.suptitle(args.prefix)
    fig.tight_layout()
    save_figure = './figures/'+args.prefix+'/line_chart.png'
    print('save line_chart to', save_figure)
    plt.savefig(save_figure)
    
    
    return line_chart_mid_goal






def plot_manifold_change(axes, hippo_theta, row_text, type='mid', set_title=False):
    hippo, theta = hippo_theta
    c_idx = jnp.arange(2).repeat(hippo.shape[0]//2)
    axes[0].text(0.5, 0.5, row_text, horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
    axes[0].scatter(hippo[:,0], hippo[:,1], c=c_idx, cmap='coolwarm')
    axes[1].scatter(theta[:,0], theta[:,1], c=c_idx, cmap='summer')
    if set_title:
        axes[0].set_title(type+'_hippo')
        axes[1].set_title(type+'_theta')
    return 
def cal_plot_manifold_deprecated(args, hippo_before_mid_of_interest, hippo_after_mid_of_interest, theta_before_mid_of_interest, theta_after_mid_of_interest,
                        hippo_before_goal_of_interest, hippo_after_goal_of_interest, theta_before_goal_of_interest, theta_after_goal_of_interest):
    fig, axes = plt.subplots(4,8,figsize=(18,10))
    #TODO: delete axis
    axes_mid = [axes[0][:2], axes[0][2:4], axes[1][:2], axes[1][2:4], 
                axes[2][:2], axes[2][2:4], axes[3][:2], axes[3][2:4]]
    axes_goal = [axes[0][4:6], axes[0][6:], axes[1][4:6], axes[1][6:],
                axes[2][4:6], axes[2][6:], axes[3][4:6], axes[3][6:]]
    row_text = ['1->2', '1-2->2', '1-2-2->2', '1-2-2-2->2',
                '2->1', '2-1->1', '2-1-1->1', '2-1-1-1->1']
    set_title = [True, True, False, False, False, False, False, False]
    pca = PCA(n_components=2)
    fig.suptitle(args.prefix)
    for i in range(8):
        if hippo_before_mid_of_interest[i] and hippo_after_mid_of_interest[i]:
            reduced_hippo = pca.fit_transform(jnp.stack(hippo_before_mid_of_interest[i]+hippo_after_mid_of_interest[i]))
            reduced_theta = pca.fit_transform(jnp.stack(theta_before_mid_of_interest[i]+theta_after_mid_of_interest[i]))
            plot_manifold_change(axes_mid[i], (reduced_hippo, reduced_theta), row_text[i], 'mid', set_title[i])
        if hippo_before_goal_of_interest[i] and hippo_after_goal_of_interest[i]:
            reduced_hippo = pca.fit_transform(jnp.stack(hippo_before_goal_of_interest[i]+hippo_after_goal_of_interest[i]))
            reduced_theta = pca.fit_transform(jnp.stack(theta_before_goal_of_interest[i]+theta_after_goal_of_interest[i]))
            plot_manifold_change(axes_goal[i], (reduced_hippo, reduced_theta), row_text[i], 'goal', set_title[i])
    fig.tight_layout()
    save_figure = args.figure_path+'/'+args.prefix+'_manifold.png'
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)
    print('save manifold to', save_figure)
    plt.savefig(save_figure)





def cal_mutual_distance(replay_trajs, replay_idxs, replay_steps, ck_function_filter):
    cluster_center = [jnp.mean(replay_trajs[idx],0) for idx in replay_idxs]
    distance_matrix = jnp.zeros((replay_steps, len(replay_idxs), len(replay_idxs)))
    ck_function_dependent_degree = jnp.zeros((replay_steps,2))
    for step in range(replay_steps):
        for i in range(len(replay_idxs)):
            for j in range(len(replay_idxs)):
                cos_similarity = replay_trajs[replay_idxs[i], step]*cluster_center[j][step]/\
                    jnp.linalg.norm(replay_trajs[replay_idxs[i], step])*jnp.linalg.norm(cluster_center[j][step])
                distance_matrix = distance_matrix.at[step,i,j].set(jnp.mean(cos_similarity))
        distance_matrix_to_be_cal = distance_matrix[step].at[(jnp.arange(len(replay_idxs)),jnp.arange(len(replay_idxs)))].set(0)
        ck_degree = (jnp.mean(distance_matrix_to_be_cal*ck_function_filter[0])-jnp.mean(distance_matrix_to_be_cal*(1-ck_function_filter[0])))/\
            jnp.std(distance_matrix_to_be_cal)
        function_degree = jnp.mean(distance_matrix_to_be_cal*ck_function_filter[1])-jnp.mean(distance_matrix_to_be_cal*(1-ck_function_filter[1]))/\
            jnp.std(distance_matrix_to_be_cal)
        ck_function_dependent_degree = ck_function_dependent_degree.at[step].set(jnp.array([ck_degree, function_degree]))
    return distance_matrix, ck_function_dependent_degree   

def plot_mutual_distance(distance_matrix, axes, titles):
    for i in range(distance_matrix.shape[0]):
        axes[i].imshow(distance_matrix[i], cmap='viridis')
        axes[i].set_title('ck:%.2f, func:%.2f'%(titles[i][0], titles[i][1]))

## calculate mutual similarity between cons and plan replays
def cal_plot_mutual_distance(args, mid_hippo, mid_theta, idxs):
    ck_function_filter = jnp.zeros((2, len(idxs), len(idxs)))
    ck_function_filter = ck_function_filter.at[(0,0,0,0), (0,1,2,3), (2,3,0,1)].set(1)
    ck_function_filter = ck_function_filter.at[(1,1,1,1), (0,1,2,3), (1,0,3,2)].set(1)

    hippo_distance_matrix, hippo_degree = cal_mutual_distance(mid_hippo, idxs, args.replay_steps, ck_function_filter)
    theta_distance_matrix, theta_degree = cal_mutual_distance(mid_theta, idxs, args.replay_steps, ck_function_filter)
    print('hippo_distance_matrix', hippo_distance_matrix)
    print('hippo_ck_function_dependent_degree', hippo_degree)
    print('theta_distance_matrix', theta_distance_matrix)
    print('theta_ck_function_dependent_degree', theta_degree)
    fig, axes = plt.subplots(2,args.replay_steps, figsize=(20,8))
    plot_mutual_distance(hippo_distance_matrix, axes[0], hippo_degree.tolist())
    plot_mutual_distance(theta_distance_matrix, axes[1], theta_degree.tolist())
    fig.tight_layout()
    fig.suptitle(args.prefix)
    if not os.path.exists('./mutual_distance'):
        os.makedirs('./mutual_distance')
    save_figure = './mutual_distance/'+args.prefix+'_mutual_distance_'+args.suffix+'.png'
    print('save mutual distance to', save_figure)
    plt.savefig(save_figure)






@partial(jax.jit, static_argnums=(0,1,4,5,6,7,8,9,10,11))
def select_consolidation_plannning(width, height, replay_traj, phase, ck0_x=env.ck0_x, ck0_y=env.ck0_y, ck1_x=env.ck1_x, ck1_y=env.ck1_y,
                                    ck0_x_g_lb=env.ck0_x_g_lb, ck0_y_g_lb=env.ck0_y_g_lb, ck1_x_g_lb=env.ck1_x_g_lb, ck1_y_g_lb=env.ck1_y_g_lb):
    """
    Choose the appropriate info (hippo_hidden/theta/theta_slow/hipp_info) based on segment represented by the trajectory.
    
    Args:
        width: `args.width`
        height: `args.height`
        replay_traj: array of replay trajectory (not trajs consisting of traj of `(start-mid-goal)`)
        phase: same as above

    Returns:
        A list of idxs that should be further processed to generate infos.

    """
    
    calculate_replay_to_scan = partial(calculate_replay, width=width, height=height, ck0_x=ck0_x, ck0_y=ck0_y, ck1_x=ck1_x, ck1_y=ck1_y,
                            ck0_x_g_lb=ck0_x_g_lb, ck0_y_g_lb=ck0_y_g_lb, ck1_x_g_lb=ck1_x_g_lb, ck1_y_g_lb=ck1_y_g_lb)
    directionality, forward_degree, score, max_segment = \
        jax.vmap(calculate_replay_to_scan,0,0)(replay_traj)
    dominant_score = jax.lax.top_k(score, 2)[0]
    # jax.debug.print('dominant_score_{a}', a=dominant_score.shape)
    significant = jnp.where((jnp.max(score, axis=-1) > 0.4) & (directionality > 0.4) & (dominant_score[:,0]>dominant_score[:,1]), 1, 0)
    phase = phase.reshape(-1)
    # jax.debug.print('significant_{a}, max_segment_{b}, phase_{c}', a=significant.shape, b=max_segment.shape, c=phase.shape)
    consolidation_ck0 = jnp.where(significant & (max_segment==0) & (phase==0), 1, 0)
    consolidation_ck1 = jnp.where(significant & (max_segment==1) & (phase==1), 1, 0)
    planning_ck0 = jnp.where(significant & (max_segment==2) & (phase==0), 1, 0)
    planning_ck1 = jnp.where(significant & (max_segment==3) & (phase==1), 1, 0)
    # jax.debug.print('consolidation_ck0_{a}, consolidation_ck1_{b}, planning_ck0_{c}, planning_ck1_{d}', a=consolidation_ck0.shape, b=consolidation_ck1.shape, c=planning_ck0.shape, d=planning_ck1.shape)
    # print('significant_proportion', jnp.mean(significant))
    # print('consolidation_ck1:',consolidation_ck1)
    # print('max_segment:',max_segment[consolidation_ck1])
    # print('score:', score[consolidation_ck1])
    # print('consolidation_ck2:',consolidation_ck2)
    # print('max_segment:',max_segment[consolidation_ck2])
    # print('score:', score[consolidation_ck2])
    # print('planning_ck1:',planning_ck1)
    # print('max_segment:',max_segment[planning_ck1])
    # print('score:', score[planning_ck1])
    # print('planning_ck2:',planning_ck2)
    # print('max_segment:',max_segment[planning_ck2])
    # print('score:', score[planning_ck2])
    return [consolidation_ck0, consolidation_ck1, planning_ck0, planning_ck1]





def cos_similarity(a, b):

    cos_sim = jnp.sum(a*b,axis=-1) / (jnp.linalg.norm(a,axis=-1) * jnp.linalg.norm(b,axis=-1))
    print(cos_sim.shape)
    return cos_sim

## calculate the similarity between the first and last point of the replay
def cal_plot_manifold_difference(args, mid_hippo, mid_theta, idxs_titles):
    idxs, titles = idxs_titles
    hippo_sim = jnp.zeros((len(idxs),))
    theta_sim = jnp.zeros((len(idxs),))
    for i in range(len(idxs)):
        hippo_of_interest = mid_hippo[idxs[i]]
        print(hippo_of_interest.shape)
        hippo_sim = hippo_sim.at[i].set(jnp.mean(cos_similarity(hippo_of_interest.at[:,0,:].get(), hippo_of_interest.at[:,-1,:].get())+1,0))
        theta_of_interest = mid_theta[idxs[i]]
        print(theta_of_interest.shape)
        theta_sim = theta_sim.at[i].set(jnp.mean(cos_similarity(theta_of_interest.at[:,0,:].get(), theta_of_interest.at[:,-1,:].get())+1,0))
    hippo_correlation = jnp.outer(hippo_sim, hippo_sim)
    theta_correlation = jnp.outer(theta_sim, theta_sim)
    all_correlation = jnp.outer(hippo_sim, theta_sim)
    fig, axes = plt.subplots(1,3, figsize=(15,8))
    axes[0].imshow(hippo_correlation)
    hippo_title = ' '.join([format(hippo_sim[i], '.2f') for i in range(len(idxs))])
    axes[0].set_title('hippo_correlation\n'+'similarity:'+hippo_title)
    axes[0].set_xticks(jnp.arange(len(idxs)), titles)
    axes[0].set_yticks(jnp.arange(len(idxs)), titles)

    axes[1].imshow(theta_correlation)
    theta_title = ' '.join([format(theta_sim[i], '.2f') for i in range(len(idxs))])
    axes[1].set_title('theta_correlation\n'+'similarity:'+theta_title)
    axes[1].set_xticks(jnp.arange(len(idxs)), titles)
    axes[1].set_yticks(jnp.arange(len(idxs)), titles)

    axes[2].imshow(all_correlation)
    axes[2].set_title('all_correlation')
    axes[2].set_xticks(jnp.arange(len(idxs)), titles)
    axes[2].set_yticks(jnp.arange(len(idxs)), titles)
    axes[2].set_xlabel('theta')
    axes[2].set_ylabel('hippo')
    
    fig.tight_layout()
    fig.suptitle(args.prefix)
    if not os.path.exists('manifold'):
        os.makedirs('manifold')
    save_figure = './manifold/'+args.prefix+'_difference_'+args.suffix+'.png'
    print('save difference to', save_figure)
    plt.savefig(save_figure)
    plt.show()