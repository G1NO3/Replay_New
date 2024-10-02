import argparse
from functools import partial
import jax
from jax import numpy as jnp
from flax import struct  # Flax dataclasses
from clu import metrics
from tensorboardX import SummaryWriter

import seaborn as sns
import env
from agent import Encoder, Hippo, Policy
import path_int
import buffer
import os 
import matplotlib.pyplot as plt
import train
import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
import pickle
import cal_plot
import imageio
import sklearn
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearnex import patch_sklearn
import config
import matplotlib.patches as mpatches
patch_sklearn()
import record


def cal_reward_times_distribution_circularly(args):
    prefix_list = ['hml_926_original',
                   'hml_926_shuffle_hipp_info']
    title_list = ['origin', 'shuffle']
    n_model = len(prefix_list)
    reward_times_met = [[] for i in range(n_model)]
    n_seed = 5
    for model_th, prefix in enumerate(prefix_list):
        args.prefix = prefix
        for j in range(n_seed):
            args.initkey = j
            hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = record.loading_record_info(args)
            reward_times_met[model_th].extend([len(hippos[0][n]) for n in range(args.n_agents)])


    reward_times_met_distribution = [jnp.histogram(jnp.array(x), bins=10, range=(0,50), density=True)[0] for x in reward_times_met]

    fig, axis = plt.subplots(1,1,figsize=(12,8))
    for j in range(len(reward_times_met_distribution)):
        axis.plot(jnp.arange(10)*5, reward_times_met_distribution[j], label=title_list[j], linewidth=3)

    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['bottom'].set_linewidth(3)
    axis.spines['left'].set_linewidth(3)
    axis.set_ylabel('Density', fontdict={'size':30})
    axis.set_xlabel('Reward times', fontdict={'size':30})
    axis.legend(fontsize=30)
    axis.tick_params(axis='both', which='major', labelsize=40, width=3, length=10)
    fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
    plt.savefig('./figures/'+args.prefix+'/step_reward_times_met_distribution.png')
    print('save reward_times_met to '+'./figures/'+args.prefix+'/step_reward_times_met_distribution.png')
    return



def compare_replay_adjacency_circularly(args):
    fig, axis = plt.subplots(1,1,figsize=(9,8))
    n_seed = 3
    originals = []
    nohpcs = []
    randoms = []
    for i in range(n_seed):
        args.initkey = i
        adjacent_step_distribution_original, adjacent_step_distribution_nohpc, adjacent_step_distribution_random, bins, dmax \
            = record.compare_replay_adjacency(args)
        originals.append(adjacent_step_distribution_original)
        nohpcs.append(adjacent_step_distribution_nohpc)
        randoms.append(adjacent_step_distribution_random)
    bin_width = dmax/bins
    originals = jnp.stack(originals, 0)
    nohpcs = jnp.stack(nohpcs, 0)
    randoms = jnp.stack(randoms, 0)
    print(originals, nohpcs, randoms)


    original_mean = originals.mean(0)
    original_std = originals.std(0)
    original_std = jnp.where(jnp.isnan(original_std), 0, original_std)
    original_expectation = jnp.sum(jnp.arange(bins)*bin_width*original_mean)
    # axis.plot(jnp.arange(bins)*bin_width, original_mean, '-', c='b', label='original')
    axis.errorbar(jnp.arange(bins)*bin_width, original_mean, original_std, fmt='-', capsize=8,
                        markersize=10, linewidth=4, capthick=4, c='b', label='original')
    axis.axvline(original_expectation, color='b', linestyle='--', linewidth=3)
    print(original_mean)
    print(original_std)

    nohpc_mean = nohpcs.mean(0)
    nohpc_std = nohpcs.std(0)
    nohpc_std = jnp.where(jnp.isnan(nohpc_std), 0, nohpc_std)
    nohpc_expectation = jnp.sum(jnp.arange(bins)*bin_width*nohpc_mean)
    axis.errorbar(jnp.arange(bins)*bin_width, nohpc_mean, nohpc_std, fmt='-', capsize=8,
                        markersize=10, linewidth=4, capthick=4, c='r', label='no pretrain')
    axis.axvline(nohpc_expectation, color='r', linestyle='--', linewidth=3)
    print(nohpc_mean)
    print(nohpc_std)
    
    random_mean = randoms.mean(0)
    random_std = randoms.std(0)
    random_std = jnp.where(jnp.isnan(random_std), 0, random_std)
    random_expectation = jnp.sum(jnp.arange(bins)*bin_width*random_mean)
    axis.errorbar(jnp.arange(bins)*bin_width, random_mean, random_std, fmt='-', capsize=8,
                        markersize=10, linewidth=4, capthick=4, c='g', label='random')  
    axis.axvline(random_expectation, color='g', linestyle='--', linewidth=3)
    print(random_mean)
    print(random_std)


    axis.legend(fontsize=35, frameon=False)
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['bottom'].set_linewidth(3)
    axis.spines['left'].set_linewidth(3)
    # axis.set_ylabel('Density', fontdict={'size':20})
    # axis.set_xlabel('Distance', fontdict={'size':20})
    axis.tick_params(axis='both', which='major', labelsize=40, width=5, length=10)
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.15, right=1)

    save_path = './figures/'+args.prefix+'/replay_distribution.png'
    plt.savefig(save_path)
    print('save replay_distribution to:',save_path)
    return


def cal_reward_times_met_circularly(args):
    # prefix_list = ['hml_926_noise_1_fromHPC',
    #                'hml_926_noise_2_fromHPC',
    #                  'hml_926_noise_3_fromHPC',
    #                     'hml_926_noise_4_fromHPC',
    #                     'hml_926_noise_5_fromHPC'
    #                ]
    # title_list = ['1st','2nd','3rd','4th','5th']
    prefix_list = ['hml_926_noreplay', 'hml_926_seed0',
                     'hml_926_seed1', 'hml_926_seed2', 'hml_926_seed3', 'hml_926_seed4',
                        'hml_926_replay2', 'hml_926_replay6']
                #    'hml_926_duplicate_0_to_5',
                #    'hml_926_duplicate_5_to_0']
    # title_list = ['origin', 'dup 1 to all','dup 5 to all']
    title_list = ['no replay', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4', '2-step', '6-step']
    n_model = len(prefix_list)
    reward_times_met = [[] for i in range(n_model)]
    n_seed = 3
    for model_th, prefix in enumerate(prefix_list):
        args.prefix = prefix
        for j in range(n_seed):
            args.initkey = j
            hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = record.loading_record_info(args)
            reward_times_met[model_th].append(jnp.array([len(hippos[0][n]) for n in range(args.n_agents)]).mean())
            print('prefix:',prefix)
            print('seed:',j)
            print(jnp.array([len(hippos[0][n]) for n in range(args.n_agents)]))
    # args.prefix = 'hml_926_original'
    # for i in range(n_seed):
    #     args.initkey = i
    #     hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = record.loading_record_info(args)
    #     reward_times_met[0].append(jnp.array([len(hippos[0][n]) for n in range(args.n_agents)]).mean())
    #     print([len(hippos[0][n]) for n in range(args.n_agents)])
    # args.prefix = 'hml_926_noise_5_fromHPC'
    # for i in range(n_seed):
    #     args.initkey = i
    #     hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = record.loading_record_info(args)
    #     reward_times_met[1].append(jnp.array([len(hippos[0][n]) for n in range(args.n_agents)]).mean())
    # args.prefix = 'hml_926_noise_34_fromHPC'
    # for i in range(n_seed):
    #     args.initkey = i
    #     hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = record.loading_record_info(args)
    #     reward_times_met[2].append(jnp.array([len(hippos[0][n]) for n in range(args.n_agents)]).mean())
    # args.prefix = 'hml_926_noise_234_fromHPC'
    # for i in range(n_seed):
    #     args.initkey = i
    #     hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = record.loading_record_info(args)
    #     reward_times_met[3].append(jnp.array([len(hippos[0][n]) for n in range(args.n_agents)]).mean())
    # args.prefix = 'hml_926_noise_fromHPC'
    # for i in range(n_seed):
    #     args.initkey = i
    #     hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = record.loading_record_info(args)
    #     reward_times_met[4].append(jnp.array([len(hippos[0][n]) for n in range(args.n_agents)]).mean())
    reward_times_met = [jnp.array(x) for x in reward_times_met]
    for i in range(len(reward_times_met)):
        change_result = cal_plot.list_ANOVA_analysis([reward_times_met[0],reward_times_met[i]])
        print('change_'+str(i)+'_reward_times_met_ANOVA_result:',change_result)
    # change_1_ANOVA_result = cal_plot.list_ANOVA_analysis(reward_times_met[:2])
    # print('change_1_reward_times_met_ANOVA_result:',change_1_ANOVA_result)
    # change_2_ANOVA_result = cal_plot.list_ANOVA_analysis(reward_times_met[1:3])
    # print('change_2_reward_times_met_ANOVA_result:',change_2_ANOVA_result)
    # change_3_ANOVA_result = cal_plot.list_ANOVA_analysis(reward_times_met[2:4])
    # print('change_3_reward_times_met_ANOVA_result:',change_3_ANOVA_result)
    # change_4_ANOVA_result = cal_plot.list_ANOVA_analysis(reward_times_met[3:])
    # print('change_4_reward_times_met_ANOVA_result:',change_4_ANOVA_result)
    reward_times_met_mean = [x.mean() for x in reward_times_met]
    reward_times_met_ste = [x.std()/jnp.sqrt(n_seed) for x in reward_times_met]
    fig, axis = plt.subplots(1,1,figsize=(16,8))
    # color_list = ['lightcoral', 'darkgreen', 'navy']
    for j in range(len(reward_times_met_mean)):
        axis.errorbar(j, reward_times_met_mean[j], reward_times_met_ste[j], fmt='-', capsize=8,
                        markersize=10, linewidth=4, capthick=4, c='b')
    axis.bar(jnp.arange(n_model), reward_times_met_mean, color='b', alpha=0.5)
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['bottom'].set_linewidth(3)
    axis.spines['left'].set_linewidth(3)
    axis.set_xticks(jnp.arange(n_model), title_list, fontdict={'size':30})
    axis.set_yticks([10,20,30,40],[10,20,30,40], fontdict={'size':30})
    # axes[0].set_ylabel('KL divergence', fontdict={'size':25})
    axis.tick_params(axis='both', which='major', labelsize=30, width=3, length=10)
    fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
    plt.savefig('./figures/'+args.prefix+'/mask_step_reward_times_met.png')
    print('save reward_times_met to '+'./figures/'+args.prefix+'/mask_step_rreward_times_met.png')
    return

def cal_forward_backward_replay_circularly(args):
    effective_forback_list = []
    n_seed = 10
    for i in range(10):
        args.initkey = i
        effective_forback_list.append(record.cal_forward_backward_replay(args))
    effective_forback_array = jnp.array(effective_forback_list)
    backward_change_1_result = cal_plot.list_ANOVA_analysis([effective_forback_array[:,0,-1],
                                                             effective_forback_array[:,1,-1]])
    print('backward_change_ANOVA_result:',backward_change_1_result)
    effective_forback_mean = effective_forback_array.mean(0)
    effective_forback_std = effective_forback_array.std(0)/jnp.sqrt(n_seed)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    xs = jnp.arange(4)
    for ax in axes:
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.set_xticks(xs, ['0','1','2','3'])
        ax.tick_params(axis='both', which='major', labelsize=40, width=3, length=10)
    effective_mean = effective_forback_mean[:,0]
    forward_mean = effective_forback_mean[:,1]
    backward_mean = effective_forback_mean[:,2]
    effective_std = effective_forback_std[:,0]
    forward_std = effective_forback_std[:,1]
    backward_std = effective_forback_std[:,2]
    # effective_std, forward_std, backward_std = effective_forback_std.T
    axes[0].errorbar(xs, effective_mean, effective_std, fmt='-', capsize=8,
                    markersize=10, linewidth=4, capthick=4, c='violet')
    axes[1].errorbar(xs, backward_mean, backward_std, fmt='-', capsize=8,
                    markersize=10, linewidth=4, capthick=4, c='violet')
    # axes[0].legend(loc='upper right', shadow=True, fontsize='x-large')
    # axes[1].legend(loc='upper right', shadow=True, fontsize='x-large')
    fig.subplots_adjust(wspace=0.5, bottom=0.35, top=0.9, left=0.2, right=0.9)
    plt.savefig('./figures/'+args.prefix+'/forward_backward_proportion.png')
    print('save forward backward proportion to ./figures/'+args.prefix+'/forward_backward_proportion.png')

    return


def eval_info_circularly(args):
    for i in range(3):
        args.initkey = i
        record.eval_info(args)

def eval_replay_circularly(args):
    optimal_count = 3
    line_chart_list = []
    step_count_list = [[] for _ in range(5)]
    for i in range(3):
        args.initkey = i
        line_chart_mid_goal, step_count = record.Igata_eval_replay(args)
        #  5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
        line_chart_list.append(line_chart_mid_goal[0])
        for j in range(len(step_count)):
            step_count_list[j].append(step_count[j])
    step_count_list = [jnp.concatenate(x) for x in step_count_list]
    change_1_ANOVA_result = cal_plot.list_ANOVA_analysis(step_count_list[:2])
    change_2_ANOVA_result = cal_plot.list_ANOVA_analysis(step_count_list[1:3])
    stablility_ANOVA_result = cal_plot.list_ANOVA_analysis([step_count_list[0], step_count_list[2], step_count_list[3], step_count_list[4]])
    print('change_1_ANOVA_result', change_1_ANOVA_result)
    print('change_2_ANOVA_result', change_2_ANOVA_result)
    print('stablility_ANOVA_result', stablility_ANOVA_result)
    step_count_mean = jnp.array([x.mean() for x in step_count_list])
    step_count_ste = jnp.array([x.std()/jnp.sqrt(len(x)) for x in step_count_list])
    fig, axis = plt.subplots(1,1,figsize=(8,8))
    axis.errorbar(jnp.arange(step_count_mean.shape[0]), step_count_mean, step_count_ste, fmt='-', capsize=8,
                    markersize=10, linewidth=4, capthick=4, c='violet')
    axis.scatter(jnp.arange(step_count_mean.shape[0]), step_count_mean, s=100, c='violet')
    axis.axhline(y=optimal_count, color='grey', linestyle='--', linewidth=3)
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['bottom'].set_linewidth(3)
    axis.spines['left'].set_linewidth(3)
    axis.set_xticks(jnp.arange(step_count_mean.shape[0]),['0','1','2','3','4'])
    # axis.set_ylabel('Step count', fontdict={'size':30})
    axis.tick_params(axis='both', which='major', labelsize=20, width=3, length=10)
    # axis.set_xlabel('Times of meeting C2', fontdict={'size':30})
    fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
    plt.savefig('./figures/'+args.prefix+'/step_count.png',dpi=300)
    print('save step count ratio to '+'./figures/'+args.prefix+'/step_count.png')
    pickle.dump({'step_count_mean':step_count_mean, 'step_count_ste':step_count_ste}, open('./figures/'+args.prefix+'/step_count.pkl','wb'))
    

    line_chart_ar = jnp.array(line_chart_list)
    print(line_chart_ar.shape)
    line_chart_mean = jnp.mean(line_chart_ar, axis=0)
    print(line_chart_mean)
    line_chart_std = jnp.std(line_chart_ar, axis=0)
    line_upper_bound = line_chart_mean + line_chart_std
    line_lower_bound = line_chart_mean - line_chart_std
    x = jnp.arange(line_chart_mean.shape[0])
    plt.close()
    color_list = ['r', 'y', 'b', 'g']
    label_list = ['past_cons', 'new_cons', 'past_plan', 'new_plan']
    fig, axis = plt.subplots(1,1,figsize=(8,4))
    for i in range(len(color_list)):
        axis.plot(x, line_chart_mean[:,i], color_list[i], label=label_list[i], linewidth=5)
        axis.fill_between(x, line_lower_bound[:,i], line_upper_bound[:,i], color=color_list[i], alpha=0.2)
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['bottom'].set_linewidth(3)
    axis.spines['left'].set_linewidth(3)
    axis.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    axis.set_yticklabels(['0','','','','0.8','1.0'], fontdict={'size':20})
    # axis.set_xticks(x, ['Pre', '', 'Learning', '', 'Post'], fontdict={'size':20})
    axis.set_xticks(x, jnp.arange(5), fontdict={'size':20})
    # axis.set_ylabel('Proportion', fontdict={'size':30})
    axis.tick_params(axis='both', which='major', labelsize=20, width=3, length=10)
    title = 'Events at open field'
    # axis.set_title(title, fontdict={'size':35})
    fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
    # axis.legend()
    plt.savefig('./figures/'+args.prefix+'/line_chart_with_error_band.png',dpi=300)
    print('save error band to '+'./figures/'+args.prefix+'/line_chart_with_error_band.png')
    pickle.dump({'line_chart_mean':line_chart_mean, 'line_chart_std':line_chart_std}, open('./figures/'+args.prefix+'/line_chart.pkl','wb'))
    return


def Ablation_KL_div(args):
    n_seed = 3
    title_list = ['no replay', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4', '2-step', '6-step']
    # color_list = ['lightcoral', 'darkgreen', 'navy']
    color_list = ['black', 'lightcoral', 'darkgreen', 'navy', 'orange', 'purple', 'red', 'blue']
    n_prefix = len(title_list)
    step_count_all = []


    args.prefix = 'hml_926_noreplay'
    line_chart_list = []
    step_count_list = [[] for _ in range(5)]
    for i in range(n_seed):
        args.initkey = i
        line_chart_mid_goal, step_count = record.Igata_eval_replay(args)
        #  5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
        # step count: 5(pre_learn, learning_1, learning_2, learning_3, post_learn)
        line_chart_list.append(line_chart_mid_goal[0])
        for j in range(len(step_count)):
            step_count_list[j].append(step_count[j])
    original_line_chart_ar = jnp.array(line_chart_list)
    # original_exploration_step_count = jnp.concatenate(step_count_list[1], axis=0)
    step_count_all.append(jnp.concatenate(step_count_list[1], axis=0))


    args.prefix = 'hml_926_seed0'
    line_chart_list = []
    step_count_list = [[] for _ in range(5)]
    for i in range(n_seed):
        args.initkey = i
        line_chart_mid_goal, step_count = record.Igata_eval_replay(args)
        #  5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
        # step count: 5(pre_learn, learning_1, learning_2, learning_3, post_learn)
        line_chart_list.append(line_chart_mid_goal[0])
        for j in range(len(step_count)):
            step_count_list[j].append(step_count[j])
    original_line_chart_ar = jnp.array(line_chart_list)
    # original_exploration_step_count = jnp.concatenate(step_count_list[1], axis=0)
    step_count_all.append(jnp.concatenate(step_count_list[1], axis=0))

    args.prefix = 'hml_926_seed1'
    line_chart_list = []
    step_count_list = [[] for _ in range(5)]
    for i in range(n_seed):
        args.initkey = i
        line_chart_mid_goal, step_count = record.Igata_eval_replay(args)
        #  5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
        line_chart_list.append(line_chart_mid_goal[0])
        for j in range(len(step_count)):
            step_count_list[j].append(step_count[j])
    noise_fromHPC_line_chart_ar = jnp.array(line_chart_list)
    # noise_fromHPC_exploration_step_count = jnp.concatenate(step_count_list[1], axis=0)
    step_count_all.append(jnp.concatenate(step_count_list[1], axis=0))

    args.prefix = 'hml_926_seed2'
    line_chart_list = []
    step_count_list = [[] for _ in range(5)]
    for i in range(n_seed):
        args.initkey = i
        line_chart_mid_goal, step_count = record.Igata_eval_replay(args)
        #  line_chart: 5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
        #  step_count: 5(pre_learn, learning_1, learning_2, learning_3, post_learn)
        line_chart_list.append(line_chart_mid_goal[0])
        for j in range(len(step_count)):
            step_count_list[j].append(step_count[j])
    noise_toHPC_line_chart_ar = jnp.array(line_chart_list)
    # noise_toHPC_exploration_step_count = jnp.concatenate(step_count_list[1], axis=0)
    step_count_all.append(jnp.concatenate(step_count_list[1], axis=0))


    args.prefix = 'hml_926_seed3'
    line_chart_list = []
    step_count_list = [[] for _ in range(5)]
    for i in range(n_seed):
        args.initkey = i
        line_chart_mid_goal, step_count = record.Igata_eval_replay(args)
        #  5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
        line_chart_list.append(line_chart_mid_goal[0])
        for j in range(len(step_count)):
            step_count_list[j].append(step_count[j])
    nomem_line_chart_ar = jnp.array(line_chart_list)
    # nomem_exploration_step_count = jnp.concatenate(step_count_list[1], axis=0)
    step_count_all.append(jnp.concatenate(step_count_list[1], axis=0))

    args.prefix = 'hml_926_seed4'
    line_chart_list = []
    step_count_list = [[] for _ in range(5)]
    for i in range(n_seed):
        args.initkey = i
        line_chart_mid_goal, step_count = record.Igata_eval_replay(args)
        #  5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
        line_chart_list.append(line_chart_mid_goal[0])
        for j in range(len(step_count)):
            step_count_list[j].append(step_count[j])
    nopred_line_chart_ar = jnp.array(line_chart_list)
    # nopred_exploration_step_count = jnp.concatenate(step_count_list[1], axis=0)
    step_count_all.append(jnp.concatenate(step_count_list[1], axis=0))

    args.prefix = 'hml_926_replay2'
    line_chart_list = []
    step_count_list = [[] for _ in range(5)]
    for i in range(n_seed):
        args.initkey = i
        line_chart_mid_goal, step_count = record.Igata_eval_replay(args)
        #  5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
        line_chart_list.append(line_chart_mid_goal[0])
        for j in range(len(step_count)):
            step_count_list[j].append(step_count[j])
    nohpc_line_chart_ar = jnp.array(line_chart_list)
    # nohpc_exploration_step_count = jnp.concatenate(step_count_list[1], axis=0)
    step_count_all.append(jnp.concatenate(step_count_list[1], axis=0))

    args.prefix = 'hml_926_replay6'
    line_chart_list = []
    step_count_list = [[] for _ in range(5)]
    for i in range(n_seed):
        args.initkey = i
        line_chart_mid_goal, step_count = record.Igata_eval_replay(args)
        #  5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
        line_chart_list.append(line_chart_mid_goal[0])
        for j in range(len(step_count)):
            step_count_list[j].append(step_count[j])
    nohpc_line_chart_ar = jnp.array(line_chart_list)
    # nohpc_exploration_step_count = jnp.concatenate(step_count_list[1], axis=0)
    step_count_all.append(jnp.concatenate(step_count_list[1], axis=0))

    fig, axes = plt.subplots(1,1,figsize=(15,6))
    # biology_line_chart_mean = jnp.load('./figures/biology_line_chart_mean.npy')
    # kl_list = [[] for _ in range(n_prefix)]
    #  10 * 5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
    # biology: 5*4
    # for j in range(len(original_line_chart_ar[0])):
    #     kl_list[0].append(jnp.nansum(original_line_chart_ar[:,j] * jnp.log(original_line_chart_ar[:,j] / biology_line_chart_mean[j]),axis=1))
    #     kl_list[1].append(jnp.nansum(noise_fromHPC_line_chart_ar[:,j] * jnp.log(noise_fromHPC_line_chart_ar[:,j] / biology_line_chart_mean[j]),axis=1))
        # kl_list[2].append(jnp.nansum(noise_toHPC_line_chart_ar[:,j] * jnp.log(noise_toHPC_line_chart_ar[:,j] / biology_line_chart_mean[j]),axis=1))
        # kl_list[1].append(jnp.nansum(nomem_line_chart_ar[:,j] * jnp.log(nomem_line_chart_ar[:,j] / biology_line_chart_mean[j]),axis=1))
        # kl_list[2].append(jnp.nansum(nopred_line_chart_ar[:,j] * jnp.log(nopred_line_chart_ar[:,j] / biology_line_chart_mean[j]),axis=1))
        # kl_list[3].append(jnp.nansum(nohpc_line_chart_ar[:,j] * jnp.log(nohpc_line_chart_ar[:,j] / biology_line_chart_mean[j]),axis=1))
    # 3 * 5 * 10
    # kl_ar = jnp.array(kl_list)[:,1:,:].reshape(n_prefix,-1)
    # print(kl_ar)
    # kl_ar = jnp.array(kl_list)[:,1:,:].reshape(3,-1)
    # print('KL')
    # change_1_ANOVA_result = cal_plot.ar_ANOVA_analysis(kl_ar[:2])
    # print('kl, original and from HPC', change_1_ANOVA_result)
    # change_2_ANOVA_result = cal_plot.ar_ANOVA_analysis(kl_ar[[0,2],:])
    # print('kl, original and to HPC', change_2_ANOVA_result)
    # change_1_ANOVA_result = cal_plot.ar_ANOVA_analysis(kl_ar[:2])
    # print('original and nomem', change_1_ANOVA_result)
    # change_2_ANOVA_result = cal_plot.ar_ANOVA_analysis(kl_ar[[0,2],:])
    # print('original and nopred', change_2_ANOVA_result)
    # change_3_ANOVA_result = cal_plot.ar_ANOVA_analysis(kl_ar[[0,3],:])
    # print('original and nohpc', change_3_ANOVA_result)
    # kl_mean = kl_ar.mean(axis=1)
    # kl_std = kl_ar.std(axis=1)/jnp.sqrt(n_seed)
    # kl_mean, kl_std, step_count_mean, step_count_std = pickle.load(open('./figures/'+args.prefix+'/Ablation_KL_div.pkl','rb')).values()
    # for j in range(len(kl_mean)):
    #     axes[0].errorbar(jnp.arange(n_prefix)[j], kl_mean[j], kl_std[j], fmt='-', capsize=8,
    #                     markersize=10, linewidth=4, capthick=4, c=color_list[j])
    #     axes[0].scatter(jnp.arange(n_prefix)[j], kl_mean[j], s=100, c=color_list[j])
    # axes[0].plot(kl_ar[0], linewidth=5, markersize=10, c='lightcoral')
    # axes[0].plot(kl_ar[1], linewidth=5, markersize=10, c='darkgreen')
    # axes[0].plot(kl_ar[2], linewidth=5, markersize=10, c='navy')
    # axes[0].spines['top'].set_color('none')
    # axes[0].spines['right'].set_color('none')
    # axes[0].spines['bottom'].set_linewidth(3)
    # axes[0].spines['left'].set_linewidth(3)
    # axes[0].set_xticks(jnp.arange(n_prefix),title_list, fontdict={'size':30}, rotation=45)
    # axes[0].set_ylabel('KL divergence', fontdict={'size':25})
    # axes[0].tick_params(axis='both', which='major', labelsize=40, width=3, length=10)
    
    # step_count_all = [original_exploration_step_count, noise_fromHPC_exploration_step_count]
    # step_count_all = [original_exploration_step_count, nomem_exploration_step_count, nopred_exploration_step_count]
    # change_1_ANOVA_result = cal_plot.list_ANOVA_analysis(step_count_all[:2])
    # print('step count, original and from HPC', change_1_ANOVA_result)
    # change_2_ANOVA_result = cal_plot.list_ANOVA_analysis([step_count_all[0], step_count_all[2]])
    # print('step count, original and to HPC', change_2_ANOVA_result)
    # print('original:',step_count_all[0])
    # print('no mem:',step_count_all[1])
    for j in range(len(step_count_all)):
        print(j)
        change_ANOVA_result = cal_plot.list_ANOVA_analysis([step_count_all[0], step_count_all[j]])
        print('original and ',title_list[j],change_ANOVA_result)
    step_count_mean = [x.mean() for x in step_count_all]
    step_count_std = [x.std()/jnp.sqrt(len(step_count_all[0])) for x in step_count_all]

    print('step_count')
    # change_1_ANOVA_result = cal_plot.list_ANOVA_analysis(step_count_all[:2])
    # print('original and nomem', change_1_ANOVA_result)
    # change_2_ANOVA_result = cal_plot.list_ANOVA_analysis([step_count_all[0], step_count_all[2]])
    # print('original and nopred', change_2_ANOVA_result)
    # change_3_ANOVA_result = cal_plot.list_ANOVA_analysis([step_count_all[0], step_count_all[3]])
    # print('original and nohpc', change_3_ANOVA_result)
    for j in range(len(step_count_mean)):
        axes.errorbar(jnp.arange(len(step_count_mean))[j], step_count_mean[j], step_count_std[j], fmt='-', capsize=8,
                        markersize=10, linewidth=4, capthick=4, c=color_list[j])
        axes.scatter(jnp.arange(len(step_count_mean))[j], step_count_mean[j], s=100, c=color_list[j])
    axes.spines['top'].set_color('none')
    axes.spines['right'].set_color('none')
    axes.spines['bottom'].set_linewidth(3)
    axes.spines['left'].set_linewidth(3)
    axes.set_xticks(jnp.arange(len(step_count_mean)),title_list, fontdict={'size':20}, rotation=45)
    # axes[1].set_ylabel('Ratio to optimal', fontdict={'size':25})
    axes.set_xlim(-0.5,0.5+len(step_count_mean))
    axes.tick_params(axis='both', which='major', labelsize=20, width=3, length=10)
    # axes[1].axhline(y=3, color='grey', linestyle='--', linewidth=3)
    fig.subplots_adjust(wspace=0.8, bottom=0.45, top=0.9, left=0.2, right=0.9)
    plt.savefig('./figures/'+args.prefix+'/Ablation_KL_div.png')
    # pickle.dump({'kl_mean':kl_mean, 'kl_std':kl_std, 'step_count_mean':step_count_mean, 'step_count_std':step_count_std}, open('./figures/'+args.prefix+'/Ablation_KL_div.pkl','wb'))
    pickle.dump({'step_count_mean':step_count_mean, 'step_count_std':step_count_std}, open('./figures/'+args.prefix+'/Ablation_KL_div.pkl','wb'))
    # print('original n:',original_line_chart_ar.shape[1])
    # print('nomem n:', nomem_line_chart_ar.shape[1])
    # print('nopred n:', nopred_line_chart_ar.shape[1])
    # print('nohpc n:', nohpc_line_chart_ar.shape[1])
 

def decoding_phase_circularly(args):
    args.info_type = 'replay_phase'
    replay_phase_mid_acc_list = []
    for i in range(10):
        args.initkey = i
        replay_phase_quantities = record.decoding_phase(args)
        replay_phase_mid_acc_list.append(replay_phase_quantities[0][1:3,[0,1,3],2:])
        # 只保留mid，删去00,0111，删去theta_slow，删去前两步
    replay_phase_mid_acc_ar = jnp.array(replay_phase_mid_acc_list).transpose(0,2,1,3)
    print(replay_phase_mid_acc_ar.shape)
    replay_phase_mid_acc_mean = jnp.mean(replay_phase_mid_acc_ar, axis=0)
    print(replay_phase_mid_acc_mean)
    replay_phase_mid_acc_std = jnp.std(replay_phase_mid_acc_ar, axis=0)/jnp.sqrt(10)
    # replay_phase_mid_acc_mean = pickle.load(open('./figures/'+args.prefix+'/replay_phase_mid_acc.pkl','rb'))['replay_phase_mid_acc_mean']
    # replay_phase_mid_acc_std = pickle.load(open('./figures/'+args.prefix+'/replay_phase_mid_acc.pkl','rb'))['replay_phase_mid_acc_std']
    replay_phase_mid_acc_upper_bound = replay_phase_mid_acc_mean + replay_phase_mid_acc_std
    replay_phase_mid_acc_lower_bound = replay_phase_mid_acc_mean - replay_phase_mid_acc_std
    # 3 * 2 * 6
    x = jnp.arange(replay_phase_mid_acc_mean.shape[-1])
    plt.close()
    fig, axes = plt.subplots(1,3, figsize=(15,4)) 
    axes = axes.flatten()
    title_list = ['hippo', 'theta', 'hipp info']
    color_all = [['darkblue', 'skyblue'],['peru', 'gold'],['darkred', 'coral']]
    label_list = ['1st', '2nd']
    for i in range(replay_phase_mid_acc_mean.shape[0]):
        color_list = color_all[i]
        if i!=2:
            for t in range(len(color_list)):
                axes[i].plot(x, replay_phase_mid_acc_mean[i,t,:], color_list[t], linewidth=1, label=label_list[t])
                axes[i].fill_between(x, replay_phase_mid_acc_lower_bound[i,t,:], replay_phase_mid_acc_upper_bound[i,t,:], color=color_list[t], alpha=0.5, linewidth=1)
        else:
            for t in range(len(color_list)):
                axes[i].plot(x[1:], replay_phase_mid_acc_mean[i,t,1:], color_list[t], linewidth=1, label=label_list[t])
                axes[i].fill_between(x[1:], replay_phase_mid_acc_lower_bound[i,t,1:], replay_phase_mid_acc_upper_bound[i,t,1:], color=color_list[t], alpha=0.5, linewidth=1)

        axes[i].spines['top'].set_color('none')
        axes[i].spines['right'].set_color('none')
        axes[i].spines['bottom'].set_linewidth(3)
        axes[i].spines['left'].set_linewidth(3)
        axes[i].set_ylim(-0.1,1.1)
        axes[i].set_yticks([0,0.2,0.4,0.6,0.8,1.0],['0','','','','','1.0'], fontdict={'size':40})
        
        axes[i].set_xticks([1,4],['start','end'], fontdict={'size':30})
        # axes[i].axvline(x=0.5, color='red', linestyle='--', linewidth=3)
        # axes[i].axvline(x=4.5, color='red', linestyle='--', linewidth=3)
        axes[i].tick_params(axis='both', which='major', labelsize=30, width=3, length=10)

        legends = axes[i].legend(frameon=False, fontsize=20, loc='lower right')
        for leg_line in legends.get_lines():
            leg_line.set_linewidth(5)
        # axes[t].legend(fontsize=20)
        # axes[i].set_title(title_list[i],fontdict={'size':35})
    # axes[0].set_ylabel('accuracy', fontdict={'size':30})
    # legend_hippo = mpatches.Patch(color='blue', label='hippo')
    # legend_theta = mpatches.Patch(color='orange', label='theta')
    # legend_hipp_info = mpatches.Patch(color='red', label='hipp_info')
    # axes[-1].legend(handles=[legend_hippo, legend_theta, legend_hipp_info], fontsize=20)
    # axes[-1].axis('off')
    # title = args.prefix+'_replay_phase_mid_acc_with_error_band\n'
    # fig.suptitle(title)

    fig.subplots_adjust(wspace=0.25,bottom=0.3, top=0.8, left=0.2, right=0.9)
    plt.savefig('./figures/'+args.prefix+'/replay_phase_mid_acc_with_error_band.png',dpi=300)
    print('save mid_phase error band to '+'./figures/'+args.prefix+'/replay_phase_mid_acc_with_error_band.png')
    pickle.dump({'replay_phase_mid_acc_mean':replay_phase_mid_acc_mean, 'replay_phase_mid_acc_std':replay_phase_mid_acc_std}, open('./figures/'+args.prefix+'/replay_phase_mid_acc.pkl','wb'))
    #eval_info = record(args)
        

def decoding_action_circularly(args):
    args.info_type = 'replay_action'
    steps_to_predict = 4
    replay_action_mid_acc_list = [[] for _ in range(steps_to_predict)]
    # replay_action_mid_acc_list_second_meet = [[] for _ in range(steps_to_predict)]
    n_seed = 10
    for i in range(n_seed):
        args.initkey = i
        replay_action_quantities_by_steps = record.decoding_action_plan(args)
        for st in range(steps_to_predict):
            replay_action_mid_acc_list[st].append(replay_action_quantities_by_steps[st][0][:,1,2:].mean(0))
            # replay_action_mid_acc_list_second_meet[st].append(replay_action_quantities_by_steps[st][1][2:,1,2:].mean(0))
            # 只保留mid，只保留theta，删去前两步
    # [3,10,4,6]

    # replay_action_curves_meets = [replay_action_mid_acc_list, replay_action_mid_acc_list_second_meet]


    replay_action_mid_acc_ar = jnp.array(replay_action_mid_acc_list)
    print(replay_action_mid_acc_ar.shape)
    replay_action_mid_acc_mean = jnp.mean(replay_action_mid_acc_ar, axis=1)
    # [3,6]
    print(replay_action_mid_acc_mean)
    replay_action_mid_acc_std = jnp.std(replay_action_mid_acc_ar, axis=1)/jnp.sqrt(n_seed)
    # replay_action_mid_acc_mean = pickle.load(open('./figures/'+args.prefix+'/replay_action_mid_acc.pkl','rb'))['replay_action_mid_acc_mean']
    # replay_action_mid_acc_std = pickle.load(open('./figures/'+args.prefix+'/replay_action_mid_acc.pkl','rb'))['replay_action_mid_acc_std']
    replay_action_mid_acc_upper_bound = replay_action_mid_acc_mean + replay_action_mid_acc_std
    replay_action_mid_acc_lower_bound = replay_action_mid_acc_mean - replay_action_mid_acc_std
    x = jnp.arange(args.replay_steps+2)
    plt.close()
    fig, axis = plt.subplots(figsize=(6,6))

    color_list = ['red','orange', 'gold','brown']
    label_list = ['t+1','t+2','t+3','t+4']
    for st in range(steps_to_predict):
        axis.plot(x, replay_action_mid_acc_mean[st,:], color=color_list[st], label=label_list[st], linewidth=1)
        axis.fill_between(x, replay_action_mid_acc_lower_bound[st,:], replay_action_mid_acc_upper_bound[st,:], color=color_list[st], alpha=1, linewidth=1)
    legends = axis.legend(frameon=False, fontsize=25, loc='upper center')
    for leg_line in legends.get_lines():
        leg_line.set_linewidth(5)
    axis.set_yticks([0.2,0.4,0.6,0.8,1.0],['0.2','','','','1.0'], fontdict={'size':40})
    axis.set_ylim(0.1,1.1)
    axis.set_xticks([1,4],['start','end'])
    axis.tick_params(axis='both', which='major', labelsize=40, width=3, length=10)
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['bottom'].set_linewidth(4)
    axis.spines['left'].set_linewidth(4)

    # leg = plt.legend()
    # leg.get_lines()[0].set_linewidth(10)
    # axis.set_ylabel('Error', fontdict={'size':35}, labelpad=-10)
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.2, right=0.9)
    plt.savefig('./figures/'+args.prefix+'/replay_action_mid_acc_with_error_band.png')
    print('save mid_action error band to '+'./figures/'+args.prefix+'/replay_action_mid_acc_with_error_band.png')
    pickle.dump({'replay_action_mid_acc_mean':replay_action_mid_acc_mean, 'replay_action_mid_acc_std':replay_action_mid_acc_std}, open('./figures/'+args.prefix+'/replay_action_mid_acc.pkl','wb'))


def eval_value_map_circularly(args):

    mid_value_map_list = [[[] for i in range(5)]]

    # for i in range(10):
    #     args.initkey = i
    #     value_map_switch = record.Igata_select_value_map_of_interest(args)
    #     for j in range(5):
    #         mid_value_map_list[0][j].append(value_map_switch[0][j])

    # for t in range(5):
    #     mid_value_map_list[0][t] = jnp.concatenate(mid_value_map_list[0][t],0)
    #     print(mid_value_map_list[0][t].shape)

    value_map_all_mean, value_advantage, delta_value_advantage, \
        outer_diff_mean, outer_diff_std = cal_plot.cal_plot_value_map(args, mid_value_map_list)


def eval_manifold_circularly(args):
    # theta_list = []
    # hippo_list = []
    # seed_len_list = []
    # mid_len_list = []
    # switch_mid_idx_list = []
    # hist_info_list = []
    # n_component = 3
    # reducer = PCA(n_components=n_component)
    # fig = plt.figure(figsize=(20,5))
    # n_seed = 10
    # for i in range(n_seed):
    #     args.initkey = i
    #     mid_hippo, goal_hippo, mid_theta, goal_theta, hist_hippo, hist_theta, reward_n, \
    #         hist_phase, hist_state = record.select_info_for_manifold(args)  
             
    #     switch_mid_idx = [0]+[len(x) for x in mid_hippo]
    #     switch_mid_idx = jnp.cumsum(jnp.array(switch_mid_idx))
    #     switch_mid_idx_list.append(switch_mid_idx)
    #     mid_hippo = jnp.concatenate(mid_hippo)
    #     goal_hippo = jnp.concatenate(goal_hippo)
    #     mid_theta = jnp.concatenate(mid_theta)
    #     goal_theta = jnp.concatenate(goal_theta)
    #     print()
    #     hist_mid_theta = jnp.concatenate((hist_theta.reshape(-1, args.theta_hidden_size),
    #         mid_theta.reshape(-1, args.theta_hidden_size)))
    #     hist_mid_hippo = jnp.concatenate((hist_hippo.reshape(-1, args.hidden_size),
    #         mid_hippo.reshape(-1, args.hidden_size)))
    #     theta_list.append(hist_mid_theta)
    #     hippo_list.append(hist_mid_hippo)
    #     seed_len_list.append(len(hist_mid_theta))
    #     mid_len_list.append(len(mid_theta))
    #     hist_info_list.append((hist_state, hist_phase, reward_n))

    # seed_idx_list = jnp.cumsum(jnp.array([0]+seed_len_list))
    # print('seed_idx_list:',seed_idx_list)
    # reduced_theta = reducer.fit_transform(jnp.concatenate(theta_list))
    # print('reduced_theta:',reduced_theta.shape)
    # reduced_hippo = reducer.fit_transform(jnp.concatenate(hippo_list))
    # print('reduced_hippo:',reduced_hippo.shape)


    # latent_theta_trajectory_00_list = []
    # latent_theta_trajectory_01_list = []
    # latent_theta_trajectory_11_list = []
    # theta_scatter_00_list = [[],[],[]]
    # theta_scatter_01_list = [[],[],[]]
    # theta_scatter_11_list = [[],[],[]]
    # for j in range(4):
    #     if n_component == 3:
    #         ax_theta = fig.add_subplot(1,4,j+1, projection='3d') 
    #     elif n_component == 2:
    #         ax_theta = fig.add_subplot(1,4,j+1)
    # axes = fig.get_axes()
    # for i in range(n_seed):
    #     args.initkey = i
    #     switch_mid_idx = switch_mid_idx_list[i]
    #     hist_mid_theta_reduced = reduced_theta[seed_idx_list[i]:seed_idx_list[i+1]]
    #     hist_theta_reduced = hist_mid_theta_reduced[:args.total_eval_steps*args.n_agents].reshape(args.total_eval_steps, args.n_agents, n_component)
    #     mid_theta_reduced = hist_mid_theta_reduced[args.total_eval_steps*args.n_agents:].reshape(mid_len_list[i], args.replay_steps, n_component)

    #     hist_mid_hippo_reduced = reduced_hippo[seed_idx_list[i]:seed_idx_list[i+1]]
    #     hist_hippo_reduced = hist_mid_hippo_reduced[:args.total_eval_steps*args.n_agents].reshape(args.total_eval_steps, args.n_agents, n_component)
    #     mid_hippo_reduced = hist_mid_hippo_reduced[args.total_eval_steps*args.n_agents:].reshape(mid_len_list[i], args.replay_steps, n_component)
        
    #     print(mid_len_list[i],args.replay_steps*mid_len_list[i],'==',mid_theta_reduced.shape)
    #     hist_state, hist_phase, reward_n = hist_info_list[i]
    #     hist_pos = hist_state[:,:,:2]
    #     hist_hippo_start_mid, hist_theta_start_mid, hist_hippo_mid_goal, hist_theta_mid_goal = \
    #             cal_plot.cal_hist_centroid_scatter(args, hist_state, hist_hippo_reduced, hist_theta_reduced, hist_phase, reward_n, hist_pos)
    #     hist_theta_start_mid_centroid = [x.mean(0) for x in hist_theta_start_mid]
    #     hist_theta_mid_goal_centroid = [x.mean(0) for x in hist_theta_mid_goal]
    #     hist_hippo_start_mid_centroid = [x.mean(0) for x in hist_hippo_start_mid]
    #     hist_hippo_mid_goal_centroid = [x.mean(0) for x in hist_hippo_mid_goal]

    #     mid_c = (jnp.arange(args.replay_steps)/2+1).reshape(1,-1).repeat(mid_len_list[i],0)

    #     colormaps = ['Reds','Blues']
    #     for j in range(4):
    #         mid_theta_of_interest = mid_theta_reduced[switch_mid_idx[j]:switch_mid_idx[j+1]]
    #         mid_theta_centroid = mid_theta_reduced[switch_mid_idx[j]:switch_mid_idx[j+1]].mean(0)
    #         print(j, j, j)
    #         print(mid_theta_of_interest.shape)
    #         print(mid_theta_centroid.shape)
    #         # mid_c_of_interest = mid_c[mid_idx_of_interest].reshape(-1)
    #         # mid_c_of_interest = mid_c_of_interest.at[0].set(0)
    #         point_size = 2
            
    #         if j==0:
    #             theta_scatter_00_list[0].append(hist_theta_start_mid[0])
    #             theta_scatter_00_list[1].append(mid_theta_of_interest)
    #             theta_scatter_00_list[2].append(hist_theta_mid_goal[0])
                
    #             print('hist_theta_start_mid_centroid:',hist_theta_start_mid_centroid[0].shape)
    #             print('mid_theta_centroid:',mid_theta_centroid.shape)
    #             print('hist_theta_mid_goal_centroid:',hist_theta_mid_goal_centroid[0].shape)
    #             latent_theta_trajectory_00 = jnp.concatenate((hist_theta_start_mid_centroid[0], mid_theta_centroid, hist_theta_mid_goal_centroid[0]))
    #             latent_theta_trajectory_00_list.append(latent_theta_trajectory_00)
    #         elif j==1:
    #             theta_scatter_01_list[0].append(hist_theta_start_mid[0])
    #             theta_scatter_01_list[1].append(mid_theta_of_interest)
    #             theta_scatter_01_list[2].append(hist_theta_mid_goal[2])
    #             theta_scatter_colormap_01 = ['Reds','Greens','Blues']
    #             latent_theta_trajectory_01 = jnp.concatenate((hist_theta_start_mid_centroid[0], mid_theta_centroid, hist_theta_mid_goal_centroid[2]))
    #             latent_theta_trajectory_01_list.append(latent_theta_trajectory_01)
    #         elif j>=2:
    #             theta_scatter_11_list[0].append(hist_theta_start_mid[2])
    #             theta_scatter_11_list[1].append(mid_theta_of_interest)
    #             theta_scatter_11_list[2].append(hist_theta_mid_goal[2])
    #             theta_scatter_colormap_11 = ['Blues','Greens','Blues']
    #             latent_theta_trajectory_11 = jnp.concatenate((hist_theta_start_mid_centroid[2], mid_theta_centroid, hist_theta_mid_goal_centroid[2]))
    #             latent_theta_trajectory_11_list.append(latent_theta_trajectory_11)

    # latent_theta_trajectory_00 = jnp.stack(latent_theta_trajectory_00_list,0).mean(0)
    # print('latent_theta_trajectory_00:',latent_theta_trajectory_00.shape)
    # latent_theta_trajectory_01 = jnp.stack(latent_theta_trajectory_01_list,0).mean(0)
    # print('latent_theta_trajectory_01:',latent_theta_trajectory_01.shape)
    # latent_theta_trajectory_11 = jnp.stack(latent_theta_trajectory_11_list,0).mean(0)
    # print('latent_theta_trajectory_11:',latent_theta_trajectory_11.shape)

    # theta_scatter_00_list = [jnp.concatenate(x,0) for x in theta_scatter_00_list]
    # print('theta_scatter_00_list:',theta_scatter_00_list[0].shape)
    # theta_scatter_00_list = [x[:500] for x in theta_scatter_00_list]

    # theta_scatter_01_list = [jnp.concatenate(x,0) for x in theta_scatter_01_list]
    # print('theta_scatter_01_list:',theta_scatter_01_list[0].shape)
    # theta_scatter_01_list = [x[:500] for x in theta_scatter_01_list]

    # theta_scatter_11_list = [jnp.concatenate(x,0) for x in theta_scatter_11_list]
    # print('theta_scatter_11_list:',theta_scatter_11_list[0].shape)
    # theta_scatter_11_list = [x[:500] for x in theta_scatter_11_list]


    # start_mid_red = cal_plot.truncate_colormap(plt.get_cmap('Reds'), 0.2, 0.5)
    # start_mid_blue = cal_plot.truncate_colormap(plt.get_cmap('Blues'), 0.2, 0.5)
    # mid_goal_red = cal_plot.truncate_colormap(plt.get_cmap('Reds'), 0.5, 0.8)
    # mid_goal_blue = cal_plot.truncate_colormap(plt.get_cmap('Blues'), 0.5, 0.8)
    # replay_green = cal_plot.truncate_colormap(plt.get_cmap('Greens'), 0.2, 0.8)
    # theta_scatter_colormap_00 = [start_mid_red, replay_green, mid_goal_red]
    # theta_scatter_colormap_01 = [start_mid_red, replay_green, mid_goal_blue]
    # theta_scatter_colormap_11 = [start_mid_blue, replay_green, mid_goal_blue]

    # scatters = [theta_scatter_00_list, theta_scatter_01_list, theta_scatter_11_list, theta_scatter_11_list]
    # scatters_colormap = [theta_scatter_colormap_00, theta_scatter_colormap_01, theta_scatter_colormap_11, theta_scatter_colormap_11]

    # latent_centroid = [latent_theta_trajectory_00, latent_theta_trajectory_01, latent_theta_trajectory_11]
    
    fig = plt.figure(figsize=(20,5))
    n_component = 3
    for j in range(4):
        if n_component == 3:
            ax_theta = fig.add_subplot(1,4,j+1, projection='3d') 
        elif n_component == 2:
            ax_theta = fig.add_subplot(1,4,j+1)
    axes = fig.get_axes()
    
    point_size = 2
    scatters, scatters_colormap, latent_centroid = pickle.load(open('./figures/'+args.prefix+'/circular_manifold.pkl','rb')).values()
    print('scatters:',len(scatters))
    centroid_traj_colors = ['salmon','magenta','deepskyblue','deepskyblue']
    dye_index = [0,1,2,2]

    # print(latent_centroid)
    axes = fig.get_axes()
    for j in range(4):
        # 不同的阶段
        start_mid_scatter_c = (jnp.arange(4)).reshape(1,-1).repeat(scatters[j][0].shape[0],0)
        mid_c = (jnp.arange(args.replay_steps)).reshape(1,-1).repeat(scatters[j][1].shape[0],0)
        mid_goal_scatter_c = (jnp.arange(4)).reshape(1,-1).repeat(scatters[j][2].shape[0],0)
        print(j)
        print(scatters[j][0].shape, scatters[j][1].shape, scatters[j][2].shape)
        print(scatters[j][0].reshape(-1,n_component).transpose(1,0).shape, scatters[j][1].reshape(-1,n_component).transpose(1,0).shape, scatters[j][2].reshape(-1,n_component).transpose(1,0).shape)
        axes[j].axis('off')
        axes[j].scatter(*(scatters[j][0].reshape(-1,n_component).transpose(1,0)), c=start_mid_scatter_c.reshape(-1), cmap=scatters_colormap[j][0], s=point_size, alpha=0.2)
        axes[j].scatter(*(scatters[j][1].reshape(-1,n_component).transpose(1,0)), c=mid_c.reshape(-1), cmap=scatters_colormap[j][1], s=point_size, alpha=0.2)
        axes[j].scatter(*(scatters[j][2].reshape(-1,n_component).transpose(1,0)), c=mid_goal_scatter_c.reshape(-1), cmap=scatters_colormap[j][2], s=point_size, alpha=0.2)
        for k in range(len(latent_centroid)):
            color = 'grey' 
            if k == dye_index[j]:
                color = centroid_traj_colors[j]
                axes[j].plot(*(latent_centroid[k].transpose(1,0)), color=color, linewidth=5, zorder=100)
            else:
                axes[j].plot(*(latent_centroid[k].transpose(1,0)), color=color, linewidth=5, zorder=1)
    
    fig.subplots_adjust(wspace=0.25,bottom=0.1, top=0.9, left=0.1, right=0.9)
    plt.savefig('./figures/'+args.prefix+'/circular_manifold.png',dpi=300)
    plt.show()   
    pickle.dump({'scatters':scatters, 'scatters_colormap':scatters_colormap, 'latent_centroid':latent_centroid}, open('./figures/'+args.prefix+'/circular_manifold.pkl','wb'))
    return



def eval_subspace_dimension_circularly(args):
    titles = ['0','1','2','3']
    # mid_theta_all = [[] for _ in range(len(titles))]
    # mid_hippo_all = [[] for _ in range(len(titles))]
    # goal_theta_all = [[] for _ in range(len(titles))]
    # goal_hippo_all = [[] for _ in range(len(titles))]

    # hist_hippo_list = []
    # hist_theta_list = []
    n_seed = 1
    # for i in range(n_seed):
    #     args.initkey = i
    #     mid_hippo, goal_hippo, mid_theta, goal_theta, hist_hippo, hist_theta, reward_n, \
    #         hist_phase, hist_state = select_info_for_manifold(args)  
        
    #     for j in range(len(titles)):
    #         mid_theta_all[j].append(mid_theta[j])
    #         mid_hippo_all[j].append(mid_hippo[j])
    #         goal_theta_all[j].append(goal_theta[j])
    #         goal_hippo_all[j].append(goal_hippo[j])

    #     hist_hippo_list.append(hist_hippo)
    #     hist_theta_list.append(hist_theta)

    # mid_hippo_all = [jnp.concatenate(x) for x in mid_hippo_all]
    # goal_hippo_all = [jnp.concatenate(x) for x in goal_hippo_all]
    # mid_theta_all = [jnp.concatenate(x) for x in mid_theta_all]
    # goal_theta_all = [jnp.concatenate(x) for x in goal_theta_all]
    # hist_hippo = jnp.concatenate(hist_hippo_list)
    # hist_theta = jnp.concatenate(hist_theta_list)

    # switch_mid_idx = [0]+[len(x) for x in mid_hippo]
    # switch_mid_idx = jnp.cumsum(jnp.array(switch_mid_idx))
    evr_hist_theta_list = []
    theta_dims_list = []
    for i in range(n_seed):
        args.initkey = i
        switch_mid_hippo, switch_goal_hippo, switch_mid_theta, switch_goal_theta, hist_hippo, hist_theta, reward_n, \
            hist_phase, hist_state = record.select_info_for_manifold(args)
        evr_hist_theta, theta_dims = cal_plot.cal_plot_subspace_dimension(args, hist_hippo, hist_theta,
                    (switch_mid_hippo, switch_goal_hippo, switch_mid_theta, switch_goal_theta),
                    hist_phase, hist_state, titles)
        print('evr_hist_theta:',evr_hist_theta)
        evr_hist_theta_list.append(evr_hist_theta)
        theta_dims_list.append(theta_dims)

    evr_hist_theta_list = jnp.stack(evr_hist_theta_list,0)
    evr_hist_theta_mean = evr_hist_theta_list.mean(0) 
    evr_hist_theta_std = evr_hist_theta_list.std(0)
    theta_dims_list = jnp.stack(theta_dims_list,0)
    theta_dims_mean = theta_dims_list.mean(0)
    theta_dims_std = theta_dims_list.std(0)/jnp.sqrt(n_seed)
    plt.close()
    fig, axes = plt.subplots(2,2,figsize=(10,8))
    axes = axes.flatten()
    for axis in axes:
        axis.spines['top'].set_color('none')
        axis.spines['right'].set_color('none')
        axis.spines['bottom'].set_linewidth(3)
        axis.spines['left'].set_linewidth(3)
        axis.tick_params(axis='both', which='major', labelsize=40, width=3, length=10)
    axes[0].errorbar(jnp.arange(len(titles)), theta_dims_mean[0], yerr=theta_dims_std[0], 
                     markersize=10, linewidth=4, capsize=8, capthick=4,fmt='-')
    axes[0].set_xticks(jnp.arange(len(titles)), jnp.arange(len(titles)), fontsize=40)
    # axes[0].set_xlabel('Times of meeting checkpoint 2 repeatedly', fontsize=25)
    axes[0].set_yticks(jnp.arange(3), jnp.arange(3), fontsize=40)
    # axes[0].set_ylabel('Dimension of neural subspace', fontsize=25)
    # axes[0].set_title('Only replay', fontsize=30)

    axes[1].errorbar(jnp.arange(len(titles)), theta_dims_mean[1], yerr=theta_dims_std[1],
                        markersize=10, linewidth=4, capsize=8, capthick=4,fmt='-')
    axes[1].set_xticks(jnp.arange(len(titles)), jnp.arange(len(titles)), fontsize=40)
    # axes[1].set_xlabel('Times of meeting checkpoint 2 repeatedly', fontsize=25)
    axes[1].set_yticks(jnp.arange(4), jnp.arange(4), fontsize=40)
    # axes[1].set_ylabel('Dimension of neural subspace', fontsize=15)
    # axes[1].set_title('All activities', fontsize=30)

    n_component = 12
    print(evr_hist_theta_std)
    axes[2].plot(evr_hist_theta_mean, color='black', linewidth=3)
    axes[2].fill_between(jnp.arange(n_component), evr_hist_theta_mean-evr_hist_theta_std, evr_hist_theta_mean+evr_hist_theta_std, color='grey', alpha=1)
    axes[2].scatter(jnp.arange(n_component), evr_hist_theta_mean, color='black', s=100)
    axes[2].set_xticks(jnp.arange(n_component), jnp.arange(n_component)+1, fontsize=40)
    # axes[2].set_xlabel('Dimension', fontsize=25)
    # axes[2].set_ylabel('Accumulated Explained Variance', fontsize=25)
    axes[2].axhline(y=0.7, color='grey', linestyle='dashed', linewidth=3)
    axes[2].axhline(y=1, color='grey', linestyle='dashed', linewidth=3)
    fig.tight_layout()
    plt.savefig('./figures/'+args.prefix+'/subspace_dimension_with_errorbar_'+args.suffix+'.png')
    pickle.dump({'evr_hist_theta_mean':evr_hist_theta_mean, 'evr_hist_theta_std':evr_hist_theta_std, 'theta_dims_mean':theta_dims_mean, 'theta_dims_std':theta_dims_std}, 
                open('./figures/'+args.prefix+'/subspace_dimension_with_errorbar_'+args.suffix+'.pkl','wb'))
    plt.show()



def eval_manifold_stability_circularly(args):
    n_seed = 3
    titles = ['0','1','2','3']
    variance_hippo_start_mid_list = [[] for _ in range(len(titles))]
    variance_theta_start_mid_list = [[] for _ in range(len(titles))]
    variance_hippo_mid_goal_list = [[] for _ in range(len(titles))]
    variance_theta_mid_goal_list = [[] for _ in range(len(titles))]
    variance_hippo_list = [[] for _ in range(len(titles))]
    variance_theta_list = [[] for _ in range(len(titles))]
    for i in range(n_seed):
        args.initkey = i
        switch_mid_hippo, switch_goal_hippo, switch_mid_theta, switch_goal_theta, hist_hippo, hist_theta, reward_n, \
            hist_phase, hist_state = record.select_info_for_manifold(args)
        variance_hippo_start_mid, variance_theta_start_mid, variance_hippo_mid_goal, variance_theta_mid_goal, \
            variance_hippo, variance_theta = \
            cal_plot.cal_plot_manifold_stability(args, hist_hippo, hist_theta, reward_n,
                    (switch_mid_hippo, switch_goal_hippo, switch_mid_theta, switch_goal_theta),
                    hist_phase, hist_state, titles)
        for j in range(len(titles)):
            # variance_hippo_start_mid_list[j].append(variance_hippo_start_mid[j].mean())
            # variance_theta_start_mid_list[j].append(variance_theta_start_mid[j].mean())
            # variance_hippo_start_mid_list[j].append(variance_hippo_mid_goal[j].mean())
            # variance_theta_start_mid_list[j].append(variance_theta_mid_goal[j].mean())
            # variance_hippo_mid_goal_list[j].append(variance_hippo_mid_goal[j].mean())
            # variance_theta_mid_goal_list[j].append(variance_theta_mid_goal[j].mean())
            variance_hippo_list[j].append(variance_hippo[j].mean())
            variance_theta_list[j].append(variance_theta[j].mean())
    # variance_hippo_start_mid_list = [jnp.array(x) for x in variance_hippo_start_mid_list]
    # variance_theta_start_mid_list = [jnp.array(x) for x in variance_theta_start_mid_list]
    # variance_hippo_mid_goal_list = [jnp.array(x) for x in variance_hippo_mid_goal_list]
    # variance_theta_mid_goal_list = [jnp.array(x) for x in variance_theta_mid_goal_list]
    # variance_hippo_start_mid_mean = [jnp.nanmean(x,0) for x in variance_hippo_start_mid_list]
    # variance_theta_start_mid_mean = [jnp.nanmean(x,0) for x in variance_theta_start_mid_list]
    # variance_hippo_mid_goal_mean = [jnp.nanmean(x,0) for x in variance_hippo_mid_goal_list]
    # variance_theta_mid_goal_mean = [jnp.nanmean(x,0) for x in variance_theta_mid_goal_list]
    # variance_hippo_start_mid_std = [jnp.nanstd(x,0)/jnp.sqrt(n_seed) for x in variance_hippo_start_mid_list]
    # variance_theta_start_mid_std = [jnp.nanstd(x,0)/jnp.sqrt(n_seed) for x in variance_theta_start_mid_list]
    # variance_hippo_mid_goal_std = [jnp.nanstd(x,0)/jnp.sqrt(n_seed) for x in variance_hippo_mid_goal_list]
    # variance_theta_mid_goal_std = [jnp.nanstd(x,0)/jnp.sqrt(n_seed) for x in variance_theta_mid_goal_list]

    variance_hippo_list = [jnp.array(x) for x in variance_hippo_list]
    variance_theta_list = [jnp.array(x) for x in variance_theta_list]
    variance_hippo_mean = [jnp.nanmean(x,0) for x in variance_hippo_list]
    variance_theta_mean = [jnp.nanmean(x,0) for x in variance_theta_list]
    variance_hippo_std = [jnp.nanstd(x,0)/jnp.sqrt(n_seed) for x in variance_hippo_list]
    variance_theta_std = [jnp.nanstd(x,0)/jnp.sqrt(n_seed) for x in variance_theta_list]

    # change_1_ANOVA_result = cal_plot.list_ANOVA_analysis(variance_hippo_start_mid_list[:2])
    # change_2_ANOVA_result = cal_plot.list_ANOVA_analysis(variance_hippo_start_mid_list[1:3])
    # stable_ANOVA_result = cal_plot.list_ANOVA_analysis([variance_hippo_start_mid_list[0],variance_hippo_start_mid_list[2],variance_hippo_start_mid_list[3]])
    # print('hippo')
    # print('change_1_ANOVA_result:',change_1_ANOVA_result)
    # print('change_2_ANOVA_result:',change_2_ANOVA_result)
    # print('stable_ANOVA_result:',stable_ANOVA_result)
    # change_1_ANOVA_result = cal_plot.list_ANOVA_analysis(variance_theta_start_mid_list[:2])
    # change_2_ANOVA_result = cal_plot.list_ANOVA_analysis(variance_theta_start_mid_list[1:3])
    # stable_ANOVA_result = cal_plot.list_ANOVA_analysis([variance_theta_start_mid_list[0],variance_theta_start_mid_list[2],variance_theta_start_mid_list[3]])
    # print('theta')
    # print('change_1_ANOVA_result:',change_1_ANOVA_result)
    # print('change_2_ANOVA_result:',change_2_ANOVA_result)
    # print('stable_ANOVA_result:',stable_ANOVA_result)

    plt.close()
    fig, axes = plt.subplots(3,2, figsize=(12,12))
    axes = axes.flatten()
    for axis in axes:
        axis.spines['top'].set_color('none')
        axis.spines['right'].set_color('none')
        axis.spines['bottom'].set_linewidth(3)
        axis.spines['left'].set_linewidth(3)
        axis.tick_params(axis='both', which='major', labelsize=35, width=3, length=10)
        axis.set_xticks(jnp.arange(len(titles)), jnp.arange(len(titles)))
        # axis.set_xlabel('Times of meeting checkpoint 2 repeatedly', fontsize=25)
        # axis.set_ylabel('Variance of neural activity', fontsize=25)
    # axes[0].errorbar(jnp.arange(len(titles)), variance_hippo_start_mid_mean, yerr=variance_hippo_start_mid_std, 
    #                  markersize=100, linewidth=2, capsize=8, capthick=2,fmt='-')
    # axes[0].set_title('HPC Start-Checkpoint', fontsize=30)
    # axes[1].errorbar(jnp.arange(len(titles)), variance_theta_start_mid_mean, yerr=variance_theta_start_mid_std,
    #                     markersize=100, linewidth=2, capsize=8, capthick=2,fmt='-')
    # axes[1].set_title('PFC Start-Checkpoint', fontsize=30)
    # axes[2].errorbar(jnp.arange(len(titles)), variance_hippo_mid_goal_mean, yerr=variance_hippo_mid_goal_std,
    #                     markersize=100, linewidth=2, capsize=8, capthick=2,fmt='-')
    # axes[2].set_title('HPC Checkpoint-Goal', fontsize=30)
    # axes[3].errorbar(jnp.arange(len(titles)), variance_theta_mid_goal_mean, yerr=variance_theta_mid_goal_std,
    #                     markersize=100, linewidth=2, capsize=8, capthick=2,fmt='-')
    # axes[3].set_title('PFC Checkpoint-Goal', fontsize=30)
    axes[4].errorbar(jnp.arange(len(titles)), variance_hippo_mean, yerr=variance_hippo_std,
                        markersize=100, linewidth=2, capsize=8, capthick=2,fmt='-')
    axes[4].set_title('HF', fontsize=30)
    axes[5].errorbar(jnp.arange(len(titles)), variance_theta_mean, yerr=variance_theta_std,
                        markersize=100, linewidth=2, capsize=8, capthick=2,fmt='-')
    axes[5].set_title('PFC', fontsize=30)
    fig.tight_layout()
    plt.savefig('./figures/'+args.prefix+'/manifold_stability_with_errorbar_'+args.suffix+'.png')
    # pickle.dump({'variance_hippo_start_mid_mean':variance_hippo_start_mid_mean, 'variance_hippo_start_mid_std':variance_hippo_start_mid_std,
    #              'variance_theta_start_mid_mean':variance_theta_start_mid_mean, 'variance_theta_start_mid_std':variance_theta_start_mid_std,
    #              'variance_hippo_mid_goal_mean':variance_hippo_mid_goal_mean, 'variance_hippo_mid_goal_std':variance_hippo_mid_goal_std,
    #              'variance_theta_mid_goal_mean':variance_theta_mid_goal_mean, 'variance_theta_mid_goal_std':variance_theta_mid_goal_std}, open('./figures/'+args.prefix+'/manifold_stability.pkl','wb'))
    
    # plt.show()
    return