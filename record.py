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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import RidgeClassifier, Ridge, LinearRegression, Lasso
from sklearnex import patch_sklearn
import config
import matplotlib.patches as mpatches
patch_sklearn()
from circular_main import *
from flax.training import train_state, checkpoints
import optax
jnp.set_printoptions(precision=3,threshold=jnp.inf)

def compare_replay_adjacency(args):

    # fig, axis = plt.subplots(1,1,figsize=(10,10))
    bins = 8
    dmax = 8
    bin_width = dmax/bins

    args.prefix = 'hml_926_original'
    save_path = './figures/'+args.prefix+'/adjacent_step_distances_seed'+str(args.initkey)+'.pkl'
    adjacent_step_distribution_original = cal_replay_adjacency(args, save_path, bins, dmax)
    print(adjacent_step_distribution_original)
    # axis.plot(np.arange(bins)*bin_width, adjacent_step_distribution_original, label='original', linewidth=3)


    args.prefix = 'nohpc'
    save_path = './figures/'+args.prefix+'/adjacent_step_distances_seed'+str(args.initkey)+'.pkl'
    adjacent_step_distribution_nohpc = cal_replay_adjacency(args, save_path, bins, dmax)
    # axis.plot(np.arange(bins)*bin_width, adjacent_step_distribution_nohpc, label='no pretrain', linewidth=3)


    args.prefix = 'hml_926_original'
    save_path = './figures/'+args.prefix+'/adjacent_step_distances_random_seed'+str(args.initkey)+'.pkl'
    adjacent_step_distribution_random = cal_replay_adjacency(args, save_path, bins, dmax)
    # axis.plot(np.arange(bins)*bin_width, adjacent_step_distribution_random, label='random', linewidth=3)

    
    # axis.legend()
    # axis.spines['top'].set_color('none')
    # axis.spines['right'].set_color('none')
    # axis.spines['bottom'].set_linewidth(3)
    # axis.spines['left'].set_linewidth(3)
    # axis.set_ylabel('Density', fontdict={'size':20})
    # axis.set_xlabel('Distance', fontdict={'size':20})
    # axis.tick_params(axis='both', which='major', labelsize=20, width=3, length=10)
    # save_path = './figures/'+args.prefix+'/replay_distribution.png'
    # plt.savefig(save_path)
    # print('save replay_distribution to:',save_path)

    return adjacent_step_distribution_original, adjacent_step_distribution_nohpc, adjacent_step_distribution_random, bins, dmax

def cal_replay_adjacency(args, save_path, bins, dmax):

    def cal_distance(replay_path_once):
        # replay_path_once.shape = (replay_steps,2)
        d2_distance = jnp.sqrt(((replay_path_once[1:] - replay_path_once[:-1])**2).sum(-1))
        # shape = (replay_steps-1,)
        return d2_distance
    
    if not os.path.exists(save_path):
        hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = loading_record_info(args)
        switch_mid_infos = cal_plot.select_switch_mid_infos(args.n_agents, hippos[0], theta_fasts[0], \
                                                                        theta_slows[0], hipp_infos[0], phases[0], trajs[0])
        switch_start_goal_infos = cal_plot.select_switch_start_goal_infos(args.n_agents, hippos[1], theta_fasts[1], \
                                                                        theta_slows[1], hipp_infos[1], phases[1], trajs[1])
        replay_hippo, replay_theta, replay_theta_slow, replay_hipp_info, replay_traj = switch_mid_infos
        adjacent_step_distances = []


        # key = jax.random.PRNGKey(0)
        # random_replay_traj = []
        # for t in range(len(replay_hippo)):
            # key, subkey = jax.random.split(key)
            # random_replay_traj.append(jax.random.randint(subkey, minval=0, maxval=args.width*args.height, shape=(len(replay_hippo),args.replay_steps)))
        # replay_traj = random_replay_traj


        for t in range(len(replay_hippo)):
            replay_path = jnp.stack([replay_traj[t]//args.width, replay_traj[t]%args.width], -1)
            d2_distance = jax.vmap(cal_distance,0,0)(replay_path).reshape(-1)
            adjacent_step_distances.append(d2_distance)
        pickle.dump(adjacent_step_distances, open(save_path, 'wb'))
        print('save adjacent_step_distances to '+save_path)


    adjacent_step_distances = pickle.load(open(save_path,'rb'))
    bin_width = dmax/bins
    # print(adjacent_step_distances[t][:50])
    adjacent_step_distribution = jnp.histogram(jnp.concatenate(adjacent_step_distances,0), \
                                                bins=bins, range=(0,dmax), density=True)[0]

    return adjacent_step_distribution

def cal_recapitulate_past_experiences(args):
    hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = loading_record_info(args)
    save_path = './figures/'+args.prefix+'/record_info_'+str(args.pseudo_reward_idx)+'_seed'+str(args.initkey)
    record_info = pickle.load(open(save_path, 'rb'))
    print('load record info from:', save_path)
    hist_hippo = record_info['hist_hippo']
    mid_ei = record_info['mid']['ei']
    goal_ei = record_info['goal']['ei']
    mid_traj = record_info['mid']['traj']
    goal_traj = record_info['goal']['traj']
    mid_overlap_score, goal_overlap_score = cal_plot.cal_overlap_score(args, hist_state, mid_ei, goal_ei, mid_traj, goal_traj)
    print('mid_overlap_score:', mid_overlap_score)
    print('goal_overlap_score:', goal_overlap_score)


def cal_forward_backward_replay(args):
    hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = loading_record_info(args)
    clf_save_path = './decoder/'+args.prefix+'_unidirectional_place_fields_clf.pkl'
    if not os.path.exists(clf_save_path):
        raise ValueError('clf not exists')
    else:
        print('load clf from:', clf_save_path)
    clf = pickle.load(open(clf_save_path, 'rb'))
    switch_mid_infos = cal_plot.select_switch_mid_infos(args.n_agents, hippos[0], theta_fasts[0], \
                                                                    theta_slows[0], hipp_infos[0], phases[0], trajs[0])
    switch_start_goal_infos = cal_plot.select_switch_start_goal_infos(args.n_agents, hippos[1], theta_fasts[1], \
                                                                    theta_slows[1], hipp_infos[1], phases[1], trajs[1])
    replay_hippo, replay_theta, replay_theta_slow, replay_hipp_info, replay_traj = switch_mid_infos
    calculate_replay_to_scan = partial(cal_plot.calculate_replay, width=args.width, height=args.height, ck0_x=env.ck0_x, ck0_y=env.ck0_y, ck1_x=env.ck1_x, ck1_y=env.ck1_y,
                                        ck0_x_g_lb=env.ck0_x_g_lb, ck0_y_g_lb=env.ck0_y_g_lb, ck1_x_g_lb=env.ck1_x_g_lb, ck1_y_g_lb=env.ck1_y_g_lb)
    ts = ['00', '01' ,'011', '0111']
    effective_forward_backward_proportion = jnp.zeros((len(ts), 3))
    for t in range(len(replay_hippo)):
        print(ts[t])
        decoded_hippo = replay_hippo[t][:,4:]
        decoded_actions = clf.predict(decoded_hippo.reshape(-1, args.hidden_size)).reshape(decoded_hippo.shape[0], decoded_hippo.shape[1])
        print('decoded_actions:',decoded_actions.shape)
        decoded_direction = jnp.where((decoded_actions==1) | (decoded_actions==2), 1, -1)
        place_cell_forwardness = jnp.mean(decoded_direction, axis=1)
        print(place_cell_forwardness.shape)
        directionality, forward_degree, score, max_segment = \
            jax.vmap(calculate_replay_to_scan, 0, 0)(replay_traj[t])
        print('score:',score.shape)
        effective_proportion = jnp.where((jnp.max(score, axis=-1) > 0.5) & (directionality > 0.5), 1, 0).mean()
        print('effective proportion:', effective_proportion)
        significant = jnp.where((jnp.max(score, axis=-1) > 0.5) & (directionality > 0.5))[0]
        print('place cell forwardness', place_cell_forwardness.shape)
        # print('forward_degree:', forward_degree[:20])
        # sum_forwardness = forward_degree*place_cell_forwardness
        # sum_forwardness = forward_degree
        sum_forwardness = place_cell_forwardness * forward_degree
        # print('sum_forwardness:', sum_forwardness[:20])
        effective_forwardness = sum_forwardness[significant]
        # print('effective forwardness:', effective_forwardness[:20])
        forward_proportion = jnp.where(effective_forwardness > 0, 1, 0).mean()
        print('forward proportion:', forward_proportion)
        backward_proportion = 1 - forward_proportion
        print('backward proportion:', backward_proportion)
        effective_forward_backward_proportion = effective_forward_backward_proportion.at[t].set(jnp.array([effective_proportion, forward_proportion, backward_proportion]))
    save_path = './figures/'+args.prefix+'/forward_backward_proportion_seed_'+str(args.initkey)+'.pkl'
    # pickle.dump(all_result, open(save_path, 'wb'))
    print('save forward_backward_proportion to:',save_path)
    return effective_forward_backward_proportion

def decoding_moving_direction(args):
    save_path = './figures/'+args.prefix+'/record_random_info_'+str(args.pseudo_reward_idx)+'_seed'+str(args.initkey)
    record_random_info = pickle.load(open(save_path, 'rb'))
    print('load record random info from:', save_path)
    hist_hippo = record_random_info['hist_hippo']
    hist_state = record_random_info['hist_state']
    hist_actions = hist_state[:,:,2]

    hist_hippo_train = hist_hippo[:int(0.8*hist_hippo.shape[0])]
    hist_hippo_test = hist_hippo[int(0.8*hist_hippo.shape[0]):]
    hist_actions_train = hist_actions[:int(0.8*hist_actions.shape[0])]
    hist_actions_test = hist_actions[int(0.8*hist_actions.shape[0]):]
    clf = DecisionTreeClassifier()
    print(hist_hippo_train[:20])
    print(hist_actions_train[:20])
    clf = clf.fit(hist_hippo_train.reshape(-1, args.hidden_size), hist_actions_train.reshape(-1))
    print('train score:', clf.score(hist_hippo_train.reshape(-1, args.hidden_size), hist_actions_train.reshape(-1)))
    print('test score:', clf.score(hist_hippo_test.reshape(-1, args.hidden_size), hist_actions_test.reshape(-1)))
    save_path = './decoder/'+args.prefix+'_unidirectional_place_fields_clf.pkl'
    pickle.dump(clf, open(save_path, 'wb'))
    print('save uni-directional place fields clf to:', save_path)
    return

def eval_info_over_random_policy(args):
    # Initialize key ================================================
    key = jax.random.PRNGKey(0)
    # Initialize env and place_cell ================================================
    key, subkey = jax.random.split(key)
    obs, env_state = env.reset(args.width, args.height, args.n_agents, subkey)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (args.n_agents, 1), minval=0, maxval=4)  # [n, 1]
    obs, rewards, done, env_state = env.step(env_state, actions)
    # Initialize model and training_state ============================
    key, subkey = jax.random.split(key)
    encoder = Encoder()
    init_samples = [jnp.zeros((args.n_agents, args.height, args.width), dtype=jnp.int8),
                    jnp.zeros((args.n_agents, 1), dtype=jnp.int8)]
    params = encoder.init(subkey, *init_samples)['params']

    running_encoder_state = path_int.TrainState.create(
        apply_fn=encoder.apply, params=params, tx=optax.adamw(0.0, weight_decay=0.0),
        metrics=path_int.Metrics.empty())

    if args.load_encoder != None:
        load_encoder = args.model_path + '/' + args.load_encoder
    else:
        encoder_path = args.model_path + '/' + args.prefix + '_encoder'
        load_encoder = encoder_path + '/' + os.listdir(encoder_path)[0]

    if os.path.exists(load_encoder):
        print('load encoder from:', load_encoder)
    else:
        print('path not exists:', load_encoder)
        print('randomly initialize encoder')
    running_encoder_state = checkpoints.restore_checkpoint(ckpt_dir=load_encoder,
                                                            target=running_encoder_state)
    # Load Hippo ===========================================================================
    obs_embed, action_embed = running_encoder_state.apply_fn({'params': params}, *init_samples)
    key, subkey = jax.random.split(key)
    hippo = Hippo(output_size=args.height * args.width + 1 + args.hippo_mem_len,
                  hidden_size=args.hidden_size)
    hidden = jnp.zeros((args.n_agents, args.hidden_size))
    pfc_input = jnp.zeros((args.n_agents, 8))
    params = hippo.init(subkey, hidden, pfc_input, (obs_embed, action_embed), jnp.zeros((args.n_agents, 1)))['params']

    running_hippo_state = path_int.TrainState.create(
        apply_fn=hippo.apply, params=params, tx=optax.adamw(0.0, weight_decay=0.0),
        metrics=path_int.Metrics.empty())

    if args.load_hippo is not None:
        load_hippo = args.model_path + '/' + args.load_hippo
    else:
        hippo_path = args.model_path + '/' + args.prefix + '_hippo'
        load_hippo = hippo_path + '/' + os.listdir(hippo_path)[0]

    if os.path.exists(load_hippo):
        print('load hippo from:', load_hippo)
    else:
        print('path not exists:', load_hippo)
        print('randomly initialize hippo')
    running_hippo_state = checkpoints.restore_checkpoint(ckpt_dir=load_hippo,
                                                            target=running_hippo_state)
    

    hist_actions = jax.random.randint(subkey, (args.total_eval_steps, args.n_agents,1),
                                      minval=0, maxval=args.n_action, dtype=jnp.int32)
    print('hist_actions:',hist_actions)
    hippo_hidden = jnp.zeros((args.n_agents, args.hidden_size))
    hist_hippo = jnp.zeros((args.total_eval_steps, *hippo_hidden.shape))
    hist_state = jnp.zeros((args.total_eval_steps, args.n_agents, 5))
    
    for ei in range(args.total_eval_steps):
        print(ei+1)
        key, subkey = jax.random.split(key)
        actions = hist_actions[ei]
        st = env_state['current_pos']
        at = actions
        # phase = env_state['phase']
        # step_count = env_state['step_count']
        obs, rewards, done, env_state = env.step(env_state, actions)
        rt = rewards
        hist_state = hist_state.at[ei].set(jnp.concatenate([st, at, rt, done], axis=-1))
        # put_to_buffer: o_t, r_t-1, action_t-1, s_t
        # put to buffer [obs_t, a_t-1, pos_t, reward_t, checked_t]
        key, subkey = jax.random.split(key)
        env_state = env.reset_reward(env_state, rewards, subkey)  # fixme: reset reward
        key, subkey = jax.random.split(key)
        mask = jax.random.uniform(subkey, (obs.shape[0], 1, 1))
        obs_incomplete = jnp.where(obs == 2, 0, obs)
        obs_incomplete = jnp.where(mask < args.visual_prob, obs_incomplete, 0)
        # obs[n, h, w], actions[n, 1], rewards[n, 1]
        # Encode obs and a_t-1 ===============================================================================
        obs_embed, action_embed = running_encoder_state.apply_fn({'params': running_encoder_state.params}, obs_incomplete, actions)
        # Update hippo_hidden ==================================================================================
        new_hippo_hidden, _ = running_hippo_state.apply_fn({'params': running_hippo_state.params},
                                               hippo_hidden, jnp.zeros((args.n_agents, 8)),
                                               (obs_embed, action_embed), rewards)
        reset_hippo_hidden = jnp.where(done & jnp.isclose(rewards, 0.), jnp.zeros(new_hippo_hidden.shape), new_hippo_hidden)

        hippo_hidden = reset_hippo_hidden
        hist_hippo = hist_hippo.at[ei].set(hippo_hidden)

    record_random_info = {'args': args,
                    'hist_hippo': hist_hippo, 'hist_state': hist_state}
    # mid_hippo is a list of (length, replay_steps+2, hidden_size) array
    # 后期优化的时候可以给每个prefix单独开一个文件夹来存
    if not os.path.exists('./figures/'+args.prefix):
        os.mkdir('./figures/'+args.prefix)
    save_path = './figures/'+args.prefix+'/record_random_info_'+str(args.pseudo_reward_idx)+'_seed'+str(args.initkey)
    pickle.dump(record_random_info, open(save_path, 'wb'))
    print('save record random info to:', save_path)

    return record_random_info



def eval_manifold(args):
    titles = ['0','1','2','3']
    switch_mid_hippo, switch_goal_hippo, switch_mid_theta, switch_goal_theta, hist_hippo, hist_theta, reward_n, \
        hist_phase, hist_state = select_info_for_manifold(args)
    # print(switch_mid_hippo.shape, switch_goal_hippo.shape, switch_mid_theta.shape, switch_goal_theta.shape)
    # cal_plot.cal_plot_manifold(args, hist_hippo, hist_theta, reward_n,
    #                 (switch_mid_hippo, switch_goal_hippo, switch_mid_theta, switch_goal_theta),
    #                 hist_phase, hist_state, titles)
    # cal_plot.cal_plot_subspace_dimension(args, hist_hippo, hist_theta,
    #                 (switch_mid_hippo, switch_goal_hippo, switch_mid_theta, switch_goal_theta),
    #                 hist_phase, hist_state, titles)
    variance_hippo_start_mid, variance_theta_start_mid, variance_hippo_mid_goal, variance_theta_mid_goal = \
        cal_plot.cal_plot_manifold_stability(args, hist_hippo, hist_theta, reward_n,
                    (switch_mid_hippo, switch_goal_hippo, switch_mid_theta, switch_goal_theta),
                    hist_phase, hist_state, titles)
    return


def select_info_for_manifold(args):
    hist_state, hippos, thetas, theta_slows, hipp_infos, trajs, phases = loading_record_info(args)
    # st=0,1 at=2, rt=3, done=4, phase=5, step_count=6 
    hist_phase = hist_state[:,:,5]
    print('hist_phase:',hist_phase.shape)

    save_path = './figures/'+args.prefix+'/record_info_'+str(args.pseudo_reward_idx)+'_seed'+str(args.initkey)
    record_info = pickle.load(open(save_path, 'rb'))
    print('load record info from:', save_path)
    ## phase: n lists; every list: length * (replay_ei, phase)
    reward_n = record_info['reward_n']

    items = ['mid', 'goal']
    mid_idx = []
    goal_idx = []
    count_mid = 0
    count_goal = 0
    mid_hippo = hippos[0]
    goal_hippo = hippos[1]
    for n in range(args.n_agents):
        mid_idx.append(jnp.arange(count_mid,count_mid+len(mid_hippo[n])))
        count_mid += len(mid_hippo[n])
        goal_idx.append(jnp.arange(count_goal,count_goal+len(goal_hippo[n])))
        count_goal += len(goal_hippo[n])
    # print(mid_idx)
    # print(jnp.concatenate(mid_hippo).shape)
    # print(goal_idx)
    # print(jnp.concatenate(goal_hippo).shape)
    switch_mid_infos = cal_plot.select_switch_mid_infos(args.n_agents, hippos[0], thetas[0], \
                                                                    theta_slows[0], hipp_infos[0], phases[0], mid_idx)
    switch_goal_infos = cal_plot.select_switch_start_goal_infos(args.n_agents, hippos[1], thetas[1], \
                                                                    theta_slows[1], hipp_infos[1], phases[1], goal_idx)
    
    switch_mid_hippo, switch_mid_theta, switch_mid_theta_slow, switch_mid_hipp_info, switch_mid_idx = switch_mid_infos
    switch_goal_hippo, switch_goal_theta, switch_goal_theta_slow, switch_goal_hipp_info, switch_goal_idx = switch_goal_infos
    # switch_mid_theta = jnp.concatenate((switch_mid_theta, switch_mid_theta_slow), axis=-1)
    # switch_goal_theta = jnp.concatenate((switch_goal_theta, switch_goal_theta_slow), axis=-1)
    hist_hippo = record_info['hist_hippo']
    hist_theta = record_info['hist_theta']
    mid_ei = record_info['mid']['ei']
    goal_ei = record_info['goal']['ei']
    reward_n = record_info['reward_n']
    # mid_ei_array = jnp.stack([jnp.array([(j,n) for j in mid_ei[n]]) for n in range(args.n_agents)])
    # goal_ei_array = jnp.stack([jnp.array([(j,n) for j in goal_ei[n]]) for n in range(args.n_agents)])
    
    # all_hippo, theta
    switch_mid_hippo = [x[:,-args.replay_steps:] for x in switch_mid_hippo]
    switch_goal_hippo = [x[:,-args.replay_steps:] for x in switch_goal_hippo]
    switch_mid_theta = [x[:,-args.replay_steps-1:-1] for x in switch_mid_theta]
    switch_goal_theta = [x[:,-args.replay_steps-1:-1] for x in switch_goal_theta]

    return switch_mid_hippo, switch_goal_hippo, switch_mid_theta, switch_goal_theta, hist_hippo, hist_theta, reward_n, \
        hist_phase, hist_state

@DeprecationWarning
def old_eval_value_map(args):
    args.total_eval_steps = 200
    #明天跑一下这个total_eval_steps，看看value advantage能不能复现
    key = jax.random.PRNGKey(2013)
    key, subkey = jax.random.split(key)
    env_state, buffer_state, running_encoder_state, running_hippo_state, \
        running_hippo_std_state, running_policy_state = train.init_states(args, subkey)
    key, subkey = jax.random.split(key)
    obs, env_state = env.pseudo_reset(args.width, args.height, args.n_agents, subkey, args.pseudo_reward)
    # env_state['checked_times'] = jnp.zeros_like(env_state['checked'])
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (args.n_agents, 1), 0, args.n_action)
    hippo_hidden = jnp.zeros((args.n_agents, args.hidden_size))
    theta = jnp.zeros((args.n_agents, args.theta_hidden_size))

    hist_pos = [[] for n in range(args.n_agents)]
    hist_replay_place = [[] for _ in range(args.n_agents)]
    hist_actions = [[] for _ in range(args.n_agents)]
    hist_rewards = [[] for _ in range(args.n_agents)]
    phase_history = [[] for n in range(args.n_agents)]
    theta_v_history = [[] for _ in range(args.n_agents)]

    value_map_all = [[[] for _ in range(2)] for j in range(2)]
    theta_cor_all = [[[] for _ in range(2)] for j in range(2)]
    if 'figures' not in os.listdir('./'):
        os.mkdir('./figures')
    if args.prefix not in os.listdir('./figures'):
        os.mkdir('./figures/'+args.prefix)

    for ei in range(args.total_eval_steps):
        # walk in the env and update buffer (model_step)
        if ei%30==0:
            print('epoch', ei)
        key, subkey = jax.random.split(key)

        env_state, buffer_state, actions, hippo_hidden, theta, rewards, done, replayed_hippo_theta_output, first_hippo_theta_output, hipp_info, value \
            = train.eval_step(env_state, buffer_state, running_encoder_state, running_hippo_state, 
                         running_hippo_std_state, running_policy_state,
                         subkey, actions, hippo_hidden, theta,
                         args.n_agents, args.bottleneck_size, args.hidden_size, args.replay_steps, args.height, args.width,
                         args.visual_prob, args.eval_temperature, args.reset_prob, args.noise_scale, args.pseudo_reward,
                         args.block_idx)
        replayed_hippo_history, replayed_theta_history, output_history, hipp_info_history = replayed_hippo_theta_output 
        
        for n in range(args.n_agents):
            if jnp.isclose(rewards[n],0.5):
                phase_history[n].append(env_state['phase'][n])
            if jnp.isclose(rewards[n], 1) and (not done[n]):
                value_map, theta_map = cal_plot.scan_value_map(args, n, env_state, buffer_state, running_encoder_state, running_hippo_state, 
                         running_hippo_std_state, running_policy_state,
                         subkey, hippo_hidden, theta)
                theta_v_history[n].append(theta_map.reshape(args.width*args.height, args.theta_hidden_size))
                # print(n, value_map)
                if len(phase_history[n])>2:
                    value_map_all[phase_history[n][-3].item()][phase_history[n][-2].item()].append(value_map)
                    # hist_value_map[phase_history[n][-2].item()].append(value_map)
                    # sns.heatmap(value_map, cmap='Oranges')
                    theta_cor_all[phase_history[n][-3].item()][phase_history[n][-2].item()].append(theta_v_history[n][-2]@(theta_v_history[n][-1].transpose(1,0)))
                    print(env_state['current_pos'][n], n, rewards[n], done[n], env_state['phase'][n], phase_history[n][-2].item())
    # print(len(value_map_all), len(value_map_all[0]), len(value_map_all[0][0]))
    value_map_all_mean = [[jnp.nanmean(jnp.array(value_map_all[j][i]),axis=0) for i in range(len(value_map_all[j]))] for j in range(len(value_map_all))]
    theta_cor_all_mean = [[jnp.nanmean(jnp.array(theta_cor_all[j][i]),axis=0) for i in range(len(theta_cor_all[j]))] for j in range(len(theta_cor_all))]
    print(len(value_map_all_mean), len(value_map_all_mean[0]), value_map_all_mean[0][1])
    x, y = jnp.meshgrid(jnp.arange(args.width), jnp.arange(args.height))
    xy = jnp.stack([x, y], axis=-1).reshape(args.width*args.height, 2)
    ck0_filter = path_int.generate_place_cell(xy,config.sigma,env.pseudo_reward_list[args.pseudo_reward_idx][0].reshape(1,2)).reshape(args.height,args.width)
    ck1_filter = path_int.generate_place_cell(xy,config.sigma,env.pseudo_reward_list[args.pseudo_reward_idx][1].reshape(1,2)).reshape(args.height,args.width)
    ck0_idx = [[jnp.sum(value_map_all_mean[j][i]*ck0_filter) for i in range(len(value_map_all_mean[j]))] for j in range(len(value_map_all_mean))]
    ck1_idx = [[jnp.sum(value_map_all_mean[j][i]*ck1_filter) for i in range(len(value_map_all_mean[j]))] for j in range(len(value_map_all_mean))]
    # print(ck0_idx)

    fig, axes = plt.subplots(2,2)
    axes[0][0].imshow(value_map_all_mean[0][0][::-1],cmap='Oranges')
    axes[0][0].set_title('ck0: '+str(ck0_idx[0][0])+' ck1: '+str(ck1_idx[0][0]))
    axes[0][1].imshow(value_map_all_mean[0][1][::-1],cmap='Oranges')
    axes[0][1].set_title('ck0: '+str(ck0_idx[0][1])+' ck1: '+str(ck1_idx[0][1]))
    axes[1][0].imshow(value_map_all_mean[1][0][::-1],cmap='Oranges')
    axes[1][0].set_title('ck0: '+str(ck0_idx[1][0])+' ck1: '+str(ck1_idx[1][0]))
    axes[1][1].imshow(value_map_all_mean[1][1][::-1],cmap='Oranges')
    axes[1][1].set_title('ck0: '+str(ck0_idx[1][1])+' ck1: '+str(ck1_idx[1][1]))
    # print(value_map_all_mean[0])
    # print(value_map_all_mean[1])
    # plt.colorbar()
    # plt.show()
    fig.suptitle('value map')
    plt.savefig('./figures/'+args.prefix+'/old_value_map.png')
    plt.close()
    value_advantage = (jnp.array(ck0_idx) - jnp.array(ck1_idx))/(jnp.array(ck0_idx) + jnp.array(ck1_idx))
    plt.bar(jnp.arange(4),value_advantage.flatten(),tick_label=['0-0','0-1','1-0','1-1'])
    # plt.show()
    plt.title('value advantage')
    plt.savefig('./figures/'+args.prefix+'/old_value_advantage.png')
    plt.close()
    fig, axes = plt.subplots(2,2)
    axes[0][0].imshow(theta_cor_all_mean[0][0][::-1],cmap='viridis')
    axes[0][0].set_title('0-0')
    axes[0][1].imshow(theta_cor_all_mean[0][1][::-1],cmap='viridis')
    axes[0][1].set_title('0-1')
    axes[1][0].imshow(theta_cor_all_mean[1][0][::-1],cmap='viridis')
    axes[1][0].set_title('1-0')
    axes[1][1].imshow(theta_cor_all_mean[1][1][::-1],cmap='viridis')
    axes[1][1].set_title('1-1')
    print(theta_cor_all_mean[0])
    print(theta_cor_all_mean[1])
    fig.suptitle('theta correlation')
    plt.savefig('./figures/'+args.prefix+'/old_theta_cor.png')
    plt.show()
    return

def eval_value_map(args):
    save_path = './figures/'+args.prefix+'/record_info_'+str(args.pseudo_reward_idx)
    record_info = pickle.load(open(save_path, 'rb'))
    print('load record_info from', save_path)
    if 'args' in record_info.keys():
        print('load args from record_info')
        args = record_info['args']
    hist_state = record_info['hist_state']
    # print(hist_state.shape)
    print('st_x, st_y, at, rt, done, phase, step_count')
    # print(hist_state[:50,0,:])
    print('hist_state:',hist_state.shape)
    # st=0,1 at=2, rt=3, done=4, phase=5, step_count=6 
    # hist_step_count = record_info['hist_state'][:,:,6]
    phases = [record_info['start']['phase'], record_info['mid']['phase'], record_info['goal']['phase']]
    # eis = [record_info['start']['ei'], record_info['mid']['ei'], record_info['goal']['ei']]
    # trajs = [record_info['start']['traj'], record_info['mid']['traj'], record_info['goal']['traj']]
    value_maps = [record_info['start']['value_map'], record_info['mid']['value_map'], record_info['goal']['value_map']]
    items = ['start', 'mid', 'goal']
    value_map_switch = [[[] for j in range(8)] for i in range(len(items))]
    # 0: mid, 1: goal, 2:start
    # Select appropriate vmap
    print('selecting replay_switch')
    for type_index in range(len(items)):
        phase = phases[type_index]
        # ei = eis[type_index]
        value_map_of_interest = value_map_switch[type_index]
        value_map_to_be_selected = value_maps[type_index]
        for n in range(args.n_agents):
            print(phase[n])
            if len(phase[n])<=2:
                continue
            # print(n,phase[n].reshape(-1))
            for i in range(len(phase[n])-1):
                if phase[n][i]==0 and phase[n][i+1]==1:
                    value_map_of_interest[0].append(value_map_to_be_selected[n][i+1])

                    if i+2<len(phase[n]) and phase[n][i+2]==1:
                        value_map_of_interest[1].append(value_map_to_be_selected[n][i+2])

                        if i+3<len(phase[n]) and phase[n][i+3]==1:
                            value_map_of_interest[2].append(value_map_to_be_selected[n][i+3])

                            if i+4<len(phase[n]) and phase[n][i+4]==1:
                                value_map_of_interest[3].append(value_map_to_be_selected[n][i+4])

                if phase[n][i]==1 and phase[n][i+1]==0:
                    value_map_of_interest[4].append(value_map_to_be_selected[n][i+1])

                    if i+2<len(phase[n]) and phase[n][i+2]==0:
                        value_map_of_interest[5].append(value_map_to_be_selected[n][i+2])

                        if i+3<len(phase[n]) and phase[n][i+3]==0:
                            value_map_of_interest[6].append(value_map_to_be_selected[n][i+3])

                            if i+4<len(phase[n]) and phase[n][i+4]==0:
                                value_map_of_interest[7].append(value_map_to_be_selected[n][i+4])

    for replay_loc in range(len(items)):
        print(items[replay_loc])
        for i in range(len(value_map_switch[replay_loc])):
            print(len(value_map_switch[replay_loc][i]), end=' ')
        print()
    value_map_switch = [list(map(lambda x: jnp.stack(x,0), value_map_switch[type_index])) for type_index in range(len(items))]
    for replay_loc in range(len(items)):
        print(items[replay_loc])
        for i in range(len(value_map_switch[replay_loc])):
            print(value_map_switch[replay_loc][i].shape, end=' ')
        print()
    ##在这里改改然后就可以看一下consolidation和planning的流形区别
    cal_plot.cal_plot_value_map(args, value_map_switch)
    # cal_plot_manifold_change()
    return 


def Igata_eval_value_map(args):
    value_map_switch = Igata_select_value_map_of_interest(args)
    ##在这里改改然后就可以看一下consolidation和planning的流形区别
    value_map_all_mean, value_advantage, delta_value_advantage, \
            outer_diff_mean, outer_diff_std = cal_plot.cal_plot_value_map(args, value_map_switch)
    # cal_plot.cal_plot_policy_map(args, policy_map_switch, policy_logit_map_switch)
    # cal_plot_manifold_change()
    return value_map_switch

def Igata_select_value_map_of_interest(args):
    save_path = './figures/'+args.prefix+'/record_info_'+str(args.pseudo_reward_idx)+'_seed'+str(args.initkey)
    record_info = pickle.load(open(save_path, 'rb'))
    print('load record_info from', save_path)
    if 'args' in record_info.keys():
        print('load args from record_info')
        args = record_info['args']
    hist_state = record_info['hist_state']
    # print(hist_state.shape)
    print('st_x, st_y, at, rt, done, phase, step_count')
    # print(hist_state[:50,0,:])
    print('hist_state:',hist_state[:50,0])
    if 'figures' not in os.listdir('./'):
        os.mkdir('./figures')
    if args.prefix not in os.listdir('./figures'):
        os.mkdir('./figures/'+args.prefix)
    # st=0,1 at=2, rt=3, done=4, phase=5, step_count=6 
    # hist_step_count = record_info['hist_state'][:,:,6]
    # eis = [record_info['start']['ei'], record_info['mid']['ei'], record_info['goal']['ei']]
    # trajs = [record_info['start']['traj'], record_info['mid']['traj'], record_info['goal']['traj']]
    # phases = [record_info['start']['phase'], record_info['mid']['phase'], record_info['goal']['phase']]
    # value_maps = [record_info['start']['value_map'], record_info['mid']['value_map'], record_info['goal']['value_map']]
    phases = [record_info['mid']['phase']]
    value_maps = [record_info['mid']['value_map']]
    policy_maps = [record_info['mid']['policy_map']]
    policy_logit_maps = [record_info['mid']['policy_logit_map']]
    items = ['mid']
    value_map_switch = [[[] for j in range(5)] for i in range(len(items))]
    policy_map_switch = [[[] for j in range(5)] for i in range(len(items))]
    policy_logit_map_switch = [[[] for j in range(5)] for i in range(len(items))]
    # 0: mid, 1: goal, 2:start
    # Select appropriate vmap
    print('selecting replay_switch')
    for type_index in range(len(items)):
        phase = phases[type_index]
        # ei = eis[type_index]
        value_map_of_interest = value_map_switch[type_index]
        value_map_to_be_selected = value_maps[type_index]
        policy_map_of_interest = policy_map_switch[type_index]
        policy_map_to_be_selected = policy_maps[type_index]
        policy_logit_map_of_interest = policy_logit_map_switch[type_index]
        policy_logit_map_to_be_selected = policy_logit_maps[type_index]

        for n in range(args.n_agents):
            print(phase[n].reshape(-1))
            if len(phase[n])<=2:
                continue
            # print(n,phase[n].reshape(-1))
            for i in range(len(phase[n])-1):
                if (phase[n][i]==0) and (i+4<len(phase[n])) and (phase[n][i+1]==phase[n][i+2]==phase[n][i+3]==phase[n][i+4]==1):
                    for j in range(5):
                        value_map_of_interest[j].append(value_map_to_be_selected[n][i+j])
                        policy_map_of_interest[j].append(policy_map_to_be_selected[n][i+j])
                        policy_logit_map_of_interest[j].append(policy_logit_map_to_be_selected[n][i+j])
                    # value_map_of_interest[0].append(value_map_to_be_selected[n][i])
                    # policy_map_of_interest[0].append(policy_map_to_be_selected[n][i])
                    # policy_logit_map_of_interest[0].append(policy_logit_map_to_be_selected[n][i])
                
                    # value_map_of_interest[1].append(value_map_to_be_selected[n][i+1])
                    # policy_map_of_interest[1].append(policy_map_to_be_selected[n][i+1])
                    # policy_logit_map_of_interest[1].append(policy_logit_map_to_be_selected[n][i+1])

                    # value_map_of_interest[2].append(value_map_to_be_selected[n][i+2])
                    # policy_map_of_interest[2].append(policy_map_to_be_selected[n][i+2])
                    # policy_logit_map_of_interest[2].append(policy_logit_map_to_be_selected[n][i+2])

                    # value_map_of_interest[3].append(value_map_to_be_selected[n][i+3])
                    # policy_map_of_interest[3].append(policy_map_to_be_selected[n][i+3])
                    # policy_logit_map_of_interest[3].append(policy_logit_map_to_be_selected[n][i+3])

                    # value_map_of_interest[4].append(value_map_to_be_selected[n][i+4])
                    # policy_map_of_interest[4].append(policy_map_to_be_selected[n][i+4])
                    # policy_logit_map_of_interest[4].append(policy_logit_map_to_be_selected[n][i+4])

                                # if i+5<len(phase[n]) and phase[n][i+5]==1:
                                #     value_map_of_interest[5].append(value_map_to_be_selected[n][i+5])
                                #     policy_map_of_interest[5].append(policy_map_to_be_selected[n][i+5])
                                #     policy_logit_map_of_interest[5].append(policy_logit_map_to_be_selected[n][i+5])
                                    
                                    # if i+6<len(phase[n]) and phase[n][i+6]==1:
                                    #     value_map_of_interest[6].append(value_map_to_be_selected[n][i+6])
                                    #     policy_map_of_interest[6].append(policy_map_to_be_selected[n][i+6])

                    # if i+2<len(phase[n]) and phase[n][i+2]==0:
                    #     value_map_of_interest[5].append(value_map_to_be_selected[n][i+2])

                    #     if i+3<len(phase[n]) and phase[n][i+3]==0:
                    #         value_map_of_interest[6].append(value_map_to_be_selected[n][i+3])

                    #         if i+4<len(phase[n]) and phase[n][i+4]==0:
                    #             value_map_of_interest[7].append(value_map_to_be_selected[n][i+4])

    value_map_switch = [list(map(lambda x: jnp.stack(x,0), value_map_switch[type_index])) for type_index in range(len(items))]
    policy_map_switch = [list(map(lambda x: jnp.stack(x,0), policy_map_switch[type_index])) for type_index in range(len(items))]
    policy_logit_map_switch = [list(map(lambda x: jnp.stack(x,0), policy_logit_map_switch[type_index])) for type_index in range(len(items))]
    for replay_loc in range(len(items)):
        print(items[replay_loc])
        for i in range(len(value_map_switch[replay_loc])):
            print(value_map_switch[replay_loc][i].shape, end=' ')
        print()
    for replay_loc in range(len(items)):
        print(items[replay_loc])
        for i in range(len(policy_map_switch[replay_loc])):
            print(policy_map_switch[replay_loc][i].shape, end=' ')
        print()
    return value_map_switch


def eval_video(args):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    env_state, buffer_state, running_encoder_state, running_hippo_state, \
    running_hippo_std_state, running_policy_state = train.init_states(args, subkey)
    key, subkey = jax.random.split(key)
    obs, env_state = env.Igata_reset(args.width, args.height, args.n_agents, subkey, args.pseudo_reward)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (args.n_agents, 1), 0, args.n_action)
    hippo_hidden = jnp.zeros((args.n_agents, args.hidden_size))
    theta = jnp.zeros((args.n_agents, args.theta_hidden_size))

    hist_pos = [[] for n in range(args.n_agents)]
    hist_replay_place = [[] for _ in range(args.n_agents)]
    hist_actions = [[] for _ in range(args.n_agents)]
    hist_rewards = [[] for _ in range(args.n_agents)]
    reward_center_history = [[env_state['reward_center'][n]] for n in range(args.n_agents)]
    replay_cut = [[] for _ in range(args.n_agents)]
    all_replay_cnt = 0
    segment_list = ['S-C1','S-C2','C1-G','C2-G']
    if 'video' not in os.listdir('./'):
        os.mkdir('./video')
    if args.prefix not in os.listdir('./video'):
        os.mkdir('./video/'+args.prefix)
    if args.prefix+'_all_replay' not in os.listdir('./video'):
        os.mkdir('./video/'+args.prefix+'_all_replay')
    for ei in range(args.total_eval_steps):
        # walk in the env and update buffer (model_step)
        if ei%30==0:
            print('epoch', ei)
        key, subkey = jax.random.split(key)
        
        env_state, buffer_state, actions, hippo_hidden, theta, rewards, done, replayed_hippo_theta_output, first_hippo_theta_output, hipp_info, value, policy \
            = train.Igata_step(env_state, buffer_state, running_encoder_state, running_hippo_state, 
                         running_hippo_std_state, running_policy_state,
                         subkey, actions, hippo_hidden, theta,
                         args.n_agents, args.bottleneck_size, args.hidden_size, args.replay_steps, args.height, args.width,
                         args.visual_prob, args.eval_temperature, args.reset_prob, args.noise_scale, args.pseudo_reward,
                         args.block_idx)
        # print(env_state['current_pos'], actions, rewards)
        replayed_hippo_history, replayed_theta_history, output_history, hipp_info_history = replayed_hippo_theta_output
        # first_hippo_output: (n_agents, 65)
        # replay_hippo_theta_output: (replay_steps, n_agents, hidden_size), (replay_steps, n_agents, hidden_size)
        # replay_step * n_agents * hidden_size
        num_cells = args.width*args.height
        # replay_step * n_agents * hw
        max_decoding_place = jnp.argmax(output_history[...,:num_cells],axis=-1)
        # replay_step * n_agents
        pred_r = output_history[...,num_cells]
        max_decoding_traj_present = [[] for _ in range(args.n_agents)]
        for n in range(args.n_agents):
            hist_actions[n].append(actions[n])
            hist_pos[n].append(env_state['current_pos'][n])
            hist_rewards[n].append(rewards[n])
            if not jnp.all(env_state['reward_center'][n] == reward_center_history[n][-1], axis=-1):
                reward_center_history[n].append(env_state['reward_center'][n])
            if rewards[n]>0:
                # including mid_reward and goal
                hist_replay_place[n].append(max_decoding_place[:,n]) # replay_step * 1
                max_decoding_traj_present[n] = [max_decoding_place[:,n]]
        
        plt.close()
        plt.figure(figsize=(20,12))
        for n in range(args.n_agents):
            plt.subplot(2,4,n+1)
            state_traj = jnp.concatenate((jnp.stack(hist_pos[n],axis=0),jnp.stack(hist_actions[n],axis=0),jnp.stack(hist_rewards[n])),axis=1)
            whole_traj = {'agent_th':n, 'state_traj':state_traj, 'replay_traj':max_decoding_traj_present[n], \
                'reward_pos_traj':jnp.array(reward_center_history[n]), 'pred_r':pred_r[:,n], 'first_output':first_hippo_theta_output[2][n]}
            cal_plot.plot_trajectory(whole_traj, args)
            if rewards[n] > 0:
                replay_cut[n].append(whole_traj)

            if done[n]:
                hist_pos[n] = []
                hist_rewards[n] = []
                if hist_replay_place[n]:
                    hist_replay_place[n] = [hist_replay_place[n][-1]]
                else:
                    hist_replay_place[n] = []
                hist_actions[n] = []
        
        plt.savefig('./video/'+args.prefix+'/video_'+str(ei)+'.png')

        
        j = 0
        for _ in range(len(replay_cut)):
            if not replay_cut[j]:
                break
            j += 1
        if j == len(replay_cut):
            plt.close()
            plt.figure(figsize=(20,12))
            for n in range(args.n_agents):
                plt.subplot(2,4,n+1)
                whole_traj = replay_cut[n].pop(0)
                cal_plot.plot_trajectory(whole_traj, args)

                calculate_replay_to_scan = partial(cal_plot.calculate_replay, width=args.width, height=args.height, ck0_x=env.ck0_x, ck0_y=env.ck0_y, ck1_x=env.ck1_x, ck1_y=env.ck1_y,
                    ck0_x_g_lb=env.ck0_x_g_lb, ck0_y_g_lb=env.ck0_y_g_lb, ck1_x_g_lb=env.ck1_x_g_lb, ck1_y_g_lb=env.ck1_y_g_lb)
                directionality, forward_degree, score, max_segment = \
                    calculate_replay_to_scan(whole_traj['replay_traj'][0])
                print(directionality, forward_degree, score, max_segment)
                dominant_score = jax.lax.top_k(score, 2)[0]
                print(dominant_score)
                significant = (jnp.max(score, axis=-1) > 0.4) & (directionality > 0.4) \
                            & (jnp.abs(forward_degree) > 0.4) & (dominant_score[0] > dominant_score[1])
                if significant:
                    color = 'green'
                else:
                    color = 'red'
                plt.title('dir:{:.2f}'.format(directionality.item())+\
                          ' f_d:{:.2f}'.format(forward_degree.item())+'\n'+\
                            's_max:{:.2f}'.format(jnp.max(score).item())+\
                            ' seg:'+segment_list[max_segment], fontdict={'fontsize': 25},
                            c=color)
            plt.subplots_adjust(wspace=0.5, hspace=0.5, left=0.1, right=0.9)
            plt.savefig('./video/'+args.prefix+'_all_replay/video_all_replay'+str(all_replay_cnt)+'.png')
            all_replay_cnt += 1
            plt.close()

    def compose_gif(prefix):
        gif_images = []
        imgs_path = sorted(os.listdir('./video/'+prefix),key=lambda x:int(x.split('.')[0].split('_')[-1]))
        for path in imgs_path:
            gif_images.append(imageio.imread('./video/'+prefix+'/'+path))
        imageio.mimsave('./video/'+prefix+"_video.gif",gif_images,duration=500)
        print('save gif in ./video/'+prefix+"_video.gif")

    compose_gif(args.prefix)
    print(env_state['total_checked'])


def eval_info(args):
    """Generate all of the trajectory information and replay information and record them.
    
    Here are some conventions that should be obeyed in all of the functions in record.py
    prefix:
        'hist_xxx' refers to all information during movement without replay.
        'start_xxx' refers to information during start-replay.
        'mid_xxx' refers to information during mid-replay (replay at the mid-reward location).
        'goal_xxx' refers to information during goal-replay.
    suffix:
        '_state' refers to every state information: st(2), at(1), rt(1), done(1), phase(1), step_count(1) with shape[-1] = 7.\n
        '_phase' refers to reward_idx as in env.pyn\n
        '_ei' (always with the prefix of start/mid/goal) refers to the step_ei when the agent was executing this replay.\n
        '_s' (`trajs` vs `traj`, `phases` vs `phase` ) refers to a list of three lists of that info in `(start-mid-goal)` period.
    
    The format for all replay information (for example, the list mid_hippo):
        mid_hippo consists of arrays of the number `args.n_agents`, and every array corresponds to all hippo replay 
            the agent has executed. The order is the same as the true order that the agent meets every reward. 
            So the shape for every array is [reward_times, args.replay_steps, args.hidden_size]
            
        However, for decoding convenience I record extra init and output information so the real shape is [reward_times, args.replay_steps+2, args.hidden_size]
            for example each step in one array of the mid_hippo: last hippo_hidden when the sensory signal is not received, 
                the immediate hippo_hidden when the signal is received, and four steps when the replay is being executed.
                And the hipp_info is the same setting.
            for example each step in one array of the mid_theta: last theta when the sensory signal is not received, 
                four steps when the replay is being executed, and the ultimate theta when the action is chosen.
            And the assumed way to understand this setting is that, for every index i, the information is transferred from 
                hippo to hipp_info then to theta, which can be better understood in the decoding_acc figure or something like that.
    

        If you concatenate them you can get the whole replay set.

    Args: args

    Returns: No return

    """
    key = jax.random.PRNGKey(args.initkey)
    key, subkey = jax.random.split(key) 
    # if os.path.exists(args.model_path+'/'+args.prefix+'_args'):
    #     args = pickle.load(open(args.model_path+'/'+args.prefix+'_args', 'rb'))
    #     print('load args from '+args.model_path+'/'+args.prefix+'_args')
    #     print(args)
    env_state, buffer_state, running_encoder_state, running_hippo_state, \
        running_hippo_std_state, running_policy_state = train.init_states(args, subkey)
    key, subkey = jax.random.split(key)
    _, env_state = env.Igata_reset(args.width, args.height, args.n_agents, subkey, args.pseudo_reward)
    actions = jax.random.randint(subkey, (args.n_agents, 1), 0, args.n_action)
    hippo_hidden = jnp.zeros((args.n_agents, args.hidden_size))
    theta = jnp.zeros((args.n_agents, args.theta_hidden_size))

    hist_hippo = jnp.zeros((args.total_eval_steps, *hippo_hidden.shape))
    hist_theta = jnp.zeros((args.total_eval_steps, *theta.shape))
    hist_state = jnp.zeros((args.total_eval_steps, args.n_agents, 7))
    hist_hipp_info = jnp.zeros((args.total_eval_steps, args.n_agents, args.bottleneck_size))

    mid_hippo = [[] for _ in range(args.n_agents)]
    mid_theta = [[] for _ in range(args.n_agents)]
    mid_traj = [[] for _ in range(args.n_agents)]
    mid_hipp_info = [[] for _ in range(args.n_agents)]
    mid_ei = [[] for _ in range(args.n_agents)]

    goal_hippo = [[] for _ in range(args.n_agents)]
    goal_theta = [[] for _ in range(args.n_agents)]
    goal_traj = [[] for _ in range(args.n_agents)]
    goal_hipp_info = [[] for _ in range(args.n_agents)]
    goal_ei = [[] for _ in range(args.n_agents)]

    start_hippo = [[] for _ in range(args.n_agents)]
    start_theta = [[] for _ in range(args.n_agents)]
    start_traj = [[] for _ in range(args.n_agents)]
    start_hipp_info = [[] for _ in range(args.n_agents)]
    start_ei = [[] for _ in range(args.n_agents)]

    mid_phase = [[] for _ in range(args.n_agents)]
    goal_phase = [[] for _ in range(args.n_agents)]
    start_phase = [[] for _ in range(args.n_agents)]

    mid_value_map = [[] for _ in range(args.n_agents)]
    goal_value_map = [[] for _ in range(args.n_agents)]
    mid_policy_map = [[] for _ in range(args.n_agents)]
    goal_policy_map = [[] for _ in range(args.n_agents)]
    start_value_map = [[] for _ in range(args.n_agents)]
    start_policy_map = [[] for _ in range(args.n_agents)]
    mid_policy_logit_map = [[] for _ in range(args.n_agents)]
    reward_n = [[] for _ in range(args.n_agents)]
    last_hipp_info = jnp.zeros((args.n_agents, args.bottleneck_size))
    for ei in range(args.total_eval_steps):
        print(ei+1)
        key, subkey = jax.random.split(subkey)
        st = env_state['current_pos']
        at = actions
        phase = env_state['phase']
        step_count = env_state['step_count']

        block_idx = jnp.where(args.block_option == 'cons', 0, jnp.where(args.block_option == 'plan', 1, -1))
        # print(block_idx)
        env_state, buffer_state, actions, hippo_hidden, theta, rewards, done, replayed_history, first_hippo_theta_output, hipp_info, value, policy \
            = train.Igata_step(env_state, buffer_state, running_encoder_state, running_hippo_state, 
                         running_hippo_std_state, running_policy_state,
                         subkey, actions, hippo_hidden, theta,
                         args.n_agents, args.bottleneck_size, args.hidden_size, args.replay_steps, args.height, args.width,
                         args.visual_prob, args.eval_temperature, args.reset_prob, args.noise_scale, args.pseudo_reward,
                         block_idx)
        rt = rewards
        replayed_hippo_history, replayed_theta_history, output_history, hipp_info_history = replayed_history
        place_cell_history = output_history[:, :, :args.height*args.width]
        first_hippo, first_theta, first_output = first_hippo_theta_output

        hist_hippo = hist_hippo.at[ei].set(hippo_hidden)
        hist_theta = hist_theta.at[ei].set(theta)
        hist_state = hist_state.at[ei].set(jnp.concatenate([st, at, rt, done, phase, step_count], axis=-1))
        hist_hipp_info = hist_hipp_info.at[ei].set(hipp_info)
        # print(hist_state[ei,2,:])
        # total_eval_steps * n_agents * hidden
        # jax.debug.print('ei:{a},n:{b},replayed_hippo:{c},replayed_theta:{c0},\
        #                     origin_theta:{c1},r:{d},phase:{e}',
        #                     a=ei,b=2,c=replayed_theta_hippo[1][2],c0=replayed_theta_hippo[0][2],
        #                     c1=theta[2],d=rewards[2],e=phase[2])
        for n in range(args.n_agents):
            if jnp.isclose(rewards[n], 0.5):
                if n==0:
                    print(ei,len(mid_hippo),phase[n])
                # print(ei, n, phase[n])
                mid_hippo[n].append(jnp.concatenate((hist_hippo.at[ei-3:ei,n].get().reshape(-1,args.hidden_size), 
                                                        first_hippo[n].reshape(1,-1),
                                                        replayed_hippo_history[:,n]),
                                                        0))
                # print(mid_hippo[n][-1].shape)
                mid_theta[n].append(jnp.concatenate((hist_theta[ei-3:ei,n].reshape(-1,args.theta_hidden_size),
                                                        replayed_theta_history[:,n],
                                                        hist_theta[ei,n].reshape(1,-1)),
                                                        0))
                mid_hipp_info[n].append(jnp.concatenate((last_hipp_info[n].reshape(1,-1),
                                                        hipp_info_history[:,n], 
                                                        hipp_info[n].reshape(1,-1)), 0))
                mid_traj[n].append(jnp.argmax(place_cell_history[:,n], axis=-1))
                key, subkey = jax.random.split(key)
                # value_map, policy_map, theta_map, policy_logit_map = cal_plot.scan_value_map(args, n, env_state, buffer_state, running_encoder_state, running_hippo_state, 
                #          running_hippo_std_state, running_policy_state,
                #          subkey, hippo_hidden, theta)
                # mid_value_map[n].append(value_map)
                # mid_policy_map[n].append(policy_map)
                # mid_policy_logit_map[n].append(policy_logit_map)
                # print(policy_logit_map.transpose(2,0,1))
                mid_phase[n].append(phase[n].item())
                mid_ei[n].append(ei)
                reward_n[n].append((ei,0))
            if jnp.isclose(rewards[n], 1):
                if done[n]:
                    # goal replay
                    goal_hippo[n].append(jnp.concatenate((hist_hippo[ei-3:ei,n].reshape(-1,args.hidden_size), 
                                                            first_hippo[n].reshape(1,-1),
                                                            replayed_hippo_history[:,n]),
                                                            0))
                    goal_theta[n].append(jnp.concatenate((hist_theta[ei-3:ei,n].reshape(-1,args.theta_hidden_size),
                                                            replayed_theta_history[:,n],
                                                            hist_theta[ei,n].reshape(1,-1)),
                                                            0))
                    goal_hipp_info[n].append(jnp.concatenate((last_hipp_info[n].reshape(1,-1),
                                                            hipp_info_history[:,n], 
                                                            hipp_info[n].reshape(1,-1)), 0))
                    
                    goal_traj[n].append(jnp.argmax(place_cell_history[:,n], axis=-1))
                    key, subkey = jax.random.split(key)
                    # value_map, policy_map, theta_map = cal_plot.scan_value_map(args, n, env_state, buffer_state, running_encoder_state, running_hippo_state, 
                    #      running_hippo_std_state, running_policy_state,
                    #      subkey, hippo_hidden, theta)
                    # goal_value_map[n].append(value_map)
                    # goal_policy_map[n].append(policy_map)
                    goal_phase[n].append(phase[n].item())
                    goal_ei[n].append(ei)
                    reward_n[n].append((ei,1))
                else:
                    # start replay
                    start_hippo[n].append(jnp.concatenate((hist_hippo[ei-1,n].reshape(1,-1),
                                                            first_hippo[n].reshape(1,-1),
                                                            replayed_hippo_history[:,n]),
                                                            0))
                    start_theta[n].append(jnp.concatenate((first_theta[n].reshape(1,-1),
                                                            replayed_theta_history[:,n],
                                                            hist_theta[ei,n].reshape(1,-1)),
                                                            0))
                    start_hipp_info[n].append(jnp.concatenate((last_hipp_info[n].reshape(1,-1),
                                                             hipp_info_history[:,n], 
                                                             hipp_info[n].reshape(1,-1)), 0))
                
                    start_traj[n].append(jnp.argmax(place_cell_history[:,n], axis=-1))
                    key, subkey = jax.random.split(key)
                    # value_map, theta_map = cal_plot.scan_value_map(args, n, env_state, buffer_state, running_encoder_state, running_hippo_state, 
                    #      running_hippo_std_state, running_policy_state,
                    #      subkey, hippo_hidden, theta)
                    # start_value_map[n].append(value_map)
                    start_phase[n].append(phase[n].item())
                    start_ei[n].append(ei)
                    reward_n[n].append((ei,2))

        last_hipp_info = hipp_info
    n_record_step = args.replay_steps+4
    for n in range(args.n_agents):
        print(1)
        print(len(mid_hippo[n]), len(mid_theta[n]), len(mid_traj[n]), len(mid_hipp_info[n]), len(mid_phase[n]), len(mid_ei[n]))
    mid_hippo = [jnp.array(mid_hippo[n]).reshape(len(mid_hippo[n]),n_record_step,args.hidden_size) for n in range(args.n_agents)]
    mid_theta = [jnp.array(mid_theta[n]).reshape(len(mid_hippo[n]),n_record_step,args.theta_hidden_size) for n in range(args.n_agents)]
    mid_hipp_info = [jnp.array(mid_hipp_info[n]).reshape(len(mid_hippo[n]),args.replay_steps+2,args.bottleneck_size) for n in range(args.n_agents)]
    mid_traj = [jnp.array(mid_traj[n]).reshape(len(mid_hippo[n]),args.replay_steps) for n in range(args.n_agents)]
    mid_phase = [jnp.array(mid_phase[n]).reshape(len(mid_hippo[n]),1) for n in range(args.n_agents)]
    mid_ei = [jnp.array(mid_ei[n]).reshape(len(mid_hippo[n]),1) for n in range(args.n_agents)]
    print(len(mid_value_map[0]))
    mid_value_map = [jnp.array(mid_value_map[n]).reshape(-1,args.height,args.width) for n in range(args.n_agents)]
    mid_policy_map = [jnp.array(mid_policy_map[n]).reshape(-1,args.height,args.width,2) for n in range(args.n_agents)]
    mid_policy_logit_map = [jnp.array(mid_policy_logit_map[n]).reshape(-1,args.height,args.width,4) for n in range(args.n_agents)]
    # for i in range(args.n_agents):
    #     print(mid_phase[i].shape)
    goal_hippo = [jnp.array(goal_hippo[n]).reshape(len(goal_hippo[n]),n_record_step,args.hidden_size) for n in range(args.n_agents)]
    goal_theta = [jnp.array(goal_theta[n]).reshape(len(goal_hippo[n]),n_record_step,args.theta_hidden_size) for n in range(args.n_agents)]
    goal_hipp_info = [jnp.array(goal_hipp_info[n]).reshape(len(goal_hippo[n]),args.replay_steps+2,args.bottleneck_size) for n in range(args.n_agents)]
    goal_traj = [jnp.array(goal_traj[n]).reshape(len(goal_hippo[n]),args.replay_steps) for n in range(args.n_agents)]
    goal_phase = [jnp.array(goal_phase[n]).reshape(len(goal_hippo[n]),1) for n in range(args.n_agents)]
    goal_ei = [jnp.array(goal_ei[n]).reshape(len(goal_hippo[n]),1) for n in range(args.n_agents)]
    goal_value_map = [jnp.array(goal_value_map[n]).reshape(-1,args.height,args.width) for n in range(args.n_agents)]
    # goal_policy_map = [jnp.array(goal_policy_map[n]).reshape(-1,args.height,args.width,args.n_action) for n in range(args.n_agents)]
    # start_hippo = [jnp.stack(start_hippo[n],0) for n in range(args.n_agents)]
    # start_theta = [jnp.stack(start_theta[n],0) for n in range(args.n_agents)]
    # start_hipp_info = [jnp.stack(start_hipp_info[n],0) for n in range(args.n_agents)]
    # start_traj = [jnp.stack(start_traj[n],0) for n in range(args.n_agents)]
    # start_phase = [jnp.array(start_phase[n]).reshape(-1,1) for n in range(args.n_agents)]
    # start_ei = [jnp.array(start_ei[n]).reshape(-1,1) for n in range(args.n_agents)]
    # start_value_map = [jnp.array(start_value_map[n]) for n in range(args.n_agents)]

    record_info = {'args': args,
                    'hist_hippo': hist_hippo, 'hist_theta': hist_theta, 'hist_state': hist_state, 
                    'hist_hipp_info': hist_hipp_info, 'reward_n':reward_n,
                    'mid':{'hippo':mid_hippo, 'theta':mid_theta, 'traj':mid_traj, 
                            'hipp_info':mid_hipp_info, 'ei':mid_ei, 'phase':mid_phase,
                            'value_map':mid_value_map, 'policy_map':mid_policy_map, 'policy_logit_map':mid_policy_logit_map},
                    'goal':{'hippo':goal_hippo, 'theta':goal_theta, 'traj':goal_traj,
                            'hipp_info':goal_hipp_info, 'ei':goal_ei, 'phase':goal_phase,
                            'value_map':goal_value_map, 'policy_map':goal_policy_map},
                    'start':{'hippo':start_hippo, 'theta':start_theta, 'traj':start_traj,
                            'hipp_info':start_hipp_info, 'ei':start_ei, 'phase':start_phase,
                            'value_map':start_value_map, 'policy_map':start_policy_map}
                    }
    # mid_hippo is a list of (length, replay_steps+2, hidden_size) array
    # 后期优化的时候可以给每个prefix单独开一个文件夹来存
    if not os.path.exists('./figures/'+args.prefix):
        os.mkdir('./figures'+args.prefix)
    save_path = './figures/'+args.prefix+'/record_info_'+str(args.pseudo_reward_idx)+'_seed'+str(args.initkey)
    pickle.dump(record_info, open(save_path, 'wb'))
    print('save record info to:', save_path)

    return record_info

def loading_record_info(args):
    """
    Load record info from record_info
    """
    save_path = './figures/'+args.prefix+'/record_info_'+str(args.pseudo_reward_idx)+'_seed'+str(args.initkey)
    record_info = pickle.load(open(save_path, 'rb'))
    print('load record info from:', save_path)
    hist_state = record_info['hist_state']
    print('st_x, st_y, at, rt, done, phase, step_count')
    # print(hist_state[:50,0,:])
    print('hist_state:',hist_state.shape)
    # st=0,1 at=2, rt=3, done=4, phase=5, step_count=6 
    hist_phase = record_info['hist_state'][:,:,5]
    print('hist_phase:',hist_phase.shape)
    mid_fast_theta = [x[...,:args.theta_fast_size] for x in record_info['mid']['theta']]
    mid_slow_theta = [x[...,args.theta_fast_size:] for x in record_info['mid']['theta']]
    goal_fast_theta = [x[...,:args.theta_fast_size] for x in record_info['goal']['theta']]
    goal_slow_theta = [x[...,args.theta_fast_size:] for x in record_info['goal']['theta']]
    mid_theta = record_info['mid']['theta']
    goal_theta = record_info['goal']['theta']
    # start_fast_theta = [x[...,:args.theta_fast_size] for x in record_info['start']['theta']]
    # start_slow_theta = [x[...,args.theta_fast_size:] for x in record_info['start']['theta']]
    # hippos = [record_info['start']['hippo'], record_info['mid']['hippo'], record_info['goal']['hippo']]
    hippos = [record_info['mid']['hippo'], record_info['goal']['hippo']]
    # theta_fasts = [start_fast_theta, mid_fast_theta, goal_fast_theta]
    # theta_slows = [start_slow_theta, mid_slow_theta, goal_slow_theta]
    theta_fasts = [mid_fast_theta, goal_fast_theta]
    theta_slows = [mid_slow_theta, goal_slow_theta]
    thetas = [mid_theta, goal_theta]
    # hipp_infos = [record_info['start']['hipp_info'], record_info['mid']['hipp_info'], record_info['goal']['hipp_info']]
    # trajs = [record_info['start']['traj'], record_info['mid']['traj'], record_info['goal']['traj']]
    # phases = [record_info['start']['phase'], record_info['mid']['phase'], record_info['goal']['phase']]
    hipp_infos = [record_info['mid']['hipp_info'], record_info['goal']['hipp_info']]
    trajs = [record_info['mid']['traj'], record_info['goal']['traj']]
    phases = [record_info['mid']['phase'], record_info['goal']['phase']]
    
    # print(jnp.diff(jnp.concatenate(theta_slows[0],0),1).mean(0))
    # print(jnp.concatenate(theta_slows[0],0).std(0).mean(1))
    # print(phases[0][:10])
    return hist_state, hippos, thetas, theta_slows, hipp_infos, trajs, phases

def decoding_phase(args):
    """
    This function is 
        loading data from record_info, \n
        preprocess them (for example, split the fast theta and slow theta, assemble start_mid_goal and pick the right process according to args.info_type) \n
        train the decoder using `train_phase_decoder`, \n
        calculate the decoding accuracy and plot them all using `cal_plot_replay_curve`

    Args: args
    Returns: No return
    """


    hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = loading_record_info(args)
    # st=0,1 at=2, rt=3, done=4, phase=5, step_count=6 
    hist_phase = hist_state[:,:,5]
    print('hist_phase:',hist_phase.shape)

    save_path = './figures/'+args.prefix+'/record_info_'+str(args.pseudo_reward_idx)+'_seed'+str(args.initkey)
    record_info = pickle.load(open(save_path, 'rb'))
    print('load record info from:', save_path)
    ## phase: n lists; every list: length * (replay_ei, phase)
    reward_n = record_info['reward_n']

    items = ['hist', 'mid', 'goal']
    
    # classifier
    decoder_path = './decoder/'+args.prefix+'_phase_decoder_'+str(args.pseudo_reward_idx)
    if os.path.exists(decoder_path):
        clfs = pickle.load(open(decoder_path, 'rb'))
        print('load decoder from:', decoder_path)
    else:
        # clf_hippo = sklearn.gaussian_process.GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
        # clf_theta = sklearn.gaussian_process.GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
        # clf_theta_slow = sklearn.gaussian_process.GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
        # clf_hipp_info = sklearn.gaussian_process.GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
        clfs = [[GaussianNB() for step in range(args.replay_steps+4)] for _ in range(4)]
        # clfs = [MLPClassifier(hidden_layer_sizes=8,max_iter=400,) for _ in range(3)]+[clf_hipp_info]
        # concatenate every list for training, however, notice that there are still four lists in hippo_to_train
        hippos_to_train = [record_info['hist_hippo']] + [jnp.concatenate(x,0) for x in hippos]
        thetas_to_train = [record_info['hist_theta'][...,:args.theta_fast_size]] + [jnp.concatenate(x,0) for x in theta_fasts]
        theta_slows_to_train = [record_info['hist_theta'][...,args.theta_fast_size:]] + [jnp.concatenate(x,0) for x in theta_slows]
        phases_to_train = [hist_phase]+[jnp.concatenate(x,0).repeat(args.replay_steps+4,1) for x in phases]
        hipp_infos_to_train = [record_info['hist_hipp_info']]
        print('hippos, thetas, theta_slows')
        for i in range(3):
            print(items[i], hippos_to_train[i].shape, thetas_to_train[i].shape, theta_slows_to_train[i].shape, phases_to_train[i].shape)
            # if i>=1:
            #     print(hipp_infos_to_train[i-1].shape)
        clfs = cal_plot.train_phase_decoder(args, clfs, hippos_to_train, thetas_to_train, theta_slows_to_train, \
                                    hipp_infos_to_train, phases_to_train)

    if args.info_type == 'hist_phase':
        cal_plot.cal_plot_hist_phase(args, clfs[0], clfs[1], record_info['hist_hippo'], \
                                    record_info['hist_theta'][...,:args.theta_fast_size], hist_state, reward_n)
    elif args.info_type == 'replay_phase':
        replay_phase_quantities = cal_plot.cal_plot_replay_curve(args, clfs, hippos, theta_fasts, theta_slows, hipp_infos, \
                                    phases, trajs, phases)
        # 4(00,01,011,0111)*4(hippo,theta,theta_slow,hipp_info)*8(4 replay_steps+4 init)
    return replay_phase_quantities

def decoding_energy(args):
    """
    This function is 
        loading data from record_info, \n
        preprocess them (for example, split the fast theta and slow theta, assemble start_mid_goal and pick the right process according to args.info_type) \n
        calculate the energy and plot them all using `cal_plot_replay_curve`

    Args: args
    Returns: No return
    """
    hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = loading_record_info(args)
    cal_plot.cal_plot_replay_curve(args, None, hippos, theta_fasts, theta_slows, hipp_infos, \
                            phases, trajs, phases)
    return

def decoding_action_plan(args):
    """
    This function is 
        loading data from record_info, \n
        preprocess them (for example, split the fast theta and slow theta, assemble start_mid_goal, pick the right process according to args.info_type\
            split the action predicting goal based on which step it is) \n
        train the action decoder [Func train_action_decoder], \n
        calculate the decoding accuracy and plot them all [Func cal_plot_replay_curve]

    Args: args

    Returns: No return
    """
    hist_state, hippos, theta_fasts, theta_slows, hipp_infos, trajs, phases = loading_record_info(args)
    save_path = './figures/'+args.prefix+'/record_info_'+str(args.pseudo_reward_idx)+'_seed'+str(args.initkey)
    record_info = pickle.load(open(save_path, 'rb'))
    print('load record info from:', save_path)
    # if 'args' in record_info.keys():
    #     print('load args from record_info')
    #     args = record_info['args']
    # train one decoder to output four actions
    # st=0,1 at=2, rt=3, done=4, phase=5, step_count=6 
    hist_pos = hist_state[:,:,0:2]
    hist_action = hist_state[:,:,2]
    mid_ei = record_info['mid']['ei']
    goal_ei = record_info['goal']['ei']
    # start_ei = record_info['start']['ei']
    mid_action = [[] for n in range(args.n_agents)]
    goal_action = [[] for n in range(args.n_agents)]
    # start_action = [[] for n in range(args.n_agents)]
    steps_to_predict = 4
    # print(mid_ei[0])
    for n in range(args.n_agents):
        for j in range(len(mid_ei[n])):
            action_to_store = hist_pos[mid_ei[n][j].item()+1:mid_ei[n][j].item()+steps_to_predict+1,n] - hist_pos[mid_ei[n][j].item(),n]
            action_to_store = jnp.concatenate([action_to_store, jnp.zeros((steps_to_predict-action_to_store.shape[0],action_to_store.shape[1]))])
            mid_action[n].append(action_to_store)
            # if action_to_store.shape[0]!=steps_to_predict:
            #     print(action_to_store.shape)
            #     print(n, j, mid_ei[n][j])
        for j in range(len(goal_ei[n])):
            action_to_store = hist_pos[goal_ei[n][j].item()+1:goal_ei[n][j].item()+steps_to_predict+1,n] - hist_pos[goal_ei[n][j].item(),n]
            action_to_store = jnp.concatenate([action_to_store, jnp.zeros((steps_to_predict-action_to_store.shape[0],action_to_store.shape[1]))])
            goal_action[n].append(action_to_store)
            # if action_to_store.shape[0]!=steps_to_predict:
            #     print(action_to_store.shape)
            #     print(n, j, goal_ei[n][j])
            # print(action_to_store.shape)
        # print(len(mid_action[n]), len(goal_action[n]))
        # for j in range(len(start_ei[n])):
        #     action_to_store = hist_action[start_ei[n][j].item():start_ei[n][j].item()+steps_to_predict,n]
        #     action_to_store = jnp.concatenate([action_to_store, jnp.zeros(steps_to_predict-action_to_store.shape[0],)])
        #     start_action[n].append(action_to_store)

    mid_action = [jnp.array(x).reshape(-1,steps_to_predict,2) for x in mid_action]
    goal_action = [jnp.array(x).reshape(-1,steps_to_predict,2) for x in goal_action]
    # l_n * 4 * 2
    # start_action = [jnp.array(x) for x in start_action]
    print('mid_action:',mid_action[0].shape,mid_action[1].shape)
    print('goal_action:',goal_action[0].shape,goal_action[1].shape)
    # print('start_action:',start_action[0].shape,record_info['start']['hippo'][0].shape)
    # actions = [start_action, mid_action, goal_action]
    actions = [mid_action, goal_action]
    # items = ['start', 'mid', 'goal']
    items = ['mid', 'goal']
    # clf_hipp_info = sklearn.gaussian_process.GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
    
    # clfs = [MLPClassifier(hidden_layer_sizes=8,max_iter=400,) for _ in range(3)]+[clf_hipp_info]
    hippos_to_train = [record_info['hist_hippo'][:-steps_to_predict-1]]+[jnp.concatenate(x,0) for x in hippos]
    theta_fasts_to_train = [record_info['hist_theta'][:-steps_to_predict-1]]+[jnp.concatenate(x,0) for x in theta_fasts]
    theta_slows_to_train = [jnp.concatenate(x,0) for x in theta_slows]
    hipp_infos_to_train = [jnp.concatenate(x,0) for x in hipp_infos]
    actions_0_to_train = [hist_pos[1:-steps_to_predict]-hist_pos[:-steps_to_predict-1]] + [jnp.concatenate(x,0)[:,0:1].repeat(args.replay_steps+4, 1) for x in actions]
    actions_1_to_train = [hist_pos[2:-steps_to_predict+1]-hist_pos[:-steps_to_predict-1]] + [jnp.concatenate(x,0)[:,1:2].repeat(args.replay_steps+4, 1) for x in actions]
    actions_2_to_train = [hist_pos[3:-steps_to_predict+2]-hist_pos[:-steps_to_predict-1]] + [jnp.concatenate(x,0)[:,2:3].repeat(args.replay_steps+4, 1) for x in actions]
    actions_3_to_train = [hist_pos[4:-steps_to_predict+3]-hist_pos[:-steps_to_predict-1]] + [jnp.concatenate(x,0)[:,3:4].repeat(args.replay_steps+4, 1) for x in actions]
    actions_to_train = [actions_0_to_train, actions_1_to_train, actions_2_to_train, actions_3_to_train]
    print(actions_0_to_train[0][0])
    print('hippos, thetas, theta_slows, hipp_infos, actions')
    for i in range(len(items)):
        print(items[i], hippos_to_train[i+1].shape, theta_fasts_to_train[i+1].shape, \
            theta_slows_to_train[i].shape, hipp_infos_to_train[i].shape, actions_0_to_train[i+1].shape)


    actions_0 = [list(map(lambda x: x[:,0:1], action_replay_loc)) for action_replay_loc in actions]
    actions_1 = [list(map(lambda x: x[:,1:2], action_replay_loc)) for action_replay_loc in actions]
    actions_2 = [list(map(lambda x: x[:,2:3], action_replay_loc)) for action_replay_loc in actions]
    actions_3 = [list(map(lambda x: x[:,3:4], action_replay_loc)) for action_replay_loc in actions]
    actions_all = [actions_0, actions_1, actions_2, actions_3]
    print(args.info_type)
    quantities_by_steps = []
    for step in range(steps_to_predict):
        decoder_path = './decoder/'+args.prefix+'_action_decoder_'+str(step)
        if os.path.exists(decoder_path):
            clfs = pickle.load(open(decoder_path, 'rb'))
            print('load decoder from:', decoder_path)
        else:
            clfs = [[Ridge() for step in range(args.replay_steps+4)] for _ in range(4)]
            clfs = cal_plot.train_action_decoder(args, clfs, hippos_to_train, theta_fasts_to_train, theta_slows_to_train, \
                                hipp_infos_to_train, actions_to_train[step], step)

        if args.info_type == 'hist_action':
            hist_state = record_info['hist_state']
            reward_n = record_info['reward_n']
            cal_plot.cal_plot_hist_phase(args, clfs[0], clfs[1], record_info['hist_hippo'], \
                                        record_info['hist_theta'][...,:args.theta_fast_size], hist_state, reward_n)
        elif args.info_type == 'replay_action':
            quantities = cal_plot.cal_plot_replay_curve(args, clfs, hippos, theta_fasts, theta_slows, hipp_infos, \
                                        phases, trajs, actions_all[step], 'action_'+str(step))
            quantities_by_steps.append(quantities)
    return quantities_by_steps
            




def eval_replay(args):
    save_path = './figures/'+args.prefix+'/record_info_'+str(args.pseudo_reward_idx)
    record_info = pickle.load(open(save_path, 'rb'))
    print('load record_info from', save_path)
    if 'args' in record_info.keys():
        print('load args from record_info')
        args = record_info['args']
    hist_state = record_info['hist_state']
    # print(hist_state.shape)
    print('st_x, st_y, at, rt, done, phase, step_count')
    # print(hist_state[:50,0,:])
    print('hist_state:',hist_state.shape)
    # st=0,1 at=2, rt=3, done=4, phase=5, step_count=6 
    hist_step_count = record_info['hist_state'][:,:,6]
    phases = [record_info['start']['phase'], record_info['mid']['phase'], record_info['goal']['phase']]
    eis = [record_info['start']['ei'], record_info['mid']['ei'], record_info['goal']['ei']]
    trajs = [record_info['start']['traj'], record_info['mid']['traj'], record_info['goal']['traj']]

    items = ['start', 'mid', 'goal']
    traj_switch = [[[] for j in range(8)] for i in range(len(items))]
    step_count_saved = [[[] for j in range(8)] for i in range(len(items))]
    # 0: mid, 1: goal, 2:start
    # Select appropriate replay
    print('selecting replay_switch')
    for type_index in range(len(items)):
        phase = phases[type_index]
        ei = eis[type_index]
        traj_of_interest = traj_switch[type_index]
        traj_to_be_selected = trajs[type_index]
        step_count_ratio = step_count_saved[type_index]
        for n in range(args.n_agents):
            if len(phase[n])<=2:
                continue
            # print(n,phase[n].reshape(-1))
            for i in range(len(phase[n])-1):
                if phase[n][i]==0 and phase[n][i+1]==1:
                    origin_step_count = hist_step_count[ei[n][i],n]
                    traj_of_interest[0].append(traj_to_be_selected[n][i+1])
                    step_count_ratio[0].append(hist_step_count[ei[n][i+1],n]/origin_step_count)
                    origin_step_count = hist_step_count[ei[n][i+1],n]

                    if i+2<len(phase[n]) and phase[n][i+2]==1:
                        traj_of_interest[1].append(traj_to_be_selected[n][i+2])
                        step_count_ratio[1].append(hist_step_count[ei[n][i+2],n]/origin_step_count)
                        origin_step_count = hist_step_count[ei[n][i+2],n]

                        if i+3<len(phase[n]) and phase[n][i+3]==1:
                            traj_of_interest[2].append(traj_to_be_selected[n][i+3])
                            step_count_ratio[2].append(hist_step_count[ei[n][i+3],n]/origin_step_count)
                            origin_step_count = hist_step_count[ei[n][i+3],n]

                            if i+4<len(phase[n]) and phase[n][i+4][1]==1:
                                traj_of_interest[3].append(traj_to_be_selected[n][i+4])
                                step_count_ratio[3].append(hist_step_count[ei[n][i+4],n]/origin_step_count)

                if phase[n][i]==1 and phase[n][i+1]==0:
                    origin_step_count = hist_step_count[ei[n][i],n]
                    traj_of_interest[4].append(traj_to_be_selected[n][i+1])
                    step_count_ratio[4].append(hist_step_count[ei[n][i+1],n]/origin_step_count)
                    origin_step_count = hist_step_count[ei[n][i+1],n]

                    if i+2<len(phase[n]) and phase[n][i+2]==0:
                        traj_of_interest[5].append(traj_to_be_selected[n][i+2])
                        step_count_ratio[5].append(hist_step_count[ei[n][i+2],n]/origin_step_count)
                        origin_step_count = hist_step_count[ei[n][i+2],n]

                        if i+3<len(phase[n]) and phase[n][i+3]==0:
                            traj_of_interest[6].append(traj_to_be_selected[n][i+3])
                            step_count_ratio[6].append(hist_step_count[ei[n][i+3],n]/origin_step_count)
                            origin_step_count = hist_step_count[ei[n][i+3],n]

                            if i+4<len(phase[n]) and phase[n][i+4][1]==0:
                                traj_of_interest[7].append(traj_to_be_selected[n][i+4])
                                step_count_ratio[7].append(hist_step_count[ei[n][i+4],n]/origin_step_count)


    step_count_saved = [list(map(lambda x: 1-jnp.mean(jnp.array(x)), step_count_saved[type_index])) for type_index in range(len(items))]
    traj_switch = [list(map(lambda x: jnp.array(x), traj_switch[type_index])) for type_index in range(len(items))]
    for replay_loc in range(len(items)):
        print(items[replay_loc])
        for i in range(len(traj_switch[replay_loc])):
            print(traj_switch[replay_loc][i].shape, end=' ')
        print()
    ##在这里改改然后就可以看一下consolidation和planning的流形区别
    row_text = ['01', '011', '0111', '01111', '10', '100', '1000', '10000']
    cal_plot.cal_plot_all_proportion(args, traj_switch, jnp.array(step_count_saved), row_text, items)
    # cal_plot_manifold_change()
    return 

def Igata_eval_replay(args):
    save_path = './figures/'+args.prefix+'/record_info_'+str(args.pseudo_reward_idx)+'_seed'+str(args.initkey)
    record_info = pickle.load(open(save_path, 'rb'))
    print('load record_info from', save_path)
    if 'args' in record_info.keys():
        print('load args from record_info')
        args = record_info['args']
    hist_state = record_info['hist_state']
    # print(hist_state.shape)
    print('st_x, st_y, at, rt, done, phase, step_count')
    # print(hist_state[:50,0,:])
    print('hist_state:',hist_state.shape)
    # st=0,1 at=2, rt=3, done=4, phase=5, step_count=6 
    hist_step_count = record_info['hist_state'][:,:,6]
    # phases = [record_info['start']['phase'], record_info['mid']['phase'], record_info['goal']['phase']]
    # eis = [record_info['start']['ei'], record_info['mid']['ei'], record_info['goal']['ei']]
    # trajs = [record_info['start']['traj'], record_info['mid']['traj'], record_info['goal']['traj']]
    phases = [record_info['mid']['phase'], record_info['goal']['phase']]
    eis = [record_info['mid']['ei'], record_info['goal']['ei']]
    trajs = [record_info['mid']['traj'], record_info['goal']['traj']]
    # print('phases:',len(phases[0]),phases[0][0])
    items = ['mid', 'goal']
    row_text = ['0', '1', '2', '3', '4']
    traj_switch = [[[] for j in range(len(row_text))] for i in range(len(items))]
    step_count = [[[] for j in range(len(row_text))] for i in range(len(items))]
    # 0: mid, 1: goal, 2:start
    # Select appropriate replay
    optimal_count = 1
    print('selecting replay_switch')
    for type_index in range(len(items)):
        phase = phases[type_index]
        ei = eis[type_index]
        traj_of_interest = traj_switch[type_index]
        traj_to_be_selected = trajs[type_index]
        step_count_ratio = step_count[type_index]
        for n in range(args.n_agents):
            if len(phase[n])<=2:
                continue
            # print(n,phase[n].reshape(-1))
            for i in range(len(phase[n])-1):
                if phase[n][i]==0 and phase[n][i+1]==1:
                    # origin_step_count = hist_step_count[ei[n][i],n]
                    traj_of_interest[1].append(traj_to_be_selected[n][i+1])
                    step_count_ratio[1].append(hist_step_count[ei[n][i+1],n]/optimal_count)
                    # origin_step_count = hist_step_count[ei[n][i+1],n]

                    if i+2<len(phase[n]) and phase[n][i+2]==1:
                        traj_of_interest[2].append(traj_to_be_selected[n][i+2])
                        step_count_ratio[2].append(hist_step_count[ei[n][i+2],n]/optimal_count)
                        # origin_step_count = hist_step_count[ei[n][i+2],n]

                        if i+3<len(phase[n]) and phase[n][i+3]==1:
                            traj_of_interest[3].append(traj_to_be_selected[n][i+3])
                            step_count_ratio[3].append(hist_step_count[ei[n][i+3],n]/optimal_count)
                            # origin_step_count = hist_step_count[ei[n][i+3],n]

                            if i+4<len(phase[n]) and phase[n][i+4]==1:
                                traj_of_interest[4].append(traj_to_be_selected[n][i+4])
                                step_count_ratio[4].append(hist_step_count[ei[n][i+4],n]/optimal_count)
                            
                                # if i+5<len(phase[n]) and phase[n][i+5]==1:
                                #     traj_of_interest[5].append(traj_to_be_selected[n][i+5])
                                #     step_count_ratio[5].append(hist_step_count[ei[n][i+5],n]/optimal_count)

                                    # if i+6<len(phase[n]) and phase[n][i+6]==1:
                                    #     traj_of_interest[6].append(traj_to_be_selected[n][i+6])
                                    #     step_count_ratio[6].append(hist_step_count[ei[n][i+6],n]/optimal_count)

                if phase[n][i]==0 and phase[n][i+1]==0:
                    # origin_step_count = hist_step_count[ei[n][i],n]
                    traj_of_interest[0].append(traj_to_be_selected[n][i+1])
                    step_count_ratio[0].append(hist_step_count[ei[n][i+1],n]/optimal_count)
                    # origin_step_count = hist_step_count[ei[n][i],n]

                #     if i+2<len(phase[n]) and phase[n][i+2]==0:
                #         traj_of_interest[5].append(traj_to_be_selected[n][i+2])
                #         step_count_ratio[5].append(hist_step_count[ei[n][i+2],n]/origin_step_count)
                #         origin_step_count = hist_step_count[ei[n][i+2],n]

                #         if i+3<len(phase[n]) and phase[n][i+3]==0:
                #             traj_of_interest[6].append(traj_to_be_selected[n][i+3])
                #             step_count_ratio[6].append(hist_step_count[ei[n][i+3],n]/origin_step_count)
                #             origin_step_count = hist_step_count[ei[n][i+3],n]

                #             if i+4<len(phase[n]) and phase[n][i+4][1]==0:
                #                 traj_of_interest[7].append(traj_to_be_selected[n][i+4])
                #                 step_count_ratio[7].append(hist_step_count[ei[n][i+4],n]/origin_step_count)

    for replay_loc in range(len(items)):
        print(items[replay_loc])
        for i in range(len(traj_switch[replay_loc])):
            print(len(traj_switch[replay_loc][i]), end=' ')
        print()
    step_count = [list(map(lambda x: jnp.array(x), step_count[type_index])) for type_index in range(len(items))]
    step_count_mean = [list(map(lambda x: jnp.mean(jnp.array(x)), step_count[type_index])) for type_index in range(len(items))]
    traj_switch = [list(map(lambda x: jnp.stack(x,0), traj_switch[type_index])) for type_index in range(len(items))]
    for replay_loc in range(len(items)):
        print(items[replay_loc])
        for i in range(len(traj_switch[replay_loc])):
            print(traj_switch[replay_loc][i].shape, end=' ')
        print()
    ##在这里改改然后就可以看一下consolidation和planning的流形区别)
    line_chart_mid_goal = cal_plot.cal_plot_all_proportion(args, traj_switch, jnp.array(step_count_mean), row_text, items)

    #  5(pre_learn, learning_1, learning_2, learning_3, post_learn)*4(past_cons, new_cons, past_plan, new_plan)
    # cal_plot_manifold_change()
    # only return mid step count
    return line_chart_mid_goal, step_count[0]

def eval_all_hippo_theta(args):
    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key) 
    env_state, buffer_state, running_encoder_state, running_hippo_state, \
        running_hippo_std_state, running_policy_state = train.init_states(args, subkey)
    key, subkey = jax.random.split(key)
    _, env_state = env.pseudo_reset(args.width, args.height, args.n_agents, subkey, args.pseudo_reward)
    actions = jnp.zeros((args.n_agents, 1), dtype=jnp.int8)
    hippo_hidden = jnp.zeros((args.n_agents, args.hidden_size))
    theta = jnp.zeros((args.n_agents, args.theta_hidden_size))

    hist_hippo = jnp.zeros((args.total_eval_steps, *hippo_hidden.shape))
    hist_theta = jnp.zeros((args.total_eval_steps, *theta.shape))
    hist_phase = jnp.zeros((args.total_eval_steps, args.n_agents))
    reward_ei = []
    goal_ei = []
    reward_n = [[] for _ in range(args.n_agents)]
    phase_n = [[] for _ in range(args.n_agents)]
    hippo_mid_replay = []
    theta_mid_replay = []
    hippo_goal_replay = []
    theta_goal_replay = []
    traj_mid_replay = []
    traj_goal_replay = []
    phase_mid_replay = []
    output_goal_replay = []
    mid_replay_idx_00 = []
    mid_replay_idx_01 = []
    mid_replay_idx_10 = []
    mid_replay_idx_11 = []


    mid_replay_idx_011 = []
    mid_replay_idx_0111 = []

    mid_replay_idx_100 = []
    mid_replay_idx_1000 = []

    for ei in range(args.total_eval_steps):
        if ei % 500 == 499:
            print(ei+1)
        key, subkey = jax.random.split(subkey)
        st = env_state['current_pos']
        at = actions
        rc = env_state['reward_center']
        phase = env_state['phase']
        step_count = env_state['step_count']
        env_state, buffer_state, actions, hippo_hidden, theta, rewards, done, replayed_history, first_hippo_theta_output, hipp_info, value \
            = train.eval_step(env_state, buffer_state, running_encoder_state, running_hippo_state, 
                         running_hippo_std_state, running_policy_state,
                         subkey, actions, hippo_hidden, theta,
                         args.n_agents, args.bottleneck_size, args.hidden_size, args.replay_steps, args.height, args.width,
                         args.visual_prob, args.eval_temperature, args.reset_prob,
                         args.noise_scale, args.pseudo_reward, args.block_idx)
        
        hist_hippo = hist_hippo.at[ei].set(hippo_hidden)
        hist_theta = hist_theta.at[ei].set(theta)
        hist_phase = hist_phase.at[ei].set(phase.reshape(-1))
        replayed_hippo_history, replayed_theta_history, output_history, hipp_info_history = replayed_history
        for n in range(args.n_agents):
            if jnp.isclose(rewards[n], 0.5):
                reward_ei.append((ei, n))
                reward_n[n].append((ei,0))
                hippo_mid_replay.append(replayed_history[0][:,n])
                theta_mid_replay.append(replayed_history[1][:,n])
                traj_mid_replay.append(jnp.argmax(output_history[...,:args.width*args.height],axis=-1)[:,n])
                phase_mid_replay.append(phase[n].item())
                if phase[n]==0 and phase_n[n] and phase_n[n][-1]==0:
                    mid_replay_idx_00.append(len(hippo_mid_replay)-1)
                if phase[n]==1 and phase_n[n] and phase_n[n][-1]==0:
                    mid_replay_idx_01.append(len(hippo_mid_replay)-1)
                if phase[n]==0 and phase_n[n] and phase_n[n][-1]==1:
                    mid_replay_idx_10.append(len(hippo_mid_replay)-1)
                if phase[n]==1 and phase_n[n] and phase_n[n][-1]==1:
                    mid_replay_idx_11.append(len(hippo_mid_replay)-1)
                #这里的phase[n]表示的就是“原来这个奖励在什么地方”，而不是改变之后的

                if phase[n]==0 and len(phase_n[n])>2 and phase_n[n][-1]==0 and phase_n[n][-2]==1:
                    mid_replay_idx_100.append(len(hippo_mid_replay)-1)
                if phase[n]==0 and len(phase_n[n])>3 and phase_n[n][-1]==0 and phase_n[n][-2]==0 and phase_n[n][-3]==1:
                    mid_replay_idx_1000.append(len(hippo_mid_replay)-1)
                if phase[n]==1 and len(phase_n[n])>2 and phase_n[n][-1]==1 and phase_n[n][-2]==0:
                    mid_replay_idx_011.append(len(hippo_mid_replay)-1)
                if phase[n]==1 and len(phase_n[n])>3 and phase_n[n][-1]==1 and phase_n[n][-2]==1 and phase_n[n][-3]==0:
                    mid_replay_idx_0111.append(len(hippo_mid_replay)-1)

                phase_n[n].append(phase[n].item())
            if jnp.isclose(rewards[n], 1):
                goal_ei.append((ei, n))
                reward_n[n].append((ei,1))
                # hippo_goal_replay.append(replayed_history[0][:,n])
                # theta_goal_replay.append(replayed_history[1][:,n])
                # output_goal_replay.append(output_history[:,n])
                # traj_goal_replay.append(jnp.argmax(output_history[...,:args.width*args.height],axis=-1)[:,n])

        mid_hippo = jnp.array(hippo_mid_replay)
        mid_theta = jnp.array(theta_mid_replay)      
        # goal_hippo = jnp.array(hippo_goal_replay)
        # goal_theta = jnp.array(theta_goal_replay)

    def output_traj(replay_trajs: jax.numpy.ndarray):
        row = replay_trajs//args.width
        col = replay_trajs%args.width
        trajs = jnp.stack((row, col), axis=-1)
        return trajs

    cons_plan_condition = \
        cal_plot.select_consolidation_plannning(args.width, args.height, jnp.stack(traj_mid_replay), jnp.stack(phase_mid_replay),
                                                env.ck0_x, env.ck0_y, env.ck1_x, env.ck1_y,
                                                env.ck0_x_g_lb, env.ck0_y_g_lb, env.ck1_x_g_lb, env.ck1_y_g_lb)
    cons_plan_idx = [jnp.array(jnp.where(x)) for x in cons_plan_condition]

    # sample_cons1 = jnp.stack(traj_mid_replay)[consolidation_idx_ck1]
    # sample_cons2 = jnp.stack(traj_mid_replay)[consolidation_idx_ck2]
    # sample_plan1 = jnp.stack(traj_mid_replay)[planning_idx_ck1]
    # sample_plan2 = jnp.stack(traj_mid_replay)[planning_idx_ck2]
    # print('sample_cons1', [output_traj(i) for i in sample_cons1])
    # print('sample_cons2', [output_traj(i) for i in sample_cons2])
    # print('sample_plan1', [output_traj(i) for i in sample_plan1])
    # print('sample_plan2', [output_traj(i) for i in sample_plan2])
        
    # func_proportion = jnp.mean(consolidation_ck1 | consolidation_ck2 | planning_ck1 | planning_ck2)
    # print('functional replay proportion', func_proportion)
    cons_plan_titles = ['consolidation_ck0', 'consolidation_ck1', 'planning_ck0', 'planning_ck1']
    switch_idx = [jnp.array(mid_replay_idx_00), jnp.array(mid_replay_idx_01), jnp.array(mid_replay_idx_10), jnp.array(mid_replay_idx_11)]
    switch_titles = ['mid_replay_idx_00', 'mid_replay_idx_01', 'mid_replay_idx_10', 'mid_replay_idx_11']
    transition_idx = [jnp.array(mid_replay_idx_10), jnp.array(mid_replay_idx_100), jnp.array(mid_replay_idx_1000),\
         jnp.array(mid_replay_idx_01), jnp.array(mid_replay_idx_011), jnp.array(mid_replay_idx_0111)]
    transition_titles = ['mid_replay_idx_10', 'mid_replay_idx_100', 'mid_replay_idx_1000',\
         'mid_replay_idx_01', 'mid_replay_idx_011', 'mid_replay_idx_0111']
    if args.suffix == 'cons_plan':
        idx_titles = [cons_plan_idx, cons_plan_titles]
    elif args.suffix == 'switch':
        idx_titles = [switch_idx, switch_titles]
    elif args.suffix == 'transition':
        idx_titles = [transition_idx, transition_titles]
    # cal_plot.plot_manifold(args, hist_hippo, hist_theta, 
    #                 (jnp.array(reward_ei), jnp.array(goal_ei)), reward_n,
    #                 (mid_hippo, goal_hippo, mid_theta, goal_theta),
    #                 hist_phase, idx_titles[0], idx_titles[1])

    cal_plot.cal_plot_mutual_distance(args, mid_hippo, mid_theta, idx_titles[0])

    cal_plot.cal_plot_manifold_difference(args, mid_hippo, mid_theta, idx_titles)
    return 



if __name__ == '__main__':
    # args_from_train = train.parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--train_every', type=int, default=250)
    parser.add_argument('--n_agents', type=int, default=128)
    parser.add_argument('--max_size', type=int, default=250)  # max_size of buffer
    parser.add_argument('--sample_len', type=int, default=250)  # sample len from buffer: at most max_size - 1
    parser.add_argument('--epochs', type=int, default=int(1e6))

    parser.add_argument('--log_name', type=str, default='train_log')
    parser.add_argument('--model_path', type=str, default='./modelzoo')
    parser.add_argument('--replay_path', type=str, default='./replayzoo')
    parser.add_argument('--no_save', action='store_true', default=False)

    parser.add_argument('--mid_reward', type=float, default=0.5)
    parser.add_argument('--replay_steps', type=int, default=4)  # todo: tune
    parser.add_argument('--eval_temperature', type=float, default=0.5)

    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=1e-2)
    parser.add_argument('--n_train_time', type=int, default=6)
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--n_eval_steps', type=int, default=200)
    parser.add_argument('--record_every', type=int, default=int(1e5))
    # params that should be the same with config.py
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--width', type=int, default=5)
    parser.add_argument('--height', type=int, default=5)
    parser.add_argument('--n_action', type=int, default=4)
    parser.add_argument('--visual_prob', type=float, default=0.2)

    parser.add_argument('--load_encoder', type=str)  # todo: checkpoint
    parser.add_argument('--load_hippo', type=str)
    parser.add_argument('--load_policy', type=str)
    parser.add_argument('--load_hippo_std', type=str, default='hippo_std_1000000')
    parser.add_argument('--record_save', action='store_true', default=False)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--suffix', type=str, default='switch')
    # Init params for hippo and policy
    parser.add_argument('--bottleneck_size', type=int)
    parser.add_argument('--policy_scan_len', type=int, default=240)
    parser.add_argument('--theta_hidden_size', type=int)
    parser.add_argument('--theta_fast_size', type=int)
    parser.add_argument('--drop_out_rate', type=float, default=0)
    parser.add_argument('--noise_scale', type=float)
    parser.add_argument('--hippo_mem_len', type=int, default=5)

    parser.add_argument('--total_eval_steps', '-et', type=int, default=400)
    # for all_hippo_theta
    parser.add_argument('--reset_prob', type=float, default=0.1)

    parser.add_argument('--dr_every', type=int, default=50)
    parser.add_argument('--figure_path', type=str, default='./figure')
    parser.add_argument('--option', '-op', type=str)

    parser.add_argument('--zero_mid_idx', action='store_true')
    parser.add_argument('--zero_goal_idx', action='store_true')
    parser.add_argument('--zero_traj_idx', action='store_true')
    parser.add_argument('--replay_before_mid_value', type=int, default=0)
    parser.add_argument('--replay_after_mid_value', type=int, default=0)
    parser.add_argument('--replay_before_goal_value', type=int, default=0)
    parser.add_argument('--replay_after_goal_value', type=int, default=0)
    parser.add_argument('--path_start_value', type=int, default=0)
    parser.add_argument('--path_mid_value', type=int, default=0)
    parser.add_argument('--path_goal_value', type=int, default=0)
    parser.add_argument('--colormap', type=str, default='bwr')
    parser.add_argument('--replay_interest', action='store_true')

    parser.add_argument('--replay_type', type=str, default='mid')
    parser.add_argument('--info_type', type=str, default='replay_acc')
    parser.add_argument('--pseudo_reward_idx', type=int, default=0)
    parser.add_argument('--block_option', type=str)
    parser.add_argument('--block_idx', type=int, default=-1)
    parser.add_argument('--initkey', type=int, default=0)
    args = parser.parse_args()
    
    # for k,v in args_from_train.__dict__.items():
    #     args.__setattr__(k,v)
    assert (args.bottleneck_size is not None)&(args.theta_hidden_size is not None)\
            &(args.theta_fast_size is not None)&(args.noise_scale is not None)\
            &(args.prefix is not None)&(args.option is not None), \
            'missing args in [bottleneck_size, theta_hidden_size, theta_fast_size, noise_scale, prefix, option]'
    args.pseudo_reward = env.pseudo_reward_list[args.pseudo_reward_idx]
    plt.rc('font', size=30)
    if 'figures' not in os.listdir('./'):
        os.mkdir('./figures')
    if args.prefix not in os.listdir('./figures'):
        os.mkdir('./figures/'+args.prefix)
    if args.option=='eval_replay':
        
        Igata_eval_replay(args)
    elif args.option=='eval_all_hippo_theta':
        eval_all_hippo_theta(args)
    elif args.option=='eval_video':
        args.total_eval_steps=300
        args.n_agents=8
        eval_video(args)
    elif args.option=='eval_info':
        eval_info(args)
    elif args.option=='decoding_info':
        if args.info_type in ['hist_phase', 'replay_phase']:
            decoding_phase(args)
        elif args.info_type in ['hist_action', 'replay_action']:
            decoding_action_plan(args)
        elif args.info_type == 'energy':
            decoding_energy(args)
    elif args.option == 'eval_all':
        args.pseudo_reward = env.pseudo_reward_list[0]
        args.pseudo_reward_idx = 0
        eval_info(args)
        decoding_phase(args)
        args.pseudo_reward = env.pseudo_reward_list[1]
        args.pseudo_reward_idx = 1
        eval_info(args)
        eval_replay(args)
        args.total_eval_steps=100
        args.n_agents=8
        eval_video(args)
    elif args.option == 'eval_value_map':
        Igata_eval_value_map(args)
    elif args.option == 'eval_manifold':
        eval_manifold(args)
    elif args.option=='eval_info_circularly':
        eval_info_circularly(args)
    elif args.option=='eval_replay_circularly':
        eval_replay_circularly(args)
    elif args.option=='Ablation_KL_div':
        Ablation_KL_div(args)
    elif args.option=='decoding_phase_circularly':
        decoding_phase_circularly(args)
    elif args.option=='decoding_action_circularly':
        decoding_action_circularly(args)
    elif args.option == 'eval_value_map_circularly':
        eval_value_map_circularly(args)
    elif args.option == 'eval_manifold_circularly':
        eval_manifold_circularly(args)
    elif args.option == 'eval_subspace_dimension_circularly':
        eval_subspace_dimension_circularly(args)
    elif args.option == 'eval_manifold_stability_circularly':
        eval_manifold_stability_circularly(args)

    elif args.option == 'eval_info_over_random_policy':
        eval_info_over_random_policy(args)
    elif args.option == 'decoding_moving_direction':
        decoding_moving_direction(args)
    elif args.option == 'cal_forward_backward_replay':
        cal_forward_backward_replay(args)
    elif args.option == 'cal_forward_backward_replay_circularly':
        cal_forward_backward_replay_circularly(args)
    elif args.option == 'cal_recapitulate_past_experiences':
        cal_recapitulate_past_experiences(args)
    elif args.option == 'cal_reward_times_met_circularly':
        cal_reward_times_met_circularly(args)
    elif args.option == 'compare_replay_adjacency':
        compare_replay_adjacency(args)
    elif args.option == 'compare_replay_adjacency_circularly':
        compare_replay_adjacency_circularly(args)
    elif args.option == 'cal_reward_times_distribution_circularly':
        cal_reward_times_distribution_circularly(args)
    