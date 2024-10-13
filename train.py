"""
Main module with PPO navigation task
"""
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

    parser.add_argument('--log_name', type=str, default='train_log')
    parser.add_argument('--model_path', type=str, default='./modelzoo')
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
    parser.add_argument('--load_encoder', type=str, default=None)  # todo: checkpoint
    parser.add_argument('--load_hippo', type=str)
    parser.add_argument('--load_policy', type=str)

    parser.add_argument('--prefix', type=str, required=True)

    # Init params for hippo and policy
    parser.add_argument('--bottleneck_size', type=int, default=4)
    parser.add_argument('--policy_scan_len', type=int, default=20)
    parser.add_argument('--hippo_mem_len', type=int, default=5)
    parser.add_argument('--hippo_pred_len', type=int, default=5)
    parser.add_argument('--pc_sigma', type=float, default=1)

    parser.add_argument('--eval_temperature', type=float, default=0.5)
    args = parser.parse_args()
    return args


@partial(jax.jit, static_argnames=['temperature'])
def sample_from_policy(logit, key, temperature):
    # logit [n, 4]
    # return action [n, 1]
    def sample_once(logit_n, subkey):
        # logit_n[4,]
        action_n = jax.random.choice(subkey, jnp.arange(0, logit_n.shape[-1]), shape=(1,),
                                     p=jax.nn.softmax(logit_n / temperature, axis=-1))
        return action_n

    subkeys = jax.random.split(key, num=logit.shape[0] + 1)
    key, subkeys = subkeys[0], subkeys[1:]
    action = jax.vmap(sample_once, (0, 0), 0)(logit, subkeys).astype(jnp.int8)
    return action


@partial(jax.jit, static_argnames=['replay_steps', 'clip_param', 'entropy_coef', 'n_train_time', 'policy_scan_len',
                                   'hippo_pred_len', 'hippo_mem_len', 'grid_size', 'pc_sigma'])
def train_step(states, batch, replay_steps, clip_param, entropy_coef, n_train_time, policy_scan_len, key, hippo_pred_len,
               hippo_mem_len, grid_size, pc_centers, pc_sigma):
    """Train for a single step with rollouts from the buffer, update policy_state, encoder_state and hippo_state"""
    # state (encoder_state, hippo_state, policy_state)
    # obs_emb_t[t, n, d], action_emb_t-1, h_t[t, n, h] (before replay), theta_t[t, n, h] (before replay)，
    # rewards_t[t, n, 1], action_t[t, n, 1], policy_t[t, n, 4], value_t[t, n, 1]
    # traced_rewards_t[t, n, 1]: exp avg of rewards_t (MC sample of true value)
    # all rollouts start from where with rewards or reset, and first obs is not zero (optional)
    # his means that the data is from buffer (history)
    # index:
    # o_e(t), a_e(t-1), h(t), theta(t), r(t-1), a(t), logit(t), v(t), done(t), pos(t)

    encoder_state, hippo_state, policy_state = states
    sample_len = batch['theta'].shape[0]
    n_agents = batch['theta'].shape[1]
    theta_hidden_size = batch['theta'].shape[-1]
    num_cells = grid_size ** 2
    grid_size = batch['obs'].shape[-2]
    theta_zero = jnp.zeros((n_agents, theta_hidden_size))
    next_s_label = batch['next_s'][..., 0] * grid_size + batch['next_s'][..., 1]
    rewards = batch['oe_ae_r'][..., -1]
    jax.debug.print('replay ratio all {a}', a=jnp.where(rewards > 0, 1, 0).mean())
    key, subkey = jax.random.split(key) 


    def loss_fn(policy_params, hippo_params, encoder_params, batch, key):
        """Train the hippocampus and the prefrontal cortex"""
        # his_theta [l, n, h]
        # obs_embed [l, n, 64], action_embed [l, n, 4]
        # his_action [l, n, 1], his_logits [l, n, 4], his_rewards [l, n, 1], his_values[l, n, 1]
        # his_traced_rewards [l, n, 1]

        # 1. Replay: generate his_replayed_theta =======================================================
        # hippo_and_theta, policy_params, hippo_state, policy_state, s_a_r


        def propagate_theta_hippo(key, index, policy_scan_len):
            replay_fn_to_scan = partial(replay_fn, policy_params=policy_params, hippo_state=hippo_state, policy_state=policy_state,
                                    oe_ae_r=jnp.zeros_like(batch['oe_ae_r'][0]))
            def propagate_hippo_once(prev_hippo_hidden, input_of_hippo, theta):
                obs, prev_action, prev_reward = input_of_hippo
                obs_embed, action_embed = encoder_state.apply_fn({'params': encoder_params},
                                                                obs, prev_action)
                oe_ae_r = jnp.concatenate((obs_embed, action_embed, prev_reward), axis=-1)
                new_hidden, output, theta_info = hippo_state.apply_fn({'params': hippo_params},
                                                                    prev_hippo_hidden, theta, oe_ae_r)
                return new_hidden, output
        
            def propagate_theta_once(prev_theta, input_of_policy):
                # [n, th]
                # [n, 1], [n, h]
                key, oe_ae_r, hippo_hidden = input_of_policy # ot, at-1, rt-1, ht
                rewards = oe_ae_r[..., -1:]
                replay_keys = jax.random.split(key, replay_steps + 1)
                key, replay_keys = replay_keys[0], replay_keys[1:]
                # outside_hipp_info = jnp.zeros((replay_steps, n_agents, bottleneck_size))
                (replayed_hippo_hidden, replayed_theta), _ = jax.lax.scan(replay_fn_to_scan,
                                                                        init=(hippo_hidden, prev_theta),
                                                                        xs=replay_keys)
                # replayed_theta = prev_theta
                # replayed_hippo_hidden = hippo_hidden
                replayed_theta = jnp.where(rewards > 0, replayed_theta, prev_theta)
                replayed_hippo_hidden = jnp.where(rewards > 0, replayed_hippo_hidden,
                                                jnp.zeros(replayed_hippo_hidden.shape))
                # replayed_hippo_hidden = jnp.zeros(replayed_hippo_hidden.shape)
                key, dropout_key = jax.random.split(key)
                # outside_hipp_info = jnp.zeros((n_agents, bottleneck_size))
                new_theta, (policy, value, _) = policy_state.apply_fn({'params': policy_params},
                                                            replayed_hippo_hidden, replayed_theta, oe_ae_r)
                # reset_theta = jnp.where(done & jnp.isclose(rewards, 0.), 0, new_theta)

                return new_theta, (new_theta, policy, value)
        
            indexes_for_propagate_once = jnp.arange(policy_scan_len) + index

            key_to_scan = jax.random.split(key, num=policy_scan_len + 1)
            key, key_to_scan = key_to_scan[0], key_to_scan[1:]
            oe_ae_r_to_scan = jnp.take(batch['oe_ae_r'], indexes_for_propagate_once, axis=0) # ot, at-1, rt-1
            # hippo_hidden_to_scan = jnp.take(batch['hippo_hidden'], indexes_for_propagate_once, axis=0) # ht-1
            new_hippo_hidden_to_scan = jnp.take(batch['new_hippo_hidden'], indexes_for_propagate_once, axis=0) # ht
            obs_to_scan = jnp.take(batch['obs'], indexes_for_propagate_once, axis=0) # ot
            # done_to_scan = jnp.take(batch['done'], indexes_for_propagate_once, axis=0) # done t-1
            prev_action_to_scan = jnp.take(batch['prev_action'], indexes_for_propagate_once, axis=0) # at-1
            prev_reward_to_scan = oe_ae_r_to_scan[..., -1:] # rt-1

            new_hippo_hidden, hist_hippo_output = jax.lax.scan(partial(propagate_hippo_once, theta=theta_zero),
                                                         init=jnp.take(batch['hippo_hidden'], index, axis=0),
                                                        xs=(obs_to_scan, prev_action_to_scan, prev_reward_to_scan))
            new_theta, (_, hist_policy, hist_value) = jax.lax.scan(propagate_theta_once,
                                                                   init=jnp.take(batch['theta'], index, axis=0),
                                                                   xs=(key_to_scan, oe_ae_r_to_scan, 
                                                                       new_hippo_hidden_to_scan))
            return hist_policy, hist_value, hist_hippo_output



        # _, his_replayed_theta = jax.lax.scan(propagate_theta, init=batch['his_theta'][0], xs=(batch['his_rewards'],
        #                                                                                       his_replayed_theta))
        # theta(t) = f(r(t-1), theta(t))
        # propagate theta
        # his_replayed_theta = jnp.where(batch['his_rewards'] > 0, his_replayed_theta, batch['his_theta'])
        # using scan instead of jnp.where to ensure that gradients of each action should affect the invariable theta 
        # generated by the same previous replay 
        propagate_theta_hippo_for_seq = partial(propagate_theta_hippo, policy_scan_len=policy_scan_len)
        # sample_len - 1 - policy_scan_len
        key_to_propagate = jax.random.split(key, num=sample_len - policy_scan_len + 1)
        key, key_to_propagate = key_to_propagate[0], key_to_propagate[1:]
        scan_policy_logits, scan_value, scan_hippo_output = jax.vmap(propagate_theta_hippo_for_seq, (0, 0), 0)(
            key_to_propagate, jnp.arange(sample_len - policy_scan_len))
        # [t, policy_scan_len, n, h], [t, policy_scan_len, n, 4], [t, policy_scan_len, n, 1]


        ## Calculate hippo loss
        def hippo_loss(all_preds, index):
            index_to_scan = jnp.arange(sample_len - policy_scan_len) + index
            next_s_label_to_scan = jnp.take(next_s_label, index_to_scan, axis=0) # [t, n, 2]
            def generate_place_cell(x):
                centers = pc_centers
                sigma = pc_sigma
                # x[n, 2], centers[m, 2]
                @jax.jit
                def cal_dist(pos, cents, sigma):
                    # pos[2,], cents[m, 2]
                    return - ((pos.reshape((1, -1)) - cents) ** 2).sum(axis=-1) / (2 * sigma ** 2)  # [m,]
                activation = jax.vmap(cal_dist, (0, None, None), 0)(x, centers, sigma)  # [n, m]
                activation = nn.softmax(activation, axis=-1)
                return activation
            
            place_cell_activation = jax.vmap(generate_place_cell)(next_s_label_to_scan) # [t, n, m]
            

            preds_place = all_preds[:, :, :num_cells]
            mem_preds_rewards = all_preds[:, :, num_cells:]  # [l, n, 1+hml]
            preds_rewards = mem_preds_rewards[-hippo_pred_len:,:,-1]  # [hpl, n]

            mem_rewards = mem_preds_rewards[-hippo_pred_len,:,:-1]  # [n, hml]
            mem_rewards = jnp.transpose(mem_rewards, (1, 0))  # [hml, n]

            loss_pathint = optax.softmax_cross_entropy(preds_place, place_cell_activation).mean()
            # exclude the last step of place_cell prediction
            # [:-1]: not to pred the last place cell because of the masked rewards
            # [l, n]
            acc_pred = (jnp.argmax(preds_place, axis=-1) == next_s_label_to_scan).astype(jnp.float32).mean()

            # pred last
            rewards = batch['oe_ae_r'][:, :, -1]  # [l, n]
            rewards_label = rewards[-(hippo_pred_len+hippo_mem_len):]  # [hpl+hml, n]  # todo: 1: :-1; only predict the last one
            # rewards_label = batch['rewards'][-(hippo_pred_len+hippo_mem_len):-hippo_pred_len,:,0]
            # state = jnp.concatenate((batch['action'], batch['current_pos'], batch['rewards'], preds_rewards, \
            #     batch['checked'], batch['step_count'], batch['reward_center'], jnp.argmax(batch['place_cells'],-1,keepdims=True), jnp.argmax(preds_place,-1,keepdims=True)),-1)
            # # jax.debug.breakpoint()

            # jax.debug.print("reward_label_{shape}:{a}",shape=rewards_label.shape,a=batch['rewards'][-hippo_pred_len-hippo_mem_len:,0].reshape(-1))
            # jax.debug.print("mem_rewards_{shape}:{a}",shape=mem_rewards.shape,a=mem_rewards[:,0].reshape(-1))
            # jax.debug.print("preds_rewards_{shape}:{a}",shape=preds_rewards.shape,a=preds_rewards[:,0].reshape(-1))
            # st=0,1 at=2, rt=3, done=4, reward_center=5,6, step_count=7

            all_r = jnp.concatenate((mem_rewards, preds_rewards),0)
            loss_r = jnp.square(all_r - rewards_label)/2  # [hpl+hml, n]  # only consider the last hpl step

            recall_0 = jnp.where((loss_r < 0.1) & (jnp.abs(rewards_label - 0.) < 1e-3), 1, 0)  # todo: acc criterion: < 0.2
            recall_r = jnp.where((loss_r < 0.1) & (jnp.abs(rewards_label - 0.5) < 1e-3), 1, 0)
            recall_g = jnp.where((loss_r < 0.1) & (jnp.abs(rewards_label - 1.) < 1e-3), 1, 0)

            precision_0 = jnp.where((jnp.abs(all_r - 0.) < 0.1)&(jnp.abs(rewards_label - 0.) < 1e-3), 1, 0)
            precision_r = jnp.where((jnp.abs(all_r - 0.5) < 0.1)&(jnp.abs(rewards_label - 0.5) < 1e-3), 1, 0)
            precision_g = jnp.where((jnp.abs(all_r - 1.) < 0.1)&(jnp.abs(rewards_label - 1.) < 1e-3), 1, 0)
            loss_r = jnp.where(rewards_label > 0.4, loss_r * 40, loss_r).mean()  # todo: weighted loss, times by 10

            recall_0 = jnp.sum(recall_0)/jnp.sum(jnp.where((jnp.abs(rewards_label-0)<1e-3),1,0))
            recall_r = jnp.sum(recall_r)/jnp.sum(jnp.where((jnp.abs(rewards_label-0.5)<1e-3),1,0))
            recall_g = jnp.sum(recall_g)/jnp.sum(jnp.where((jnp.abs(rewards_label-1)<1e-3),1,0))
            
            precision_0 = jnp.sum(precision_0)/jnp.sum(jnp.where((jnp.abs(all_r-0)<0.1),1,0))
            precision_r = jnp.sum(precision_r)/jnp.sum(jnp.where((jnp.abs(all_r-0.5)<0.1),1,0))
            precision_g = jnp.sum(precision_g)/jnp.sum(jnp.where((jnp.abs(all_r-1)<0.1),1,0))

            f1_0 = 2*precision_0*recall_0/(precision_0+recall_0)
            f1_r = 2*precision_r*recall_r/(precision_r+recall_r)
            f1_g = 2*precision_g*recall_g/(precision_g+recall_g)
            # fixme: cumsum_rewards > 0.6, considering random reward value is 0.5, > 0.6 means the second time met a reward
            
            loss = loss_pathint + loss_r * 0.8  # todo: *0.1
            return loss, (loss_r, loss_pathint, recall_0, recall_r, recall_g, \
                precision_0, precision_r, precision_g, f1_0, f1_r, f1_g, acc_pred)
        
        hippo_losses, (losses_r, losses_pathint, recall_0, recall_r, recall_g, \
            precision_0, precision_r, precision_g, f1_0, f1_r, f1_g, acc_preds) = \
                jax.vmap(hippo_loss, (1, 0), 0)(scan_hippo_output, jnp.arange(policy_scan_len))
        # 2. Take action ===================================================================================
        '''        
            def forward_fn1(theta_t, obs_embed_t, action_embed_t, hippo_hidden_t):
            # theta[n, h]; obs[n, 48]; hippo_hidden_t[n, h]
            _, (policy_t, value_t, _) = policy_state.apply_fn({'params': policy_params},
                                                              theta_t, obs_embed_t,
                                                              action_embed_t,
                                                              hippo_hidden_t)
            return policy_t, value_t  # [n, 4], [n, 1]
        '''

        # [scan_len:-1, n, h]

        # a(t), v(t) = f(theta(t), o(t))
        # 实际上这里应该是o(t-1)
        # the pfc cannot see hippo during walk
        # policy_logits_t[t, n, 4], value_t[t, n, 1]

        # 3. PPO =========================================================================================
        his_logits = batch['logits'][:-1]
        his_action = batch['next_action'][:-1]
        his_values = batch['values'][:-1]
        his_traced_rewards = batch['his_traced_rewards'][1:]

        def PPO_loss(policy_logits, value, index):
            index_to_scan = jnp.arange(sample_len - 1 - policy_scan_len) + index
            his_logits_to_scan = jnp.take(his_logits, index_to_scan, axis=0)
            his_action_to_scan = jnp.take(his_action, index_to_scan, axis=0)
            his_values_to_scan = jnp.take(his_values, index_to_scan, axis=0)
            his_traced_rewards_to_scan = jnp.take(his_traced_rewards, index_to_scan, axis=0)
            # [l, n, 4], [l, n, 1], [l, n, 4], [l, n, 1], [l, n, 1]
            ratio = jnp.exp(
                jax.nn.log_softmax(policy_logits, axis=-1) - jax.nn.log_softmax(his_logits_to_scan,
                                                                                axis=-1))  # [t, n, 4]
            t, n, _ = ratio.shape
            index_t = jnp.repeat(jnp.arange(0, t).reshape((t, 1)), repeats=n, axis=-1).reshape((t, n, 1))
            index_n = jnp.repeat(jnp.arange(0, n).reshape((1, n)), repeats=t, axis=0).reshape((t, n, 1))
            index_action = jnp.concatenate((index_t, index_n, his_action_to_scan), axis=-1)  # [n, 3]
            ratio = jax.lax.gather(ratio, start_indices=index_action,
                                   dimension_numbers=jax.lax.GatherDimensionNumbers(offset_dims=(2,),
                                                                                    collapsed_slice_dims=(0, 1),
                                                                                    start_index_map=(0, 1, 2)),
                                   slice_sizes=(1, 1, 1))  # [t, n, 1]  # fixme: how to use gather in jax? maybe bugs
            # debug variables ------------------------------------
            # # if n_train == 0:  # fixme
            # #     assert ((ratio < 1 + 1e-3) & (ratio > 1 - 1e-3)).float().mean() > 0.999, (ratio.max(), ratio.min())
            approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
            # -----------------------------------------------------
            advantage = his_traced_rewards_to_scan - his_values_to_scan  # [t, n, 1]
            surr1 = ratio * advantage
            surr2 = jnp.clip(ratio, 1.0 - clip_param,
                             1.0 + clip_param) * advantage
            action_loss = -jnp.minimum(surr1, surr2).mean()
            entropy_loss = - (jax.nn.log_softmax(policy_logits, axis=-1) * jax.nn.softmax(policy_logits, axis=-1)).sum(
                axis=-1).mean()
            value_loss = ((value - his_traced_rewards_to_scan) ** 2).mean()

            loss = action_loss - entropy_loss * entropy_coef + 0.5 * value_loss

            return loss, action_loss, entropy_loss, value_loss, approx_kl

        scan_policy_logits = scan_policy_logits[:-1]
        scan_value = scan_value[:-1]
        pfc_losses, action_losses, entropy_losses, value_losses, approx_kls = \
            jax.vmap(PPO_loss, (1, 1, 0), 0)(scan_policy_logits, scan_value, jnp.arange(policy_scan_len))
        pfc_loss = pfc_losses.mean()
        action_loss = action_losses.mean()
        entropy_loss = entropy_losses.mean()
        value_loss = value_losses.mean()
        approx_kl = approx_kls.mean()
        hf_loss = hippo_losses.mean()
        loss_pathint = losses_pathint.mean()
        loss_r = losses_r.mean()
        acc_pred = acc_preds.mean()
        loss = pfc_loss + hf_loss
        return loss, (pfc_loss, action_loss, entropy_loss, value_loss, approx_kl, hf_loss, loss_pathint, loss_r, acc_pred)

    for _ in range(n_train_time):
        ###FIXME
        key, subkey = jax.random.split(key)
        grad_fn = jax.value_and_grad(partial(loss_fn, batch=batch, key=key), has_aux=True, argnums=(0, 1, 2))
        (loss, (pfc_loss, action_loss, entropy_loss, value_loss, approx_kl, hf_loss, loss_pathint, loss_r, acc_pred)), \
            (policy_grad, hippo_grad, encoder_grad) = grad_fn(policy_state.params, hippo_state.params, encoder_state.params)
        # print('encoder_grad:',encoder_grad)
        clip_fn = lambda z: z / jnp.maximum(jnp.linalg.norm(z.flatten(), ord=2), 1)  # fixme: clip by value / by grad

        # jax.debug.breakpoint()
        jax.debug.print('grad_{a}_{b}_{c}', a=jnp.linalg.norm(policy_grad['Dense_0']['kernel'], ord=2),
                                        b=jnp.linalg.norm(hippo_grad['GRUCell_0']['hz']['kernel'], ord=2),
                                        c=jnp.linalg.norm(encoder_grad['Conv_0']['kernel'].flatten(), ord=2))
        # policy_grad = jax.tree_util.tree_map(clip_fn, policy_grad)
        # hippo_grad = jax.tree_util.tree_map(clip_fn, hippo_grad)
        # encoder_grad = jax.tree_util.tree_map(clip_fn, encoder_grad)
        policy_state = policy_state.apply_gradients(grads=policy_grad)
        hippo_state = hippo_state.apply_gradients(grads=hippo_grad)
        encoder_state = encoder_state.apply_gradients(grads=encoder_grad)

    # compute metrics
    pfc_metric_updates = policy_state.metrics.single_from_model_output(
        loss=pfc_loss, action_loss=action_loss, entropy_loss=entropy_loss, value_loss=value_loss, approx_kl=approx_kl)
    policy_state = policy_state.replace(metrics=pfc_metric_updates)
    hippo_metric_updates = hippo_state.metrics.single_from_model_output(
        loss=hf_loss, loss_pathint=loss_pathint, loss_r=loss_r, acc_pred=acc_pred)
    hippo_state = hippo_state.replace(metrics=hippo_metric_updates)
    return policy_state, hippo_state, encoder_state


@struct.dataclass
class PFC_Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    action_loss: metrics.Average.from_output('action_loss')
    entropy_loss: metrics.Average.from_output('entropy_loss')
    value_loss: metrics.Average.from_output('value_loss')
    approx_kl: metrics.Average.from_output('approx_kl')
@struct.dataclass
class Hippo_Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    loss_pathint: metrics.Average.from_output('loss_pathint')
    loss_r: metrics.Average.from_output('loss_r')
    acc_pred: metrics.Average.from_output('acc_pred')
@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')

class PFC_TrainState(train_state.TrainState):
    # key: jax.random.KeyArray
    metrics: PFC_Metrics

class Hippo_TrainState(train_state.TrainState):
    metrics: Hippo_Metrics

class TrainState(train_state.TrainState):
    metrics: Metrics


def init_states(args, initkey):
    if args.prefix == None:
        raise ValueError('prefix is None')
    # print('theta:', args.theta_hidden_size)
    # print('visual_prob:', args.visual_prob)
    # print('policy_scan_len:', args.policy_scan_len)
    # obs, env_state = env.reset(args.width, args.height, args.n_agents, initkey)
    maze_list = jnp.load('maze_list.npy')
    wall_maze = jnp.repeat(maze_list[0].reshape(1,args.grid_size,args.grid_size,args.n_action), args.n_agents, axis=0)
    start_s, env_state = env.reset(initkey, args.grid_size, args.n_agents, env_state={'wall_maze': wall_maze})
    # print('n_train_time:', args.n_train_time)
    # print('replay_steps:', args.replay_steps)
    # print('total_epochs:', args.epochs)
    # print('entropy_coef:', args.entropy_coef)
    # print('drop_out_rate:', args.drop_out_rate)
    # print('clip_param:', args.clip_param)
    # print('weight_decay:', args.wd)
    print(args)
    # Load encoder =================================================================================
    key, subkey = jax.random.split(initkey)
    encoder = Encoder()
    s = jnp.zeros((args.n_agents, 2), dtype=jnp.int8)
    obs = jnp.zeros((args.n_agents, args.grid_size, args.grid_size, 5), dtype=jnp.int8)
    a = jnp.zeros((args.n_agents, 1), dtype=jnp.int8)
    params = encoder.init(subkey, obs, a)['params']

    encoder_state = TrainState.create(
        apply_fn=encoder.apply, params=params, tx=optax.adamw(args.encoder_lr, weight_decay=0.0),
        metrics=Metrics.empty())

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
    encoder_state = checkpoints.restore_checkpoint(ckpt_dir=load_encoder,
                                                            target=encoder_state)
    # Load Hippo ===========================================================================
    obs_embed, action_embed = encoder_state.apply_fn({'params': params}, obs, a)
    r = jnp.zeros((args.n_agents, 1))
    theta = jnp.zeros((args.n_agents, args.theta_hidden_size))
    oe_ae_r = jnp.concatenate((obs_embed, action_embed, r), axis=-1)
    key, subkey = jax.random.split(key)
    hippo = Hippo(output_size=args.grid_size**2 + 1 + args.hippo_mem_len,
                  hidden_size=args.hippo_hidden_size,
                  bottleneck_size=args.bottleneck_size)
    hippo_hidden = jnp.zeros((args.n_agents, args.hippo_hidden_size))
    params = hippo.init(subkey, hippo_hidden, theta, oe_ae_r)['params']

    hippo_state = Hippo_TrainState.create(
        apply_fn=hippo.apply, params=params, tx=optax.adamw(args.hippo_lr, weight_decay=0.0),
        metrics=Hippo_Metrics.empty())

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
    hippo_state = checkpoints.restore_checkpoint(ckpt_dir=load_hippo,
                                                            target=hippo_state)

    # Init policy state ===============================================================================
    policy = Policy(args.n_action, args.theta_hidden_size, args.bottleneck_size)
    key, subkey, dropout_key = jax.random.split(key, 3)
    params = policy.init(subkey, hippo_hidden, theta, oe_ae_r)['params']
    policy_state = PFC_TrainState.create(
        apply_fn=policy.apply, params=params, tx=optax.adamw(args.policy_lr, weight_decay=args.wd),
        metrics=PFC_Metrics.empty())

    load_policy = 'r'
    if args.load_policy is not None:
        load_policy = args.model_path + '/' + args.load_policy
    else:
        for filename in os.listdir(args.model_path):
            if filename.startswith(args.prefix + '_policy'):
                load_policy = args.model_path + '/' + filename
    if os.path.exists(load_policy):
        print('load policy from:', load_policy)
    else:
        print('path not exists:', load_policy)
        print('randomly initialize policy')
    policy_state = checkpoints.restore_checkpoint(ckpt_dir=load_policy,
                                                            target=policy_state)
    # ===============================================
    # [s_a_r, new_hippo_hidden, theta, next_a, policy, value, done, next_s]
    init_samples_for_buffer = [oe_ae_r, hippo_hidden, theta,
        # new_actions
        jnp.zeros((args.n_agents, 1), dtype=jnp.int8),
        # logits
        jnp.zeros((args.n_agents, args.n_action)),
        # values
        jnp.zeros((args.n_agents, 1)),
        # done
        jnp.zeros((args.n_agents, 1), dtype=jnp.int8),
        # obs
        jnp.zeros((args.n_agents, args.grid_size, args.grid_size, 5), dtype=jnp.int8),
        # prev actions
        jnp.zeros((args.n_agents, 1), dtype=jnp.int8),
        # new hippo_hidden
        hippo_hidden,
        # next_s
        jnp.zeros((args.n_agents, 2), dtype=jnp.int8),
        # 
    ]

    buffer_state = buffer.create_buffer_states(args.max_size, init_samples_for_buffer)
    return env_state, buffer_state, encoder_state, hippo_state, policy_state


@jax.jit
def replay_fn(hippo_and_theta, key, policy_params, hippo_state, policy_state, oe_ae_r):
    # to match the input/output stream of jax.lax.scan
    # and also need to calculate grad of policy_params
    hippo_hidden, theta = hippo_and_theta
    new_theta, (policy, value, hippo_info) = policy_state.apply_fn({'params': policy_params},
                                                                hippo_hidden, theta, oe_ae_r)
    new_hippo_hidden, output, theta_info = hippo_state.apply_fn({'params': hippo_state.params},
                                                    hippo_hidden, new_theta, oe_ae_r)
    # todo: resample
    # key, outside_hipp_info = key_and_hipp_info
    # key, subkey = jax.random.split(key)
    # noise = jax.random.truncated_normal(subkey, -2, 2, new_hippo_hidden_mean.shape) * jnp.exp(jnp.clip(new_hippo_hidden_std, -2, 0.5))
    # new_hippo_hidden = new_hippo_hidden_mean
    # jax.debug.print('std {a} {b} {c}', a=noise.std(), b=new_hippo_hidden_mean.std(), c=new_hippo_hidden.std())
    return (new_hippo_hidden, new_theta), (new_hippo_hidden, new_theta, output, hippo_info, theta_info)


@partial(jax.jit, static_argnames=['temperature', 'replay_steps'])
def model_env_step(states, key, s, a, hippo_hidden, theta, temperature, replay_steps):
    env_state, buffer_state, encoder_state, hippo_state, policy_state = states
    # Input: actions_t-1, h_t-1, theta_t-1,
    key, subkey = jax.random.split(key)
    next_s, rewards, done, env_state = env.step(subkey, env_state, s, a)
    # Mask obs ==========================================================================================
    key, subkey = jax.random.split(key)
    obs = env.get_obs(env_state, next_s) # [n, g, g, 5]
    assert obs.shape == (args.n_agents, args.grid_size, args.grid_size, 5)
    # mask = jax.random.uniform(subkey, (wall_loc.shape[0], 1, 1))
    # obs_incomplete = jnp.where(obs == 2, 0, obs)
    # obs_incomplete = jnp.where(mask < visual_prob, obs_incomplete, 0)
    # obs[n, h, w], actions[n, 1], rewards[n, 1]
    # Encode obs and a_t-1 ===============================================================================
    obs_embed, action_embed = encoder_state.apply_fn({'params': encoder_state.params}, obs, a)
    oe_ae_r = jnp.concatenate((obs_embed, action_embed, rewards), axis=-1) # include ot, at-1, rt-1
    # Update hippo_hidden ==================================================================================
    new_hippo_hidden, output, theta_info = hippo_state.apply_fn({'params': hippo_state.params},
                                               hippo_hidden, theta, oe_ae_r)
    # Replay, only when rewards > 0 ===============================================================
    replay_fn_to_scan = partial(replay_fn, policy_params=policy_state.params, 
                                hippo_state=hippo_state, policy_state=policy_state,
                                oe_ae_r=jnp.zeros((obs_embed.shape[0], obs_embed.shape[-1] + action_embed.shape[-1] + 1)))

    replay_keys = jax.random.split(key, replay_steps + 1)
    key = replay_keys[0]
    replay_keys = replay_keys[1:]
    # outside_hipp_info = jnp.zeros((replay_steps, n_agents, bottleneck_size))
    (replayed_hippo_hidden, replayed_theta), replayed_history = jax.lax.scan(replay_fn_to_scan,
                                                                             init=(new_hippo_hidden, theta),
                                                                             xs=replay_keys)
    # only pass one step information
    # replayed_theta = theta
    # replayed_hippo_hidden = new_hippo_hidden
    
    replayed_theta = jnp.where(rewards > 0, replayed_theta, theta)
    replayed_hippo_hidden = jnp.where(rewards > 0, replayed_hippo_hidden, jnp.zeros(replayed_hippo_hidden.shape))
    # replayed_hippo_hidden = jnp.zeros(replayed_hippo_hidden.shape)
    # Take action ==================================================================================

    ###改一下train_step
    # outside_hipp_info = jnp.zeros((n_agents, bottleneck_size))
    new_theta, (policy, value, hippo_info) = policy_state.apply_fn({'params': policy_state.params},
                                                          replayed_hippo_hidden, replayed_theta, oe_ae_r)
    key, subkey = jax.random.split(key)
    # new_actions = jnp.argmax(policy, axis=-1, keepdims=True)
    next_a = sample_from_policy(policy, subkey, temperature)
    # fixme: reset reward; consider the checkpoint logic of env
    buffer_state = buffer.put_to_buffer(buffer_state,
                                        [oe_ae_r, hippo_hidden, theta,
                                         next_a, policy, value, done, obs, a,
                                         new_hippo_hidden, next_s])
    # reset if out of time
    # reset_theta = jnp.where(done & jnp.isclose(rewards, 0.), jnp.zeros(new_theta.shape), new_theta)
    # reset_hippo_hidden = jnp.where(done & jnp.isclose(rewards, 0.), jnp.zeros(new_hippo_hidden.shape), new_hippo_hidden)
    return env_state, buffer_state, new_hippo_hidden, new_theta, next_s, next_a, rewards, done, replayed_history

@partial(jax.jit, static_argnames=['temperature', 'replay_steps'])
def eval_step(states, key, s, a, hippo_hidden, theta, temperature, replay_steps):
    env_state, buffer_state, encoder_state, hippo_state, policy_state = states
    # Input: actions_t-1, h_t-1, theta_t-1,
    key, subkey = jax.random.split(key)
    next_s, rewards, done, env_state = env.step(subkey, env_state, s, a)
    # Mask obs ==========================================================================================
    key, subkey = jax.random.split(key)
    obs = env.get_obs(env_state, next_s) # [n, g, g, 5]
    assert obs.shape == (args.n_agents, args.grid_size, args.grid_size, 5)
    # mask = jax.random.uniform(subkey, (wall_loc.shape[0], 1, 1))
    # obs_incomplete = jnp.where(obs == 2, 0, obs)
    # obs_incomplete = jnp.where(mask < visual_prob, obs_incomplete, 0)
    # obs[n, h, w], actions[n, 1], rewards[n, 1]
    # Encode obs and a_t-1 ===============================================================================
    obs_embed, action_embed = encoder_state.apply_fn({'params': encoder_state.params}, obs, a)
    oe_ae_r = jnp.concatenate((obs_embed, action_embed, rewards), axis=-1) # include ot, at-1, rt-1
    # Update hippo_hidden ==================================================================================
    new_hippo_hidden, output, theta_info = hippo_state.apply_fn({'params': hippo_state.params},
                                               hippo_hidden, theta, oe_ae_r)
    # Replay, only when rewards > 0 ===============================================================
    replay_fn_to_scan = partial(replay_fn, policy_params=policy_state.params, 
                                hippo_state=hippo_state, policy_state=policy_state,
                                oe_ae_r=jnp.zeros((obs_embed.shape[0], obs_embed.shape[-1] + action_embed.shape[-1] + 1)))

    replay_keys = jax.random.split(key, replay_steps + 1)
    key = replay_keys[0]
    replay_keys = replay_keys[1:]
    # outside_hipp_info = jnp.zeros((replay_steps, n_agents, bottleneck_size))
    (replayed_hippo_hidden, replayed_theta), replayed_history = jax.lax.scan(replay_fn_to_scan,
                                                                             init=(new_hippo_hidden, theta),
                                                                             xs=replay_keys)
    # only pass one step information
    # replayed_theta = theta
    # replayed_hippo_hidden = new_hippo_hidden
    
    replayed_theta = jnp.where(rewards > 0, replayed_theta, theta)
    replayed_hippo_hidden = jnp.where(rewards > 0, replayed_hippo_hidden, jnp.zeros(replayed_hippo_hidden.shape))
    # replayed_hippo_hidden = jnp.zeros(replayed_hippo_hidden.shape)
    # Take action ==================================================================================

    ###改一下train_step
    # outside_hipp_info = jnp.zeros((n_agents, bottleneck_size))
    new_theta, (policy, value, hippo_info) = policy_state.apply_fn({'params': policy_state.params},
                                                          replayed_hippo_hidden, replayed_theta, oe_ae_r)
    key, subkey = jax.random.split(key)
    # new_actions = jnp.argmax(policy, axis=-1, keepdims=True)
    next_a = sample_from_policy(policy, subkey, temperature)
    # fixme: reset reward; consider the checkpoint logic of env
    buffer_state = buffer.put_to_buffer(buffer_state,
                                        [oe_ae_r, hippo_hidden, theta,
                                         next_a, policy, value, done, obs, a,
                                         new_hippo_hidden, next_s])
    # reset if out of time
    # reset_theta = jnp.where(done & jnp.isclose(rewards, 0.), jnp.zeros(new_theta.shape), new_theta)
    # reset_hippo_hidden = jnp.where(done & jnp.isclose(rewards, 0.), jnp.zeros(new_hippo_hidden.shape), new_hippo_hidden)
    return env_state, buffer_state, new_hippo_hidden, new_theta, next_s, next_a, rewards, done, replayed_history

def eval(states, key, args):
    '''
    num_cells = height * width
    replay_path = [[[] for n in range(args.n_agents)],[[] for n in range(args.n_agents)]]
    hist_state = []
    '''
    buffer_state, encoder_state, hippo_state, policy_state = states
    all_rewards = []
    key, subkey = jax.random.split(key)
    maze_list = jnp.load('maze_list.npy')
    wall_maze = jnp.repeat(maze_list[0].reshape(1,args.grid_size,args.grid_size,args.n_action), args.n_agents, axis=0)
    start_s, env_state = env.reset(subkey, args.grid_size, args.n_agents, env_state={'wall_maze': wall_maze})
    key, subkey = jax.random.split(key)
    s = env_state['start_s']
    a = jax.random.randint(subkey, (args.n_agents, 1), 0, args.n_action)
    hippo_hidden = jnp.zeros((args.n_agents, args.hippo_hidden_size))
    theta = jnp.zeros((args.n_agents, args.theta_hidden_size))
    for _ in range(args.n_eval_steps):
        key, subkey = jax.random.split(subkey)
        env_state, buffer_state, hippo_hidden, theta, next_s, next_a, rewards, done, replayed_history \
            = eval_step((env_state, buffer_state, encoder_state, hippo_state, policy_state),
                         subkey, s, a, hippo_hidden, theta, temperature=args.eval_temperature, replay_steps=args.replay_steps)
        all_rewards.append(rewards.mean().item())
    return sum(all_rewards) / len(all_rewards)


@partial(jax.jit, static_argnums=(2,))
def trace_back(rewards, done, gamma):
    # rewards [l, n, 1], carry, y = f(carry, x)
    t, n, _ = rewards.shape

    def trace_a_step(v, r_done):
        # v[n, 1], r[n, 1]
        r, done = r_done
        v_prime = v * gamma + r
        v_prime = jnp.where(done, r, v_prime)
        return v_prime, v_prime

    def trace_gamma(gn, xs):
        # gn [1, 1]
        gn_prime = gn * gamma + 1
        return gn_prime, gn_prime

    _, all_v = jax.lax.scan(trace_a_step, jnp.zeros((n, 1)), (jnp.flip(rewards, axis=0), jnp.flip(done, axis=0)))
    # _, exp_gamma = jax.lax.scan(trace_gamma, jnp.zeros((1, 1)), xs=None, length=t)
    # all_v = jnp.flip(all_v, axis=0) / jnp.flip(exp_gamma, axis=0)
    all_v = jnp.flip(all_v, axis=0)
    return all_v

def place_cell_centers(grid_size):
    centers = jnp.array([[i, j] for i in range(grid_size) for j in range(grid_size)])
    return centers


def main(args):
    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key)
    env_state, buffer_state, encoder_state, hippo_state, policy_state = init_states(args, subkey)
    writer = SummaryWriter(f"./train_logs/{args.prefix}")
    # Initialize actions, hippo_hidden, and theta ==================
    s = env_state['start_s']
    a = jax.random.randint(subkey, (args.n_agents, 1), 0, args.n_action)
    hippo_hidden = jnp.zeros((args.n_agents, args.hippo_hidden_size))
    theta = jnp.zeros((args.n_agents, args.theta_hidden_size))

    
    if args.model_path[2:] not in os.listdir():
        os.mkdir(args.model_path)
    for ei in range(args.epochs):
        # walk in the env and update buffer (model_step)
        key, subkey = jax.random.split(subkey)
        env_state, buffer_state, hippo_hidden, theta, next_s, next_a, rewards, done, replayed_history \
            = model_env_step((env_state, buffer_state, encoder_state, hippo_state, policy_state),
                         subkey, s, a, hippo_hidden, theta, temperature=1, replay_steps=args.replay_steps)
        # jax.debug.print('ei:{a},n:{b},rewards:{c},hippo:{d},theta:{e}',
        #                 a=ei, b=2, c=rewards[2], d=hippo_hidden[2], e=theta[2])
        s = next_s
        a = next_a
        if ei % 10 == 0:
            writer.add_scalar('train_reward', rewards.mean().item(), ei + 1)

        if ei % args.train_every == args.train_every - 1 and ei > args.max_size:
            # train for a step and empty buffer
            print(ei + 1)
            # for _ in range(args.n_train_time):
            key, subkey = jax.random.split(key)
            batch = buffer.sample_from_buffer(buffer_state, args.sample_len, subkey)
            # batch['his_rewards'] = batch['his_rewards'] - 0.05
            # batch['his_rewards'] = jnp.where(batch['step_count']==100, batch['his_rewards']-1, batch['his_rewards'])
            rewards = batch['oe_ae_r'][..., -1:]
            batch['his_traced_rewards'] = trace_back(rewards, batch['done'], args.gamma)
            key, subkey = jax.random.split(key)
            policy_state, hippo_state, encoder_state = train_step((encoder_state, hippo_state, policy_state),
                                                                      batch, args.replay_steps, args.clip_param, args.entropy_coef, args.n_train_time,
                                                                      args.policy_scan_len, subkey, args.hippo_pred_len, args.hippo_mem_len, args.grid_size, 
                                                                      place_cell_centers(args.grid_size), args.pc_sigma)
            # buffer_state = buffer.clear_buffer(buffer_state)
        if ei % args.eval_every == args.eval_every - 1 and ei > args.max_size:
            print('train_rewards:', rewards.mean().item())
            key, subkey = jax.random.split(key)
            eval_rewards = eval((buffer_state, encoder_state, hippo_state, policy_state), subkey, args)
            writer.add_scalar(f'eval_reward', eval_rewards, ei + 1)
            print('eval_rewards:', eval_rewards)
            for k, v in policy_state.metrics.compute().items():
                print(k, v.item())
                writer.add_scalar(f'pfc_{k}', v.item(), ei + 1)
            for k,v in hippo_state.metrics.compute().items():
                print(k, v.item())
                writer.add_scalar(f'hf_{k}', v.item(), ei + 1)
            checkpoints.save_checkpoint(args.model_path, policy_state, ei + 1, prefix=args.prefix+'_policy_', overwrite=True)

if __name__ == '__main__':
    args = parse_args()
    main(args)
