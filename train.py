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
import cal_plot
import pickle

def parse_args():
    # args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--train_every', type=int, default=400)
    parser.add_argument('--n_agents', type=int, default=128)
    parser.add_argument('--max_size', type=int, default=400)  # max_size of buffer
    parser.add_argument('--sample_len', type=int, default=400)  # sample len from buffer: at most max_size - 1
    parser.add_argument('--epochs', type=int, default=int(1e6))

    parser.add_argument('--log_name', type=str, default='train_log')
    parser.add_argument('--model_path', type=str, default='./modelzoo')
    parser.add_argument('--replay_path', type=str, default='./replayzoo')
    parser.add_argument('--no_save', action='store_true', default=False)

    parser.add_argument('--mid_reward', type=float, default=0.5)
    parser.add_argument('--replay_steps', type=int, default=4)  # todo: tune

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
    parser.add_argument('--reset_prob', type=float, default=0.1)
    parser.add_argument('--block_idx', type=int, default=-1)
    parser.add_argument('--load_encoder', type=str)  # todo: checkpoint
    parser.add_argument('--load_hippo', type=str)
    parser.add_argument('--load_policy', type=str)
    parser.add_argument('--load_hippo_std', type=str, default='hippo_std_1000000')

    parser.add_argument('--prefix', type=str)

    # Init params for hippo and policy
    parser.add_argument('--bottleneck_size', type=int, default=4)
    parser.add_argument('--policy_scan_len', type=int, default=397)
    parser.add_argument('--theta_hidden_size', type=int)
    parser.add_argument('--theta_fast_size', type=int)
    parser.add_argument('--drop_out_rate', type=float, default=0)
    parser.add_argument('--noise_scale', type=float, default=1)
    parser.add_argument('--hippo_mem_len', type=int, default=5)

    parser.add_argument('--eval_temperature', type=float, default=0.5)
    args = parser.parse_args()
    args.pseudo_reward = env.pseudo_reward_list[1]
    return args


@jax.jit
def sample_from_policy(logit, key, temperature):
    # logit [n, 4]
    # return action [n, 1]
    # jax.debug.print('logit:{a}', a=logit)
    # jax.debug.print('prob:{b}', b=jax.nn.softmax(logit / temperature, axis=-1))
    def sample_once(logit_n, subkey):
        # logit_n[4,]
        action_n = jax.random.choice(subkey, jnp.arange(0, logit_n.shape[-1]), shape=(1,),
                                     p=jax.nn.softmax(logit_n / temperature, axis=-1))
        return action_n

    subkeys = jax.random.split(key, num=logit.shape[0] + 1)
    key, subkeys = subkeys[0], subkeys[1:]
    # subkeys = jnp.stack(subkeys, axis=0)
    action = jax.vmap(sample_once, (0, 0), 0)(logit, subkeys).astype(jnp.int8)
    return action


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13))
def train_step(state, batch, sample_len, n_agents, hidden_size, theta_hidden_size, \
               replay_steps, clip_param, entropy_coef, n_train_time, policy_scan_len, key, noise_scale, bottleneck_size):
    """Train for a single step with rollouts from the buffer, update policy_state only"""
    # state (encoder_state, hippo_state, policy_state)
    # obs_emb_t[t, n, d], action_emb_t-1, h_t[t, n, h] (before replay), theta_t[t, n, h] (before replay)，
    # rewards_t[t, n, 1], action_t[t, n, 1], policy_t[t, n, 4], value_t[t, n, 1]
    # traced_rewards_t[t, n, 1]: exp avg of rewards_t (MC sample of true value)
    # all rollouts start from where with rewards or reset, and first obs is not zero (optional)
    # his means that the data is from buffer (history)
    # index:
    # o_e(t), a_e(t-1), h(t), theta(t), r(t-1), a(t), logit(t), v(t), done(t), pos(t)

    encoder_state, hippo_state, hippo_std_state, policy_state = state
    jax.debug.print('replay ratio all {a}', a=jnp.where(batch['his_rewards'] > 0, 1, 0).mean())

    def loss_fn(policy_params, hippo_std_params, batch, key):
        # his_theta [l, n, h]
        # obs_embed [l, n, 64], action_embed [l, n, 4]
        # his_action [l, n, 1], his_logits [l, n, 4], his_rewards [l, n, 1], his_values[l, n, 1]
        # his_traced_rewards [l, n, 1]

        # 1. Replay: generate his_replayed_theta =======================================================
        # todo: Was here a bug? policy_params=policy_state.params
        replay_fn_to_scan = partial(replay_fn, policy_params=policy_params, hippo_std_params=hippo_std_params,
                                    hippo_state=hippo_state, hippo_std_state=hippo_std_state, policy_state=policy_state,
                                    n_agents=n_agents,
                                    obs_embed_size=batch['obs_embed'].shape[-1],
                                    action_embed_size=batch['action_embed'].shape[-1],
                                    training=True, noise_scale=noise_scale,
                                    bottleneck_size=bottleneck_size)

        def propagate_theta(key, index, policy_scan_len):
            indexes_for_propagate_once = jnp.arange(policy_scan_len) + index

            key_to_scan = jax.random.split(key, num=policy_scan_len + 1)
            key, key_to_scan = key_to_scan[0], key_to_scan[1:]
            obs_to_scan = jnp.take(batch['obs_embed'], indexes_for_propagate_once, axis=0)
            action_to_scan = jnp.take(batch['action_embed'], indexes_for_propagate_once, axis=0)
            hippo_hidden_to_scan = jnp.take(batch['his_hippo_hidden'], indexes_for_propagate_once, axis=0)
            rewards_to_scan = jnp.take(batch['his_rewards'], indexes_for_propagate_once, axis=0)
            pos_to_scan = jnp.take(batch['pos'], indexes_for_propagate_once, axis=0)
            done_to_scan = jnp.take(batch['done'], indexes_for_propagate_once, axis=0)

            # jax.debug.print('pos_{a},reward_{b}', a=pos_to_scan, b=rewards_to_scan)
            # jax.debug.breakpoint()
            # jax.debug.print('pos_{a},reward_{b}', a=pos_to_scan, b=rewards_to_scan)
            # jax.debug.breakpoint()
            new_theta, (_, hist_policy, hist_value) = jax.lax.scan(propagate_theta_once,
                                                                   init=jnp.take(batch['his_theta'], index, axis=0),
                                                                   xs=(key_to_scan, obs_to_scan, action_to_scan,
                                                                       hippo_hidden_to_scan,
                                                                       rewards_to_scan, pos_to_scan, done_to_scan))
            return new_theta, hist_policy, hist_value

        def propagate_theta_once(prev_theta, input_of_policy):
            # [n, th]
            # [n, 1], [n, h]
            key, obs_embed, prev_action_embed, hippo_hidden, rewards, current_pos, done = input_of_policy
            replay_keys = jax.random.split(key, replay_steps + 1)
            key, replay_keys = replay_keys[0], replay_keys[1:]
            outside_hipp_info = jnp.zeros((replay_steps, n_agents, bottleneck_size))
            (replayed_hippo_hidden, replayed_theta), _ = jax.lax.scan(replay_fn_to_scan,
                                                                      init=(hippo_hidden, prev_theta),
                                                                      xs=(replay_keys, outside_hipp_info))
            # replayed_theta = prev_theta
            # replayed_hippo_hidden = hippo_hidden
            replayed_theta = jnp.where(rewards > 0, replayed_theta, prev_theta)
            replayed_hippo_hidden = jnp.where(rewards > 0, replayed_hippo_hidden,
                                              jnp.zeros(replayed_hippo_hidden.shape))
            # replayed_hippo_hidden = jnp.zeros(replayed_hippo_hidden.shape)
            key, dropout_key = jax.random.split(key)
            outside_hipp_info = jnp.zeros((n_agents, bottleneck_size))
            new_theta, (policy, value, _, _) = policy_state.apply_fn({'params': policy_params},
                                                                  replayed_theta, obs_embed,
                                                                  prev_action_embed,
                                                                  replayed_hippo_hidden,
                                                                  noise_key=dropout_key,
                                                                  outside_hipp_info=outside_hipp_info,
                                                                  rngs={'dropout': dropout_key})
            reset_theta = jnp.where(done & jnp.isclose(rewards, 0.), 0, new_theta)

            return reset_theta, (reset_theta, policy, value)

        # _, his_replayed_theta = jax.lax.scan(propagate_theta, init=batch['his_theta'][0], xs=(batch['his_rewards'],
        #                                                                                       his_replayed_theta))
        # theta(t) = f(r(t-1), theta(t))
        # propagate theta
        # his_replayed_theta = jnp.where(batch['his_rewards'] > 0, his_replayed_theta, batch['his_theta'])
        # using scan instead of jnp.where to ensure that gradients of each action should affect the invariable theta 
        # generated by the same previous replay 
        propagate_theta_for_seq = partial(propagate_theta, policy_scan_len=policy_scan_len)
        # sample_len - 1 - policy_scan_len
        key_to_propagate = jax.random.split(key, num=sample_len - policy_scan_len + 1)
        key, key_to_propagate = key_to_propagate[0], key_to_propagate[1:]
        his_replayed_theta, scan_policy_logits, scan_value = jax.vmap(propagate_theta_for_seq, (0, 0), 0)(
            key_to_propagate, jnp.arange(sample_len - policy_scan_len))
        # [l, policy_scan_len, n, h], [l, policy_scan_len, n, 4], [l, policy_scan_len, n, 1]

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
        his_logits = batch['his_logits'][:-1]
        his_action = batch['his_action'][:-1]
        his_values = batch['his_values'][:-1]
        his_traced_rewards = batch['his_traced_rewards'][1:]
        # jax.debug.print('pos:{a}',a=batch['pos'][:-1])
        # jax.debug.print('his_action:{a}',a=his_action)
        # jax.debug.print('his_logits:{a}',a=his_logits)
        # jax.debug.print('his_values:{a}',a=his_values)
        # jax.debug.print('his_traces_rewards:{a}',a=his_traced_rewards)
        # jax.debug.breakpoint()
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
        losses, action_losses, entropy_losses, value_losses, approx_kls = \
            jax.vmap(PPO_loss, (1, 1, 0), 0)(scan_policy_logits, scan_value, jnp.arange(policy_scan_len))
        loss = losses.mean()
        action_loss = action_losses.mean()
        entropy_loss = entropy_losses.mean()
        value_loss = value_losses.mean()
        approx_kl = approx_kls.mean()
        return loss, (action_loss, entropy_loss, value_loss, approx_kl)

    for _ in range(n_train_time):
        ###FIXME
        key, subkey = jax.random.split(key)
        grad_fn = jax.value_and_grad(partial(loss_fn, batch=batch, key=key), has_aux=True, argnums=(0, 1))
        (loss, (action_loss, entropy_loss, value_loss, approx_kl)), (policy_grad, hippo_std_grad) = grad_fn(policy_state.params, hippo_std_state.params)

        clip_fn = lambda z: z / jnp.maximum(jnp.linalg.norm(z, ord=2), 0.5) * 0.5  # fixme: clip by value / by grad
        # jax.debug.breakpoint()
        jax.debug.print('grad_{a}_{b}', a=jnp.linalg.norm(policy_grad['Dense_0']['kernel'], ord=2),
                                        b=jnp.linalg.norm(hippo_std_grad['GRUCell_0']['hz']['kernel'], ord=2))
        policy_grad = jax.tree_util.tree_map(clip_fn, policy_grad)
        hippo_std_grad = jax.tree_util.tree_map(clip_fn, hippo_std_grad)
        policy_state = policy_state.apply_gradients(grads=policy_grad)
        hippo_std_state = hippo_std_state.apply_gradients(grads=hippo_std_grad)

    # compute metrics
    metric_updates = policy_state.metrics.single_from_model_output(
        loss=loss, action_loss=action_loss, entropy_loss=entropy_loss, value_loss=value_loss, approx_kl=approx_kl)
    policy_state = policy_state.replace(metrics=metric_updates)

    return policy_state, hippo_std_state


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    action_loss: metrics.Average.from_output('action_loss')
    entropy_loss: metrics.Average.from_output('entropy_loss')
    value_loss: metrics.Average.from_output('value_loss')
    approx_kl: metrics.Average.from_output('approx_kl')


class TrainState(train_state.TrainState):
    key: jax.random.KeyArray
    metrics: Metrics


def init_states(args, initkey):
    if args.prefix == None:
        raise ValueError('prefix is None')
    print('theta:', args.theta_hidden_size)
    print('visual_prob:', args.visual_prob)
    print('policy_scan_len:', args.policy_scan_len)
    obs, env_state = env.reset(args.width, args.height, args.n_agents, initkey)
    n_reward = jnp.sum(jnp.where(env_state['grid'] == 2, 1, 0)) // args.n_agents
    print('n_reward:', n_reward)
    print('n_train_time:', args.n_train_time)
    print('replay_steps:', args.replay_steps)
    print('total_epochs:', args.epochs)
    print('entropy_coef:', args.entropy_coef)
    print('drop_out_rate:', args.drop_out_rate)
    print('clip_param:', args.clip_param)
    print('weight_decay:', args.wd)
    print(args)
    # Load encoder =================================================================================
    key, subkey = jax.random.split(initkey)
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
    # Init std of hippo ==============================================================================
    key, subkey = jax.random.split(key)
    hippo_std = Hippo(output_size=args.height * args.width + 1 + args.hippo_mem_len,
                      hidden_size=args.hidden_size)
    hidden = jnp.zeros((args.n_agents, args.hidden_size))
    pfc_input = jnp.zeros((args.n_agents, 8))
    params = hippo.init(subkey, hidden, pfc_input, (obs_embed, action_embed), jnp.zeros((args.n_agents, 1)))['params']
    jax.debug.print('init {a}', a=list(params['GRUCell_0']['hz'].keys()))

    running_hippo_std_state = path_int.TrainState.create(
        apply_fn=hippo_std.apply, params=params, tx=optax.adamw(args.lr, weight_decay=args.wd),
        metrics=path_int.Metrics.empty())
    print('randomly initialize hippo std')
    # make sure the load is successful
    # Init policy state ===============================================================================
    policy = Policy(args.n_action, args.theta_hidden_size, args.theta_fast_size, args.bottleneck_size,
                    args.drop_out_rate)
    key, subkey, dropout_key = jax.random.split(key, 3)
    theta = jnp.zeros((args.n_agents, args.theta_hidden_size))
    # coordinate = jnp.zeros((args.n_agents, 2))
    # obs_complete = jnp.concatenate((coordinate, env_state['reward_center']),axis=-1)
    outside_hipp_info = jnp.zeros((args.n_agents, args.bottleneck_size))
    params = policy.init(subkey, theta, obs_embed, action_embed, hidden, dropout_key, outside_hipp_info)['params']
    running_policy_state = TrainState.create(
        apply_fn=policy.apply, params=params, tx=optax.adamw(args.lr, weight_decay=args.wd),
        metrics=Metrics.empty(), key=dropout_key)

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
    running_policy_state = checkpoints.restore_checkpoint(ckpt_dir=load_policy,
                                                            target=running_policy_state)
    # ===============================================
    init_samples_for_buffer = [
        obs_embed, action_embed, hidden, theta,
        # rewards
        jnp.zeros((args.n_agents, 1)),
        # new_actions
        jnp.zeros((args.n_agents, 1), dtype=jnp.int8),
        # logits
        jnp.zeros((args.n_agents, args.n_action)),
        # values
        jnp.zeros((args.n_agents, 1)),
        # done
        jnp.zeros((args.n_agents, 1), dtype=jnp.int8),
        # current_pos
        jnp.zeros((args.n_agents, 2), dtype=jnp.int8),
        # step_count
        jnp.zeros((args.n_agents, 1), dtype=jnp.int8)
    ]

    buffer_state = buffer.create_buffer_states(args.max_size, init_samples_for_buffer)
    return env_state, buffer_state, running_encoder_state, running_hippo_state, running_hippo_std_state, running_policy_state


@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12))
def replay_fn(hippo_and_theta, key_and_hipp_info, policy_params, hippo_std_params, hippo_state, hippo_std_state, policy_state,
              n_agents, obs_embed_size, action_embed_size, training, noise_scale, bottleneck_size):
    # to match the input/output stream of jax.lax.scan
    # and also need to calculate grad of policy_params
    hippo_hidden, theta = hippo_and_theta
    key, outside_hipp_info = key_and_hipp_info
    new_theta, (policy, value, to_hipp, hipp_info) = policy_state.apply_fn({'params': policy_params},
                                                                theta, jnp.zeros((n_agents, obs_embed_size)),
                                                                jnp.zeros((n_agents, action_embed_size)),
                                                                hippo_hidden, noise_key=key,
                                                                outside_hipp_info = outside_hipp_info,
                                                                rngs={'dropout': key})
    # jax.debug.print('{a}',a=to_hipp)
    new_hippo_hidden_mean, output = hippo_state.apply_fn({'params': hippo_state.params},
                                                    hippo_hidden, to_hipp,
                                                    (jnp.zeros((n_agents, obs_embed_size)),
                                                     jnp.zeros((n_agents, action_embed_size))),
                                                    jnp.zeros((n_agents, 1)))
    new_hippo_hidden_std, _ = hippo_std_state.apply_fn({'params': hippo_std_params},
                                                    hippo_hidden, to_hipp,
                                                    (jnp.zeros((n_agents, obs_embed_size)),
                                                     jnp.zeros((n_agents, action_embed_size))),
                                                    jnp.zeros((n_agents, 1)))
    # todo: resample
    key, subkey = jax.random.split(key)
    new_hippo_hidden = new_hippo_hidden_mean + jax.random.normal(subkey, new_hippo_hidden_mean.shape) * noise_scale #* nn.sigmoid(new_hippo_hidden_std)
    # jax.debug.print('std {a} {b}', a=nn.sigmoid(new_hippo_hidden_std).mean(), b=new_hippo_hidden_mean.std())
    # jax.debug.print('std {a} {b}', a=nn.sigmoid(new_hippo_hidden_std).mean(), b=new_hippo_hidden_mean.std())
    # noise = jax.random.truncated_normal(subkey, -2, 2, new_hippo_hidden_mean.shape) * jnp.exp(jnp.clip(new_hippo_hidden_std, -2, 0.5))
    # new_hippo_hidden = new_hippo_hidden_mean
    # jax.debug.print('std {a} {b} {c}', a=noise.std(), b=new_hippo_hidden_mean.std(), c=new_hippo_hidden.std())
    return (new_hippo_hidden, new_theta), (new_hippo_hidden, new_theta, output, hipp_info)


'''@partial(jax.jit, static_argnums=(5, 6, 7))
def replay_fn_for_seq(hippo_and_theta, xs, policy_params, hippo_state, policy_state,
                      n_agents, obs_embed_size, action_embed_size):
    # the only difference with replay_fn is that this func replays for whole seq [l, ..., ...]
    # to match the input/output stream of jax.lax.scan (the first dimension is sample length)
    # and also need to calculate grad of policy_params
    # In fact, h(t-1)+theta(t-1)->theta(t)+to_hippo(t); to_hippo(t)+h(t-1)->h(t)
    hippo_hidden, theta = hippo_and_theta
    new_theta, (policy, value, to_hipp) = policy_state.apply_fn({'params': policy_params},
                                                                theta, jnp.zeros((hippo_hidden.shape[0], n_agents, obs_embed_size)),
                                                                jnp.zeros((hippo_hidden.shape[0], n_agents, action_embed_size)),
                                                                hippo_hidden)
    new_hippo_hidden, _ = hippo_state.apply_fn({'params': hippo_state.params},
                                               hippo_hidden, to_hipp,
                                               (jnp.zeros((hippo_hidden.shape[0], n_agents, obs_embed_size)),
                                                jnp.zeros((hippo_hidden.shape[0], n_agents, action_embed_size))),
                                               jnp.zeros((hippo_hidden.shape[0], n_agents, 1)))
    return (new_hippo_hidden, new_theta), (new_hippo_hidden, new_theta)'''


@partial(jax.jit, static_argnums=(10, 11, 12, 13, 14, 15, 16, 17, 18))
def model_step(env_state, buffer_state, encoder_state, hippo_state, hippo_std_state, policy_state,
               key, actions, hippo_hidden, theta,
               n_agents, bottleneck_size, hidden_size, replay_steps, height, width, visual_prob, temperature,
               noise_scale):
    # Input: actions_t-1, h_t-1, theta_t-1,
    obs, rewards, done, env_state = env.step(env_state, actions)  # todo: reset
    # obs_embed_for_hippo, _ = encoder_state.apply_fn({'params': encoder_state.params}, obs, actions)
    # o(t), r(t-1) = f(o(t-1),a(t-1))
    key, subkey = jax.random.split(key)
    env_state = env.reset_reward(env_state, rewards, subkey)  # fixme: reset reward with 0.9 prob
    # Mask obs ==========================================================================================
    key, subkey = jax.random.split(key)
    mask = jax.random.uniform(subkey, (obs.shape[0], 1, 1))
    obs_incomplete = jnp.where(obs == 2, 0, obs)
    obs_incomplete = jnp.where(mask < visual_prob, obs_incomplete, 0)
    # obs[n, h, w], actions[n, 1], rewards[n, 1]
    # Encode obs and a_t-1 ===============================================================================
    obs_embed, action_embed = encoder_state.apply_fn({'params': encoder_state.params}, obs_incomplete, actions)
    # Update hippo_hidden ==================================================================================
    new_hippo_hidden, _ = hippo_state.apply_fn({'params': hippo_state.params},
                                               hippo_hidden, jnp.zeros((n_agents, 8)),
                                               (obs_embed, action_embed), rewards)
    # Replay, only when rewards > 0 ===============================================================
    replay_fn_to_scan = partial(replay_fn, policy_params=policy_state.params, hippo_std_params=hippo_std_state.params,
                                hippo_state=hippo_state, hippo_std_state=hippo_std_state, policy_state=policy_state,
                                n_agents=n_agents,
                                obs_embed_size=obs_embed.shape[-1], action_embed_size=action_embed.shape[-1],
                                training=True, noise_scale=noise_scale, bottleneck_size=bottleneck_size)

    replay_keys = jax.random.split(key, replay_steps + 1)
    key = replay_keys[0]
    replay_keys = replay_keys[1:]
    outside_hipp_info = jnp.zeros((replay_steps, n_agents, bottleneck_size))
    #实际上这里不能用,输入缺少outside_hipp_info
    (replayed_hippo_hidden, replayed_theta), replayed_history = jax.lax.scan(replay_fn_to_scan,
                                                                             init=(new_hippo_hidden, theta),
                                                                             xs=(replay_keys, outside_hipp_info))
    # replayed_hippo_hidden = jnp.where(rewards > 0, replayed_hippo_hidden, new_hippo_hidden)
    # fixme: not save replayed_hippo_hidden

    # only pass one step information
    # replayed_theta = theta
    # replayed_hippo_hidden = new_hippo_hidden
    
    replayed_theta = jnp.where(rewards > 0, replayed_theta, theta)
    replayed_hippo_hidden = jnp.where(rewards > 0, replayed_hippo_hidden, jnp.zeros(replayed_hippo_hidden.shape))
    # replayed_hippo_hidden = jnp.zeros(replayed_hippo_hidden.shape)
    # Take action ==================================================================================
    obs_complete = jnp.concatenate((env_state['current_pos'], env_state['reward_center']), axis=-1)

    ###改一下train_step
    key, dropout_key = jax.random.split(key)
    outside_hipp_info = jnp.zeros((n_agents, bottleneck_size))
    new_theta, (policy, value, to_hipp, hipp_info) = policy_state.apply_fn({'params': policy_state.params},
                                                          replayed_theta, obs_embed, action_embed,
                                                          replayed_hippo_hidden,
                                                          noise_key=dropout_key,
                                                          outside_hipp_info = outside_hipp_info,
                                                            rngs={'dropout': dropout_key})
    key, subkey = jax.random.split(key)
    # new_actions = jnp.argmax(policy, axis=-1, keepdims=True)
    new_actions = sample_from_policy(policy, subkey, temperature)
    # fixme: reset reward; consider the checkpoint logic of env
    buffer_state = buffer.put_to_buffer(buffer_state,
                                        [obs_embed, action_embed, new_hippo_hidden, theta,
                                         rewards, new_actions, policy, value, done,
                                         env_state['current_pos'], env_state['step_count']])
    # put to Buffer:
    # obs_emb_t, action_emb_t-1, h_t (before replay), theta_t (before replay)，
    # rewards_t-1, action_t, policy_t, value_t
    # jax.debug.print('obs{a}_actions_{b}_theta_{c}_rewards_{d}_newa_{e}',
    #                 a=env_state['current_pos'][0], b=actions[0], c=theta.mean(), d=rewards[0],
    #                 e=new_actions[0])
    # jax.debug.breakpoint()
    reset_theta = jnp.where(done & jnp.isclose(rewards, 0.), jnp.zeros(new_theta.shape), new_theta)
    reset_hippo_hidden = jnp.where(done & jnp.isclose(rewards, 0.), jnp.zeros(new_hippo_hidden.shape), new_hippo_hidden)
    return env_state, buffer_state, new_actions, reset_hippo_hidden, reset_theta, rewards, done, replayed_history
    # return action_t, h_t, theta_t (after replay), rewards_t-1 (for logging)


def Igata_eval(buffer_state, running_encoder_state, running_hippo_state, running_hippo_std_state, running_policy_state,
         key,
         n_agents, bottleneck_size, theta_hidden_size, replay_steps, height, width,
         visual_prob, n_eval_steps, hidden_size, reset_prob, noise_scale, temperature, pseudo_reward,
         block_idx):
    '''
    num_cells = height * width
    replay_path = [[[] for n in range(args.n_agents)],[[] for n in range(args.n_agents)]]
    hist_state = []
    '''

    all_rewards = []
    key, subkey = jax.random.split(key)
    _, env_state = env.pseudo_reset(width, height, n_agents, subkey, pseudo_reward)
    env_state['checked_times'] = jnp.zeros_like(env_state['checked'])
    actions = jax.random.randint(subkey, (args.n_agents, 1), 0, args.n_action)
    hippo_hidden = jnp.zeros((n_agents, hidden_size))
    theta = jnp.zeros((n_agents, theta_hidden_size))
    for _ in range(n_eval_steps):
        key, subkey = jax.random.split(subkey)

        '''        
        st = env_state['current_pos']
        at = actions
        '''

        env_state, buffer_state, actions, hippo_hidden, theta, rewards, done, replayed_history, first_hippo_theta_output, hipp_info, value, theta \
            = Igata_step(env_state, buffer_state, running_encoder_state, running_hippo_state, running_hippo_std_state, running_policy_state,
                        subkey, actions, hippo_hidden, theta,
                        n_agents, bottleneck_size, hidden_size, replay_steps, height, width,
                        visual_prob, temperature, reset_prob, noise_scale,
                        pseudo_reward, block_idx)
        all_rewards.append(rewards.mean().item())
    '''        
        rt = rewards  
        hist_state.append(jnp.concatenate((st,at,rt,done),1))
        replayed_hippo_history, replayed_theta_history, replayed_output_history = replayed_history
        for n in range(n_agents):
            if jnp.abs(rewards[n]-0.5)<1e-3:
                replay_path[0][n].append(replayed_output_history[:,n,:num_cells])
            elif jnp.abs(rewards[n]-1.0)<1e-3:
                replay_path[1][n].append(replayed_output_history[:,n,:num_cells])
        
    return replay_path, jnp.stack(hist_state, 0), sum(all_rewards)/len(all_rewards)
    '''

    return sum(all_rewards) / len(all_rewards)
    
@partial(jax.jit, static_argnums=(10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
def Igata_step(env_state, buffer_state, encoder_state, hippo_state, hippo_std_state, policy_state,
              key, actions, hippo_hidden, theta,
              n_agents, bottleneck_size, hidden_size, replay_steps, height, width,
              visual_prob, temperature, reset_prob, noise_scale, pseudo_reward,
              block_idx):
    obs, rewards, done, env_state = env.step(env_state, actions)
    env_state = env.Igata_reset_reward(env_state, rewards, key, n_agents,
                                        pseudo_reward)  # fixme: reset reward with 0.9 prob
    # Mask obs ==========================================================================================
    key, subkey = jax.random.split(key)
    mask = jax.random.uniform(subkey, (obs.shape[0], 1, 1))
    obs_incomplete = jnp.where(obs == 2, 0, obs)
    obs_incomplete = jnp.where(mask < visual_prob, obs_incomplete, 0)
    # jax.debug.print('obs_{a}', a=jnp.where(obs_incomplete==2,1,0).sum())
    # Encode obs and a_t-1 ===============================================================================
    obs_embed, action_embed = encoder_state.apply_fn({'params': encoder_state.params}, obs_incomplete, actions)
    # Update hippo_hidden ==================================================================================
    new_hippo_hidden, first_output = hippo_state.apply_fn({'params': hippo_state.params},
                                               hippo_hidden, jnp.zeros((n_agents, 8)),
                                               (obs_embed, action_embed), rewards)
    first_hippo_theta_output = (new_hippo_hidden, theta, first_output)
    # Replay, only when rewards > 0 ===============================================================
    replay_fn_to_scan = partial(replay_fn, policy_params=policy_state.params, hippo_std_params=hippo_std_state.params,
                                hippo_state=hippo_state, hippo_std_state=hippo_std_state, policy_state=policy_state,
                                n_agents=n_agents,
                                obs_embed_size=obs_embed.shape[-1], action_embed_size=action_embed.shape[-1],
                                training=False, noise_scale=noise_scale, bottleneck_size=bottleneck_size)
    replay_keys = jax.random.split(key, replay_steps + 1)
    key, replay_keys = replay_keys[0], replay_keys[1:]
    # n_mask_steps = 1

    ### Mask specified steps during replay
    # replay_keys = jnp.concatenate((jnp.zeros_like(replay_keys[:4]),
    #                                 replay_keys[4:5]
    #                                 ),0)
    ###

    replay_keys = jnp.zeros_like(replay_keys)
    outside_hipp_info = jnp.zeros((replay_steps, n_agents, bottleneck_size))
    # jax.debug.print('start:{a}',a=replay_keys)
    (replayed_hippo_hidden, replayed_theta), replayed_history = jax.lax.scan(replay_fn_to_scan,
                                                                             init=(new_hippo_hidden, theta),
                                                                             xs=(replay_keys, outside_hipp_info))
    replayed_hippo_history, replayed_theta_history, output_history, hipp_info_history = replayed_history
    

    replay_traj = jnp.argmax(output_history[...,:width*height], axis=-1).transpose(1,0)
    # jax.debug.print('replay_traj_{a}',a=replay_traj.shape)
    # n, 4
    cons_plan_condition = cal_plot.select_consolidation_plannning(width, height, replay_traj, env_state['phase'])
    blocked = jnp.where(((block_idx == 0) & ((cons_plan_condition[0] | cons_plan_condition[1]))) \
                | ((block_idx == 1) & ((cons_plan_condition[2] | cons_plan_condition[3]))),
                jnp.ones((n_agents,), dtype=jnp.int8), jnp.zeros((n_agents,), dtype=jnp.int8))
    
    blocked = blocked.reshape(blocked.shape[0],1)
    # blocked = jnp.zeros((n_agents,1), dtype=jnp.int8)
    # jax.debug.print('blocked_{a},{b}', a=blocked.shape,b=blocked)

    # #cancel out replay
    # replayed_theta = theta
    # replayed_hippo_hidden = new_hippo_hidden


    
    replayed_theta = jnp.where((rewards > 0), replayed_theta, theta)
    replayed_hippo_hidden = jnp.where((rewards > 0), replayed_hippo_hidden, jnp.zeros(new_hippo_hidden.shape))
    # replayed_hippo_hidden = jnp.zeros(new_hippo_hidden.shape)
    # Take action ==================================================================================
    # obs_complete = jnp.concatenate((env_state['current_pos'], env_state['reward_center']),axis=-1)
    key, dropout_key = jax.random.split(key)
    new_theta, (policy, value, _, hipp_info) = policy_state.apply_fn({'params': policy_state.params},
                                                          replayed_theta, obs_embed, action_embed,
                                                          replayed_hippo_hidden,
                                                          noise_key=dropout_key,
                                                          outside_hipp_info=jnp.zeros_like(outside_hipp_info.at[0].get()))

    ### change the order of hipp info and generate replay once again
     
    # key, subkey = jax.random.split(key)
    # new_replay_order = jax.random.permutation(key, replay_steps)
    # duplicate all info
    # new_replay_order = jnp.zeros_like(new_replay_order) + 3
    # shuffled_hipp_info_history = jnp.stack([hipp_info]*replay_steps, axis=0)
    # jax.debug.print('origin_info:{a}, shuffled_info:{b}',
    #                 a=hipp_info_history[:,0,:], b=shuffled_hipp_info_history[:,0,:])
    # (replayed_hippo_hidden, replayed_theta), replayed_history = jax.lax.scan(replay_fn_to_scan,
    #                                                                          init=(new_hippo_hidden, theta),
    #                                                                          xs=(replay_keys, shuffled_hipp_info_history))
    ###

    # replayed_theta = jnp.where((rewards > 0), replayed_theta, theta)
    # replayed_hippo_hidden = jnp.where((rewards > 0), replayed_hippo_hidden, jnp.zeros(new_hippo_hidden.shape))
    # Take action ==================================================================================
    # obs_complete = jnp.concatenate((env_state['current_pos'], env_state['reward_center']),axis=-1)
    # key, dropout_key = jax.random.split(key)
    # new_theta, (policy, value, _, hipp_info) = policy_state.apply_fn({'params': policy_state.params},
    #                                                       replayed_theta, obs_embed, action_embed,
    #                                                       replayed_hippo_hidden,
    #                                                       noise_key=dropout_key,
    #                                                       outside_hipp_info=shuffled_hipp_info_history.at[0].get())




    key, subkey = jax.random.split(key)
    new_actions = sample_from_policy(policy, subkey, temperature)

    reset_theta = jnp.where(done & jnp.isclose(rewards, 0.), jnp.zeros(new_theta.shape), new_theta)
    reset_hippo_hidden = jnp.where(done & jnp.isclose(rewards, 0.), jnp.zeros(new_hippo_hidden.shape), new_hippo_hidden)

    return env_state, buffer_state, new_actions, reset_hippo_hidden, reset_theta, rewards, done, \
        replayed_history, first_hippo_theta_output, hipp_info, value, policy

@partial(jax.jit, static_argnums=(10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
def eval_step(env_state, buffer_state, encoder_state, hippo_state, hippo_std_state, policy_state,
              key, actions, hippo_hidden, theta,
              n_agents, bottleneck_size, hidden_size, replay_steps, height, width,
              visual_prob, temperature, reset_prob, noise_scale, pseudo_reward,
              block_idx):
    obs, rewards, done, env_state = env.step(env_state, actions)
    env_state = env.pseudo_reset_reward(env_state, rewards, key, n_agents,
                                        reset_prob, pseudo_reward)  # fixme: reset reward with 0.9 prob
    # Mask obs ==========================================================================================
    key, subkey = jax.random.split(key)
    mask = jax.random.uniform(subkey, (obs.shape[0], 1, 1))
    obs_incomplete = jnp.where(obs == 2, 0, obs)
    obs_incomplete = jnp.where(mask < visual_prob, obs_incomplete, 0)
    # Encode obs and a_t-1 ===============================================================================
    obs_embed, action_embed = encoder_state.apply_fn({'params': encoder_state.params}, obs_incomplete, actions)
    # Update hippo_hidden ==================================================================================
    new_hippo_hidden, first_output = hippo_state.apply_fn({'params': hippo_state.params},
                                               hippo_hidden, jnp.zeros((n_agents, 8)),
                                               (obs_embed, action_embed), rewards)
    first_hippo_theta_output = (new_hippo_hidden, theta, first_output)
    # Replay, only when rewards > 0 ===============================================================
    replay_fn_to_scan = partial(replay_fn, policy_params=policy_state.params, hippo_std_params=hippo_std_state.params,
                                hippo_state=hippo_state, hippo_std_state=hippo_std_state, policy_state=policy_state,
                                n_agents=n_agents,
                                obs_embed_size=obs_embed.shape[-1], action_embed_size=action_embed.shape[-1],
                                training=False, noise_scale=noise_scale, bottleneck_size=bottleneck_size)
    replay_keys = jax.random.split(key, replay_steps + 1)
    key, replay_keys = replay_keys[0], replay_keys[1:]
    outside_hipp_info = jnp.zeros((replay_steps, n_agents, bottleneck_size))
    (replayed_hippo_hidden, replayed_theta), replayed_history = jax.lax.scan(replay_fn_to_scan,
                                                                             init=(new_hippo_hidden, theta),
                                                                             xs=(replay_keys, outside_hipp_info))
    replayed_hippo_history, replayed_theta_history, output_history, hipp_info_history = replayed_history
    replay_traj = jnp.argmax(output_history[...,:width*height], axis=-1).transpose(1,0)
    # jax.debug.print('replay_traj_{a}',a=replay_traj.shape)
    # n, 4
    cons_plan_condition = cal_plot.select_consolidation_plannning(width, height, replay_traj, env_state['phase'])
    blocked = jnp.where(((block_idx == 0) & ((cons_plan_condition[0] | cons_plan_condition[1]))) \
                | ((block_idx == 1) & ((cons_plan_condition[2] | cons_plan_condition[3]))),
                jnp.ones((n_agents,), dtype=jnp.int8), jnp.zeros((n_agents,), dtype=jnp.int8))
    
    blocked = blocked.reshape(blocked.shape[0],1)
    # blocked = jnp.zeros((n_agents,1), dtype=jnp.int8)
    # jax.debug.print('blocked_{a},{b}', a=blocked.shape,b=blocked)
    # replayed_theta = theta
    # replayed_hippo_hidden = new_hippo_hidden
    replayed_theta = jnp.where((rewards > 0), replayed_theta, theta)
    replayed_hippo_hidden = jnp.where((rewards > 0), replayed_hippo_hidden, jnp.zeros(new_hippo_hidden.shape))
    # replayed_hippo_hidden = jnp.zeros(new_hippo_hidden.shape)
    # replayed_theta = jnp.where((rewards > 0) & (~blocked), replayed_theta, theta)
    # replayed_hippo_hidden = jnp.where((rewards > 0) & (~blocked), replayed_hippo_hidden, jnp.zeros(new_hippo_hidden.shape))
    # replayed_theta_hippo = (replayed_theta, replayed_hippo_hidden)
    # Take action ==================================================================================
    # obs_complete = jnp.concatenate((env_state['current_pos'], env_state['reward_center']),axis=-1)
    key, dropout_key = jax.random.split(key)
    outside_hipp_info = jnp.zeros((n_agents, bottleneck_size))
    new_theta, (policy, value, _, hipp_info) = policy_state.apply_fn({'params': policy_state.params},
                                                          replayed_theta, obs_embed, action_embed,
                                                          replayed_hippo_hidden,
                                                          noise_key=dropout_key,
                                                          outside_hipp_info=outside_hipp_info)
    key, subkey = jax.random.split(key)
    new_actions = sample_from_policy(policy, subkey, temperature)

    reset_theta = jnp.where(done & jnp.isclose(rewards, 0.), jnp.zeros(new_theta.shape), new_theta)
    reset_hippo_hidden = jnp.where(done & jnp.isclose(rewards, 0.), jnp.zeros(new_hippo_hidden.shape), new_hippo_hidden)

    return env_state, buffer_state, new_actions, reset_hippo_hidden, reset_theta, rewards, done, \
        replayed_history, first_hippo_theta_output, hipp_info, value


def eval(buffer_state, running_encoder_state, running_hippo_state, running_hippo_std_state, running_policy_state,
         key,
         n_agents, bottleneck_size, theta_hidden_size, replay_steps, height, width,
         visual_prob, n_eval_steps, hidden_size, reset_prob, noise_scale, temperature, pseudo_reward,
         block_idx):
    '''
    num_cells = height * width
    replay_path = [[[] for n in range(args.n_agents)],[[] for n in range(args.n_agents)]]
    hist_state = []
    '''

    all_rewards = []
    key, subkey = jax.random.split(key)
    _, env_state = env.pseudo_reset(width, height, n_agents, subkey, pseudo_reward)
    actions = jax.random.randint(subkey, (args.n_agents, 1), 0, args.n_action)
    hippo_hidden = jnp.zeros((n_agents, hidden_size))
    theta = jnp.zeros((n_agents, theta_hidden_size))
    for _ in range(n_eval_steps):
        key, subkey = jax.random.split(subkey)

        '''        
        st = env_state['current_pos']
        at = actions
        '''

        env_state, buffer_state, actions, hippo_hidden, theta, rewards, done, replayed_history, first_hippo_theta_output, hipp_info, value \
            = eval_step(env_state, buffer_state, running_encoder_state, running_hippo_state, running_hippo_std_state, running_policy_state,
                        subkey, actions, hippo_hidden, theta,
                        n_agents, bottleneck_size, hidden_size, replay_steps, height, width,
                        visual_prob, temperature, reset_prob, noise_scale,
                        pseudo_reward, block_idx)
        all_rewards.append(rewards.mean().item())
    '''        
        rt = rewards  
        hist_state.append(jnp.concatenate((st,at,rt,done),1))
        replayed_hippo_history, replayed_theta_history, replayed_output_history = replayed_history
        for n in range(n_agents):
            if jnp.abs(rewards[n]-0.5)<1e-3:
                replay_path[0][n].append(replayed_output_history[:,n,:num_cells])
            elif jnp.abs(rewards[n]-1.0)<1e-3:
                replay_path[1][n].append(replayed_output_history[:,n,:num_cells])
        
    return replay_path, jnp.stack(hist_state, 0), sum(all_rewards)/len(all_rewards)
    '''

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


def main(args):
    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key)
    env_state, buffer_state, running_encoder_state, running_hippo_state, running_hippo_std_state, \
    running_policy_state = init_states(args, subkey)
    writer = SummaryWriter(f"./train_logs/{args.prefix}")
    # Initialize actions, hippo_hidden, and theta ==================
    actions = jax.random.randint(subkey, (args.n_agents, 1), 0, args.n_action)
    hippo_hidden = jnp.zeros((args.n_agents, args.hidden_size))
    theta = jnp.zeros((args.n_agents, args.theta_hidden_size))
    # ===============================================================
    # Initialize Sub Record State(the last dimension is rewards)
    # total_record = [[[] for n in range(args.n_agents)],[[] for n in range(args.n_agents)]]
    if args.model_path[2:] not in os.listdir():
        os.mkdir(args.model_path)
    # pickle.dump(args,open(args.model_path + '/' + args.prefix + '_args','wb'))
    for ei in range(args.epochs):
        # walk in the env and update buffer (model_step)
        key, subkey = jax.random.split(subkey)
        env_state, buffer_state, actions, hippo_hidden, theta, rewards, done, replayed_history \
            = model_step(env_state, buffer_state, running_encoder_state, running_hippo_state, running_hippo_std_state, running_policy_state,
                         subkey, actions, hippo_hidden, theta,
                         args.n_agents, args.bottleneck_size, args.hidden_size, args.replay_steps, args.height,
                         args.width,
                         args.visual_prob, temperature=1, noise_scale=args.noise_scale)
        # jax.debug.print('ei:{a},n:{b},rewards:{c},hippo:{d},theta:{e}',
        #                 a=ei, b=2, c=rewards[2], d=hippo_hidden[2], e=theta[2])
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
            batch['his_traced_rewards'] = trace_back(batch['his_rewards'], batch['done'], args.gamma)
            # jax.debug.breakpoint()
            key, subkey = jax.random.split(key)
            running_policy_state, running_hippo_std_state = train_step((running_encoder_state, running_hippo_state, running_hippo_std_state, running_policy_state),
                                                                      batch,
                                                                      args.sample_len, args.n_agents, args.hidden_size, args.theta_hidden_size,
                                                                      args.replay_steps,
                                                                      args.clip_param, args.entropy_coef, args.n_train_time,
                                                                      args.policy_scan_len, subkey, args.noise_scale, args.bottleneck_size)
            buffer_state = buffer.clear_buffer(buffer_state)
        if ei % args.eval_every == args.eval_every - 1 and ei > args.max_size:
            print('train_rewards:', rewards.mean().item())
            key, subkey = jax.random.split(key)
            '''replay_path, hist_state, eval_rewards = '''
            eval_rewards = \
                eval(buffer_state, running_encoder_state, running_hippo_state, running_hippo_std_state,
                     running_policy_state,
                     subkey,
                     args.n_agents, args.bottleneck_size, args.theta_hidden_size, args.replay_steps, args.height,
                     args.width,
                     args.visual_prob, args.n_eval_steps, args.hidden_size, args.reset_prob, args.noise_scale,
                     args.eval_temperature, args.pseudo_reward, args.block_idx)
            writer.add_scalar(f'eval_reward', eval_rewards, ei + 1)
            print('eval_rewards:', eval_rewards)
            for k, v in running_policy_state.metrics.compute().items():
                print(k, v.item())
                writer.add_scalar(f'train_{k}', v.item(), ei + 1)
            # save model
            if not args.no_save:
                print('save model to:'+args.prefix+'_policy_'+str(ei+1))
                checkpoints.save_checkpoint(args.model_path, running_policy_state, ei + 1, prefix=args.prefix+'_policy_', overwrite=True)
            '''
            print('shape of every mid buffer')
            for n in range(args.n_agents):
                # save record
                if len(replay_path[0][n]) > 0:
                    print(ei+1, 'agent:',n,' mid_total_record:',len(replay_path[0][n]))
                    total_record[0][n].append({'ei':ei+1, 'pc_path':jnp.array(replay_path[0][n]), 'state_traj':hist_state[:,n]})
                if len(replay_path[1][n]) > 0:
                    print(ei+1, 'agent:', n, ' goal_total_record', len(replay_path[1][n]))
                    total_record[1][n].append({'ei':ei+1, 'pc_path':jnp.array(replay_path[1][n]), 'state_traj':hist_state[:,n]})

    # save record
    if not args.no_save:
        save_record = [[],[]]
        print('save record to: r_policy_replay')
        for n in range(args.n_agents):
            print(len(total_record[0][n]),len(total_record[1][n]))
            if len(total_record[0][n]) > 0:
                save_record[0].append({'agent_th':n, 'reward_center':env_state['reward_center'][n], 'paths':total_record[0][n]})
            if len(total_record[1][n]) > 0:
                save_record[1].append({'agent_th':n, 'reward_center':env_state['reward_center'][n], 'paths':total_record[1][n]})

    # format: [[(reward_center, [(ei, place_map(t*hw)),(ei, place_map(t*hw)),...]),(reward_center,...)],
    #         [(reward_center, [(...)])]]
        checkpoints.save_checkpoint(args.replay_path, save_record, '_replay', prefix='r_policy', overwrite=True)
    # jax.debug.print('total_record_state_shape:mid_hippo_{a},mid_theta_{b},mid_place_cell_{c},mid_predict_r_{d}',
    #         a=total_record_state['buffer'][0].shape, b=total_record_state['buffer'][1].shape,
    #         c=total_record_state['buffer'][2].shape, d=total_record_state['buffer'][3].shape)
    '''


if __name__ == '__main__':
    args = parse_args()
    main(args)
