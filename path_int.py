"""
Pretrain hippo module and encoder module with two task: 1. predict place cell; 2. predict reward
"""

import jax
import jax.numpy as jnp
from flax import linen as nn  # Linen API
from functools import partial
from clu import metrics
import flax.core
from flax.training import train_state, checkpoints  # Useful dataclass to keep train state
from flax import struct  # Flax dataclasses
from flax.traverse_util import flatten_dict, unflatten_dict
import optax  # Common loss functions and optimizers
from jax import xla_computation
from tensorboardX import SummaryWriter
import os
import env
from agent import Encoder, Hippo
import config


def create_place_cell_state(sigma, width, height):
    all_centers = []
    for i in range(height):
        for j in range(width):
            all_centers.append(jnp.array([i, j]))
    centers = jnp.stack(all_centers, axis=0)  # [m, 2]
    return {'sigma': jnp.array(sigma), 'centers': centers}


# @jax.jit
def generate_place_cell(centers, sigma, x):
    # x[n, 2], centers[m, 2]
    # @partial(jax.jit, static_argnums=(2,))
    @jax.jit
    def cal_dist(pos, cents, sigma):
        # pos[2,], cents[m, 2]
        return - ((pos.reshape((1, -1)) - cents) ** 2).sum(axis=-1) / (2 * sigma ** 2)  # [m,]
    # print(x.shape, centers.shape)
    activation = jax.vmap(cal_dist, (0, None, None), 0)(x, centers, sigma)  # [n, m]
    # print(activation)
    activation = nn.softmax(activation, axis=-1)
    return activation


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    # acc: metrics.Average.from_output('acc')
    loss_last: metrics.Average.from_output('loss_last')
    loss_pred: metrics.Average.from_output('loss_pred')
    recall_0: metrics.Average.from_output('recall_0')
    recall_r: metrics.Average.from_output('recall_r')
    recall_g: metrics.Average.from_output('recall_g')
    precision_0: metrics.Average.from_output('precision_0')
    precision_r: metrics.Average.from_output('precision_r')
    precision_g: metrics.Average.from_output('precision_g')
    f1_0: metrics.Average.from_output('f1_0')
    f1_r: metrics.Average.from_output('f1_r')
    f1_g: metrics.Average.from_output('f1_g')
    # acc_last: metrics.Average.from_output('acc_last')
    acc_pred: metrics.Average.from_output('acc_pred')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(encoder, hippo, rng, init_sample, config):
    """Creates an initial `TrainState`."""
    # Initialize encoder ==================================================================
    rng, sub_rng = jax.random.split(rng)
    params = encoder.init(sub_rng, *init_sample)['params']
    tx = optax.adamw(config.lr, weight_decay=config.wd)
    encoder_state = TrainState.create(
        apply_fn=encoder.apply, params=params, tx=tx,
        metrics=Metrics.empty())
    # Initialize hippo ====================================================================
    obs_embed, action_embed = encoder_state.apply_fn({'params': params}, *init_sample)
    hidden = jnp.zeros((config.n_agents, config.hidden_size))
    pfc_input = jnp.zeros((config.n_agents, config.bottleneck_size))
    rng, sub_rng = jax.random.split(rng)
    params = hippo.init(sub_rng, hidden, pfc_input, (obs_embed, action_embed), jnp.zeros((config.n_agents, 1)))[
        'params']
    tx = optax.adamw(config.lr, weight_decay=config.wd)
    hippo_state = TrainState.create(
        apply_fn=hippo.apply, params=params, tx=tx,
        metrics=Metrics.empty())
    return encoder_state, hippo_state

jnp.set_printoptions(precision=2,threshold=jnp.inf,suppress=True)
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def train_step(running_encoder_state, running_hippo_state, batch, sample_len, n_agent, hidden_size, bottleneck_size, hippo_pred_len, hippo_mem_len):
    """Train for a single step with rollouts from the buffer"""
    # state, obs(o_t)[t, n, h, w], actions(action_t-1)[t, n, 1], rewards(r_t-1)[t, n, 1], real_pos(s_t)[t, n, 2]
    # o_t, r_t-1, action_t-1 => s_t, r_t

    #  Initialize hidden
    hiddens = jnp.zeros((n_agent, hidden_size))

    def forward_fn(params_encoder, params_hippo, n_agent, bottleneck_size, hiddens, inputs):
        obs, action, rewards_prev, done = inputs
        obs_embed, action_embed = running_encoder_state.apply_fn({'params': params_encoder}, obs, action)
        pfc_input = jnp.zeros((n_agent, bottleneck_size))
        new_hidden, outputs = running_hippo_state.apply_fn({'params': params_hippo},
                                                           hiddens, pfc_input, (obs_embed, action_embed), rewards_prev)
        new_hidden = jnp.where(done&jnp.isclose(rewards_prev, 0), jnp.zeros_like(new_hidden), new_hidden)
        return new_hidden, outputs

    def loss_fn(params_encoder, params_hippo, hiddens, batch):

        len_t, n_agent, num_cells = batch['place_cells'].shape
        apply_fn = partial(forward_fn, params_encoder, params_hippo, n_agent, bottleneck_size)
        masked_rewards = batch['rewards'].at[-hippo_pred_len:, :, :].set(0)  # todo: mask and predict the last hpl rewards
        _, all_preds = jax.lax.scan(apply_fn, hiddens, [batch['obs'], batch['action'], masked_rewards, batch['done']])
        preds_place = all_preds[:, :, :num_cells]
        mem_preds_rewards = all_preds[:, :, num_cells:]  # [l, n, 1+hml]
        preds_rewards = mem_preds_rewards[-hippo_pred_len:,:,-1]  # [hpl, n]

        mem_rewards = mem_preds_rewards[-hippo_pred_len,:,:-1]  # [n, hml]
        mem_rewards = jnp.transpose(mem_rewards, (1, 0))  # [hml, n]

        loss_pred = optax.softmax_cross_entropy(preds_place, batch['place_cells'])[:-hippo_pred_len].mean()
        # exclude the last step of place_cell prediction
        # [:-1]: not to pred the last place cell because of the masked rewards
        acc_pred = (jnp.argmax(preds_place, axis=-1)
                    == jnp.argmax(batch['place_cells'], axis=-1)).astype(jnp.float32)[:-hippo_pred_len].mean()

        # pred last
        rewards_label = batch['rewards'][-(hippo_pred_len+hippo_mem_len):,:,0]  # [hpl+hml, n]  # todo: 1: :-1; only predict the last one
        # rewards_label = batch['rewards'][-(hippo_pred_len+hippo_mem_len):-hippo_pred_len,:,0]
        # state = jnp.concatenate((batch['action'], batch['current_pos'], batch['rewards'], preds_rewards, \
        #     batch['checked'], batch['step_count'], batch['reward_center'], jnp.argmax(batch['place_cells'],-1,keepdims=True), jnp.argmax(preds_place,-1,keepdims=True)),-1)
        # # jax.debug.breakpoint()

        # for n in range(config.n_agents):
        #     print(n)
        #     reward_index = jnp.array(jnp.where(jnp.isclose(batch['rewards'][:,n,:].reshape(-1),0.5))[0])
        #     print(batch['rewards'][:,n,:].reshape(-1))
        #     print(reward_index)
        #     # jax.debug.breakpoint()
        #     for i in range(len(reward_index)-1):  
        #         if jnp.sum(batch['rewards'][reward_index[i]+1:reward_index[i+1],n])<1:
        #             print(n)
        #             print(reward_index[i], reward_index[i+1])
                    
        #             print(state[reward_index[i]:reward_index[i+1]+1,n])

        
        jax.debug.print("reward_label_{shape}:{a}",shape=rewards_label.shape,a=batch['rewards'][-hippo_pred_len-hippo_mem_len:,0].reshape(-1))
        jax.debug.print("mem_rewards_{shape}:{a}",shape=mem_rewards.shape,a=mem_rewards[:,0].reshape(-1))
        jax.debug.print("preds_rewards_{shape}:{a}",shape=preds_rewards.shape,a=preds_rewards[:,0].reshape(-1))
        # st=0,1 at=2, rt=3, done=4, reward_center=5,6, step_count=7
        # state = jnp.concatenate((batch['current_pos'], batch['action'], batch['rewards'], \
        #     batch['done'], batch['reward_center'], batch['step_count'], batch['move_to_start']),axis=-1)
        # print(state.shape)
        # for n in range(n_agent):
        #     for i in range(sample_len-2):
        #         if jnp.isclose(batch['rewards'][i,n],1) and jnp.isclose(batch['rewards'][i+1,n],1) and jnp.isclose(batch['rewards'][i+2,n],1):
        #             jax.debug.print('state:{a}',a=state[i:i+3,n])
        #             raise ValueError('reward error')
        
        #     jax.debug.print("preds_place: {a}",a=preds_place[-1].reshape(config.n_agents,config.height,config.width)[0])
        #     jax.debug.print("place_cell: {a}",a=batch['place_cells'][-1].reshape(config.n_agents,config.height,config.width)[0])
        


        # jax.debug.breakpoint()
        all_r = jnp.concatenate((mem_rewards, preds_rewards),0)
        # all_r = mem_rewards
        loss_last = jnp.square(all_r - rewards_label)*5  # [hpl+hml, n]  # only consider the last hpl step
         # add weight to mem_len
        # jax.debug.print(str(loss_last))
        # jax.debug.breakpoint()
        
        recall_0 = jnp.where((loss_last < 0.1) & (jnp.abs(rewards_label - 0.) < 1e-3), 1, 0)  # todo: acc criterion: < 0.2
        recall_r = jnp.where((loss_last < 0.1) & (jnp.abs(rewards_label - 0.5) < 1e-3), 1, 0)
        recall_g = jnp.where((loss_last < 0.1) & (jnp.abs(rewards_label - 1.) < 1e-3), 1, 0)

        precision_0 = jnp.where((jnp.abs(all_r - 0.) < 0.1)&(jnp.abs(rewards_label - 0.) < 1e-3), 1, 0)
        precision_r = jnp.where((jnp.abs(all_r - 0.5) < 0.1)&(jnp.abs(rewards_label - 0.5) < 1e-3), 1, 0)
        precision_g = jnp.where((jnp.abs(all_r - 1.) < 0.1)&(jnp.abs(rewards_label - 1.) < 1e-3), 1, 0)
        loss_last = jnp.where(rewards_label > 0.4, loss_last * 40, loss_last).mean()  # todo: weighted loss, times by 10
        # consider_last_flag = (jnp.sum(
        #     jnp.where(jnp.abs(masked_rewards - 1) < 1e-3, 0, masked_rewards), axis=0) > 0.4
        #                       )  # [n, 1], todo: consider: 1. gets reward twice; or 2. get goal
        # jax.debug.print('{a}_{b}', a=(jnp.sum(
        #     jnp.where(jnp.abs(masked_rewards - 1) < 1e-3, 0, masked_rewards), axis=0) > 0.4
        #                               ).mean(), b=(rewards_label > 0.9).mean())
        # loss_last = jnp.where(consider_last_flag, loss_last, 0).mean()
        # acc_0 = jnp.where(consider_last_flag, acc_0, 0).sum() \
        #         / jnp.where((jnp.abs(rewards_label - 0.) < 1e-3) & consider_last_flag, 1, 0).sum()
        # acc_r = jnp.where(consider_last_flag, acc_r, 0).sum() \
        #         / jnp.where((jnp.abs(rewards_label - 0.5) < 1e-3) & consider_last_flag, 1, 0).sum()
        # acc_g = jnp.where(consider_last_flag, acc_g, 0).sum() \
        #         / jnp.where((jnp.abs(rewards_label - 1.) < 1e-3) & consider_last_flag, 1, 0).sum()
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
        
        loss = loss_pred + loss_last * 0.8  # todo: *0.1
        return loss, (loss_last, loss_pred, recall_0, recall_r, recall_g, \
            precision_0, precision_r, precision_g, f1_0, f1_r, f1_g, acc_pred)

    grad_fn = jax.value_and_grad(partial(loss_fn, hiddens=hiddens, batch=batch), has_aux=True, argnums=(0, 1))
    (loss, (loss_last, loss_pred, recall_0, recall_r, recall_g, precision_0, precision_r, precision_g,\
        f1_0, f1_r, f1_g, acc_pred)), (grads_encoder, grads_hippo) = grad_fn(
        running_encoder_state.params,
        running_hippo_state.params)
    running_encoder_state = running_encoder_state.apply_gradients(grads=grads_encoder)
    # fixme: clip by value / by grad ============================
    # clip_fn = lambda z: jnp.clip(z, -1.0, 1.0)
    # grads_hippo = jax.tree_util.tree_map(clip_fn, grads_hippo)
    # ----------------------------------------------------------
    clip_fn = lambda z: z / jnp.maximum(jnp.linalg.norm(z, ord=2), 5.0) * 5.0  # fixme: clip by value / by grad
    # jax.debug.print('grad_{a}', a=jnp.linalg.norm(grads_hippo[('Dense_0', 'kernel')], ord=2))
    grads_hippo = jax.tree_util.tree_map(clip_fn, grads_hippo)
    # ==========================================================
    running_hippo_state = running_hippo_state.apply_gradients(grads=grads_hippo)

    # compute metrics
    metric_updates = running_encoder_state.metrics.single_from_model_output(
        loss=loss, loss_last=loss_last, loss_pred=loss_pred, recall_0=recall_0, recall_r=recall_r, recall_g=recall_g,
        precision_0=precision_0, precision_r=precision_r, precision_g=precision_g, f1_0=f1_0, f1_r=f1_r, f1_g=f1_g,
        acc_pred=acc_pred)
    running_encoder_state = running_encoder_state.replace(metrics=metric_updates)
    return running_encoder_state, running_hippo_state


def create_buffer_states(max_size, init_sample):
    buffer = [jnp.zeros((max_size, *init_sample[i].shape), init_sample[i].dtype)
              for i in range(len(init_sample))]
    insert_pos = 0
    buffer_states = {'buffer': buffer, 'insert_pos': jnp.array(insert_pos), 'max_size': jnp.array(max_size)}
    return buffer_states


@jax.jit
def put_to_buffer(buffer_state, x):
    @jax.jit
    def insert(buffer, x, position):
        for xi in range(len(x)):
            buffer[xi] = buffer[xi].at[position].set(x[xi])
        return buffer

    buffer = insert(buffer_state['buffer'], x, buffer_state['insert_pos'])
    insert_pos = (buffer_state['insert_pos'] + 1) % buffer_state['max_size']
    return dict(buffer_state, buffer=buffer, insert_pos=insert_pos)


# @partial(jax.jit, static_argnums=(1,2))
# def sample_from_buffer(buffer_state, sample_len, n_agents, key):
#     # Not consider done
#     max_val = buffer_state['insert_pos'] - sample_len + buffer_state['max_size']
#     min_val = buffer_state['insert_pos']
#     ### 改成agent数目
#     begin_index = jax.random.randint(key, (n_agents,1), minval=min_val, maxval=max_val) % buffer_state['max_size']
#     # n_agent * sample_len
#     indices = (jnp.arange(sample_len).reshape(1,-1).repeat(n_agents,0) + begin_index.repeat(sample_len,axis=1)) % buffer_state['max_size']
#     # buffer: xi * buffer_size * n_agents * xi_specific_dimension
#     samples = [[] for _ in range(len(buffer_state['buffer']))]
#     for xi in range(len(buffer_state['buffer'])):
#         for agent_th in range(n_agents):
#             samples[xi].append(buffer_state['buffer'][xi][indices[agent_th], agent_th])
#         samples[xi] = jnp.stack(samples[xi], axis=1)
#     return samples

@partial(jax.jit, static_argnums=(1,2))
def sample_from_buffer(buffer_state, sample_len, n_agents, key):
    # Not consider done
    max_val = buffer_state['insert_pos'] - sample_len + buffer_state['max_size']
    min_val = buffer_state['insert_pos']
    begin_index = jax.random.randint(key, (1,n_agents), minval=min_val, maxval=max_val) % buffer_state['max_size']

    indices = (jnp.arange(sample_len).reshape(-1,1) + begin_index) % buffer_state['max_size']
    # jax.debug.print('indices:{a}', a=indices)
    # jax.debug.print('samples:{b}',b=buffer_state['buffer'][5].at[indices,jnp.arange(n_agents).reshape(1,-1)].get().reshape(sample_len,n_agents))
    return [buffer_state['buffer'][xi].at[indices,jnp.arange(n_agents).reshape(1,-1)].get() for xi in range(len(buffer_state['buffer']))]

    # return rollout


def prepare_batch(rollouts, place_cell_state):
    # obs [l, n, h, w], actions[l, n, 1], pos[l, n, 2], rewards[l, n, 1], checked[l, n, 1]
    # t = sample_len
    batch = dict()
    batch['obs'] = rollouts[0]
    batch['action'] = rollouts[1]
    batch['place_cells'] = jax.vmap(generate_place_cell, (None, None, 0), 0)(place_cell_state['centers'],
                                                                             place_cell_state['sigma'],
                                                                             rollouts[2])
    batch['current_pos'] = rollouts[2]
    batch['rewards'] = rollouts[3]
    batch['checked'] = rollouts[4]
    batch['step_count'] = rollouts[5]
    batch['reward_center'] = rollouts[6]
    batch['done'] = rollouts[7]
    batch['move_to_start'] = rollouts[8]
    return batch


@partial(jax.jit, static_argnums=(2, 3, 4))
def mask_obs(obs, key, sample_len, n_agent, visual_prob):
    # obs[t, n, h, w]
    # fixme: should by env; not mask the first step
    mask = jax.random.uniform(key, (sample_len, n_agent, 1, 1))
    mask = mask.at[0, :, :, :].set(0)
    obs = jnp.where(mask < visual_prob, obs, 0)
    obs = jnp.where(obs == 2, 0, obs)
    return obs


# @partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10, 11))
def a_loop(key, buffer_states, place_cell_state, running_encoder_state, running_hippo_state,
           sample_len, n_agents, visual_prob, hidden_size, bottleneck_size, hippo_pred_len,
           hippo_mem_len):
    # get from buffer, train_step()
    key, subkey = jax.random.split(key)
    rollouts = sample_from_buffer(buffer_states, sample_len, n_agents, subkey)
    # print(ei, len(rollouts))
    batch = prepare_batch(rollouts, place_cell_state)
    key, subkey = jax.random.split(key)
    batch['obs'] = mask_obs(batch['obs'], subkey,
                            sample_len, n_agents, visual_prob)
    # print(batch['place_cells'].reshape((-1, 100)).std(axis=0).mean(), 'place cell std')

    running_encoder_state, running_hippo_state = train_step(running_encoder_state, running_hippo_state, batch,
                                                            sample_len, n_agents, hidden_size, bottleneck_size,
                                                            hippo_pred_len, hippo_mem_len)
    return key, buffer_states, place_cell_state, running_encoder_state, running_hippo_state


def main(config):
    # Initialize logs ==============================================
    # metrics_history = {'train_loss': [], 'train_accuracy': []}
    writer = SummaryWriter(f'./logs/{config.save_name}')
    # Initialize key ================================================
    key = jax.random.PRNGKey(0)
    # Initialize env and place_cell ================================================
    key, subkey = jax.random.split(key)
    obs, env_state = env.reset(config.width, config.height, config.n_agents, subkey)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (config.n_agents, 1), minval=0, maxval=4)  # [n, 1]
    obs, rewards, done, env_state = env.step(env_state, actions)
    # place_cell = PlaceCell(config.sigma, config.width, config.height)
    place_cell_state = create_place_cell_state(config.sigma, config.width, config.height)
    # Initialize model and training_state ============================
    encoder = Encoder()
    hippo = Hippo(output_size=place_cell_state['centers'].shape[0] + 1 + config.hippo_mem_len,
                  hidden_size=config.hidden_size)
    key, subkey = jax.random.split(key)
    running_encoder_state, running_hippo_state = create_train_state(encoder, hippo, subkey,
                                                                    (obs, actions),
                                                                    config)
    if config.load != '':
        if os.path.exists(config.load):
            print('successfully load encoder from:', config.load)
        else:
            print('randomly initialize encoder')
        running_encoder_state = checkpoints.restore_checkpoint(ckpt_dir=config.load, target=running_encoder_state)
        if os.path.exists(config.load.replace('encoder', 'hippo')):
            print('successfully load hippo from:', config.load)
        else:
            print('randomly initialize hippo')  
        running_hippo_state = checkpoints.restore_checkpoint(ckpt_dir=config.load.replace('encoder', 'hippo'),
                                                             target=running_hippo_state)
    # Initialize buffer================================================
    buffer_states = create_buffer_states(max_size=config.max_size, init_sample=[obs, actions, env_state['current_pos'],
                                                                                rewards, env_state['checked'],
                                                                                env_state['step_count'],
                                                                                env_state['reward_center'],
                                                                                done,
                                                                                env_state['move_to_start']])

    for ei in range(config.epoch):
        key, subkey = jax.random.split(key)
        actions = jax.random.randint(subkey, (config.n_agents, 1), minval=0, maxval=4)  # [n, 1]
        obs, rewards, done, env_state = env.step(env_state, actions)
        # put_to_buffer: o_t, r_t-1, action_t-1, s_t
        buffer_states = put_to_buffer(buffer_states, [obs, actions, env_state['current_pos'],
                                                      rewards, env_state['checked'],
                                                      env_state['step_count'],
                                                      env_state['reward_center'],
                                                      done,
                                                      env_state['move_to_start']])
        # put to buffer [obs_t, a_t-1, pos_t, reward_t, checked_t]
        key, subkey = jax.random.split(key)
        env_state = env.reset_reward(env_state, rewards, subkey)  # fixme: reset reward

        if ei % config.train_every == config.train_every - 1 and ei > config.max_size:
            key, buffer_states, place_cell_state, running_encoder_state, running_hippo_state = \
                a_loop(key, buffer_states, place_cell_state, running_encoder_state, running_hippo_state,
                       sample_len=config.sample_len, n_agents=config.n_agents,
                       visual_prob=config.visual_prob, hidden_size=config.hidden_size,
                       bottleneck_size=config.bottleneck_size, hippo_pred_len=config.hippo_pred_len,
                       hippo_mem_len=config.hippo_mem_len)

        if ei % 1000 == 1000-1 and ei > config.max_size:
            for k, v in running_encoder_state.metrics.compute().items():
                print(ei+1, k, v.item())
                if jnp.isnan(v).item():
                    print(k, v)  # fixme
                else:
                    writer.add_scalar(f'train_{k}', v.item(), ei + 1) 
        if ei % 5000 == 5000-1 and ei > config.max_size:
            print(f'save encoder to: {config.save_name}_encoder'+f'/checkpoint_{ei+1}')
            checkpoints.save_checkpoint(f'./modelzoo/{config.save_name}_encoder', target=running_encoder_state, step=ei+1, overwrite=True)
            print(f'save hippo to: {config.save_name}_hippo'+f'/checkpoint_{ei+1}')
            checkpoints.save_checkpoint(f'./modelzoo/{config.save_name}_hippo', target=running_hippo_state, step=ei+1, overwrite=True)


if __name__ == '__main__':
    main(config)


### mask Reward 预测最后一个
### flag 只考虑同时有两个reward的情况
##### 交替训练？### Fine-tune? 后边再试 现在不太行
### VAE : 先试试能不能过拟合，以及滤波之后的低维

### 阻断replay后再进行学习
### 学习一些生物故事