"""
Buffer for train.py
"""
from functools import partial
import jax
from jax import numpy as jnp


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


@partial(jax.jit, static_argnums=(1,))
def sample_from_buffer(buffer_state, sample_len, key):
    indices = jax.random.randint(key, (sample_len,), 0, buffer_state['max_size'], dtype=jnp.int8)
    # indices = (jnp.arange(sample_len) + begin_index) % buffer_state['max_size']
    # [s_a_r, new_hippo_hidden, theta, next_a, policy, value, done, next_s]
    batch = [jnp.take(buffer_state['buffer'][xi], indices, axis=0) for xi in range(len(buffer_state['buffer']))]
    batch = {'oe_ae_r': batch[0], 'hippo_hidden': batch[1], 'theta': batch[2],
             'next_action': batch[3], 'logits': batch[4], 'values': batch[5],
             'done': batch[6], 'obs': batch[7], 'prev_action': batch[8],
             'new_hippo_hidden': batch[9], 'next_s': batch[10]}
    return batch


def clear_buffer(buffer_state):
    buffer = [jnp.zeros_like(buffer_state['buffer'][xi])
              for xi in range(len(buffer_state['buffer']))]
    insert_pos = 0
    buffer_state = {'buffer': buffer, 'insert_pos': jnp.array(insert_pos), 'max_size': buffer_state['max_size']}
    return buffer_state