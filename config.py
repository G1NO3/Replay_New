"""
Config for path_int.py
"""

# env
width = 5
height = 5
n_agents = 2048  # batch_size
sigma = 1.  # the std of place cell
visual_prob = 0.2  # the prob that obs is not masked to zero

# network
hidden_size = 64  # hidden_size of hippo campus state
bottleneck_size = 8  # the dim of pfc's input to hippo
hippo_pred_len = 10
hippo_mem_len = 5
# optimizer
lr = 1e-4
wd = 1e-3
epoch = int(1e7)
train_every = 16
sample_len = 256

# buffer
max_size = 20000

# other
save_name = 'nohpc'
load = './modelzoo/hml_encoder/checkpoint_240000'
