 CUDA_VISIBLE_DEVICES=3 python train.py --prefix test_gae \
     --theta_hidden_size 32 --hippo_hidden_size 64 \
     --replay_steps 0 --bottleneck_size 4 --hippo_mem_len 0 --hippo_pred_len 1 \
     --reset_freq 200 --epochs 20000000


#python visualize.py --prefix reset_reward \
#    --theta_hidden_size 32 --hippo_hidden_size 64 \
#    --replay_steps 4 --bottleneck_size 4 --hippo_mem_len 0 --hippo_pred_len 1 \
#    --n_agents 32