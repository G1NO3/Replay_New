CUDA_VISIBLE_DEVICES=3 python train.py --prefix no_eval_buffer \
    --theta_hidden_size 32 --hippo_hidden_size 64\
    --replay_steps 4 --bottleneck_size 4 --hippo_mem_len 5 --hippo_pred_len 1 \
    --load_encoder r --load_hippo r --load_policy r
