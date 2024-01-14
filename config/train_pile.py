# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'latte'
wandb_run_name=''

# model
n_layer = 16
n_head = 16
n_embd = 1024
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# these make the total batch size be ~1M
# 32 batch size * 2048 block size * 2 gradaccum * 8 GPUs = 1,048,576
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 2 * 8

# this makes total number of tokens be 300B
max_iters = 300000
lr_decay_iters = 300000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
