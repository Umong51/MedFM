ckpt_dir = "checkpoints"

# MODEL CONFIG
image_size = (1, 128, 128)
in_channels = 2
base_channels = 32
out_channels = 1
channel_mult = [1, 2, 2, 4, 5]
attention_resolutions = [32, 16, 8]
num_res_blocks = 2

# TRAINING CONFIG
batch_size = 16
max_epochs = 500

# OPTIMIZER CONFIG
learning_rate = 1e-4
final_lr = 1e-6

# EMA CONFIG
ema_alpha = 0.9999

# EVALUATION CONFIG
val_max_size = 128
samples = 12
num_steps = 250
eval_freq = 100
