import os

# 'train'
# 'evaluation'
# 'experiment'
run_model='train'
num_of_train = 21
secondary_loss_weight = -1.5
alpha = 0.7

samples = 1000
batch_size = 8
epoch_num = 2
max_len_input = 400
max_len_output = 40
num_workers = 1
val_amount = 0.1
model_name = "t5-small"

data_path = os.path.join(os.getcwd(), 'data')
model_path = os.path.join(os.getcwd(), 'model')
checkpoint_path = os.path.join(model_path, 'checkpoints')
train_path = os.path.join(data_path, 'train_squad.parquet')
validation_path = os.path.join(data_path, 'validation_squad.parquet')
