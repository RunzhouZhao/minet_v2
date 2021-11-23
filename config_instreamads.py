'''
config file
'''
n_user_slot = 1 # user fts
n_one_hot_slot_1 = 16 # only item fts; no user fts; id, main_cate, brand 
# title, categories
max_len_per_slot_1 = 3
num_csv_col_1 = 193 # num of cols in the csv file = 1 (label) + n_user_slot + (n_one_hot_slot_1 + n_mul_hot_slot_1*max_len_per_slot_1)*(1 + max_n_clk_ori_1) + (n_one_hot_slot_2 + n_mul_hot_slot_2*max_len_per_slot_2)*max_n_clk_ori_2
batch_size_1 = 1024
layer_dim_1 = [256, 128, 1]

max_n_clk_ori_1 = 3 # ori val in dataset
max_n_clk_ori_2 = 3 # in dataset 1
max_n_clk_1 = 3 # actual val used in experiment; max_n_clk_1 <= max_n_clk_ori_1
max_n_clk_2 = 3

# n_one_hot_slot_2 may be different from n_one_hot_slot_1
# because the 1st dataset removes redundant user features in the src domain data
n_one_hot_slot_2 = 10 # only item fts; no user fts
n_mul_hot_slot_2 = 6
max_len_per_slot_2 = 3
num_csv_col_2 = 25  # num of cols in the csv file = 1 (label) + n_user_slot + n_one_hot_slot_2 + n_mul_hot_slot_2*max_len_per_slot_2
batch_size_2 = 2*batch_size_1
layer_dim_2 = [256, 128, 1]

input_format = 'csv' # 'csv' or 'tfrecord'
pre = './data/Movies_Books_'
suf = '.csv' # '.csv'
# validation step: train: 'train', test: 'val', xx_range: a set of values
# testing step: train: 'train'+'val', test: 'test', xx_range: one optimal value found in the validation step
train_file_name_1 = ['t_train'+suf]
test_file_name_1 = ['t_test'+suf]

pre = './data/Books_'
suf = '.csv'
train_file_name_2 = [pre+'train'+suf, pre+'val'+suf]
test_file_name_2 = [pre+'test'+suf]

save_model_ind = 0

## for DNN
num_csv_col = 74
pre1='./data/Ads_Contents_'
train_file_name = [pre1+'train'+suf, pre1+'val'+suf]
test_file_name = [pre1+'test'+suf]
batch_size = 1024
n_one_hot_slot = 49
n_mul_hot_slot = 8
max_len_per_slot = 3

layer_dim = layer_dim_1

n_ft = 98589
time_style = '%Y-%m-%d %H:%M:%S'
rnd_seed = 123
user_b_ini = 1.0
tar_clk_b_ini = 0.0
src_clk_b_ini = 0.0
wgt_1 = 1.0
wgt_2_range = [1.0] # [0.1, 0.3, 0.5, 0.7, 1.0]
eta_range = [0.001] # [0.01, 0.02, 0.05, 0.1]
inter_dim = 10
k = 10 # embedding size / number of latent factors
opt_alg = 'Adam' #'Adagrad' 'Adam'
kp_prob = 1.0
output_file_name = '0728_1629' # for model saving
item_att_hidden_dim = 64
interest_att_hidden_dim = 64
n_epoch = 1 # number of times to loop over the whole data set
record_step_size = 100 # record the loss and auc after xx mini-batches

