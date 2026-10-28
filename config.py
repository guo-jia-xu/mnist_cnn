import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lr = 0.1
weight_decay = 1e-4
label_smoothing = 0.05
epochs = 4


train_batch_size = 128
train_shuffle = True

test_batch_size = 128
test_shuffle = False


mnist_dataset_path = "./datasets"

excel_path = "./实验的数据.xlsx"

checkpoints_path = "./checkpoints"