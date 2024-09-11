import os

TRAIN_PATH_LR = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/train/lr"
TRAIN_PATH_HR = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/train/hr"
VAL_PATH_LR = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/val/lr"
VAL_PATH_HR = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/val/hr"
TEST_PATH_LR = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/celeba_hq_256/lr"
TEST_PATH_HR = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/celeba_hq_256/hr"

print("Train path lr length: ", len(os.listdir(TRAIN_PATH_LR)))
print("Train path hr length: ", len(os.listdir(TRAIN_PATH_HR)))
print("Val path lr length: ", len(os.listdir(VAL_PATH_LR)))
print("Val path hr length: ", len(os.listdir(VAL_PATH_HR)))
print("Test path lr length: ", len(os.listdir(TEST_PATH_LR)))
print("Test path hr length: ", len(os.listdir(TEST_PATH_HR)))

"""
python srgan.py --checkpoint_interval 1 --batch_size 32 --epoch 41
"""