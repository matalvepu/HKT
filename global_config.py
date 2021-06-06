import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"

DEVICE = torch.device("cuda:0")

visual_features_list=list(range(55,91))
acoustic_features_list=list(range(0,60))

ACOUSTIC_DIM = len(acoustic_features_list)
VISUAL_DIM = len(visual_features_list)
HCF_DIM=4
LANGUAGE_DIM=768

VISUAL_DIM_ALL = 91
ACOUSTIC_DIM_ALL = 81

H_MERGE_SENT = 768
DATASET_LOCATION = "./dataset/"
SEP_TOKEN_ID = 3