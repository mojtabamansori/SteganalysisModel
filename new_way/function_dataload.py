import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score
import time
import os
import glob
from PIL import Image
from tqdm import tqdm



def dataload(payload):
    payload = payload

    ste_model_cover = 'cover - Copy'
    ste_model_LSB = 'LSB - Copy'
    ste_model_HOGO = 'HOGO - Copy'
    ste_model_WOW = 'WOW - Copy'
    ste_model_UNIWARD = 'UNIWARD - Copy'

    base_folder_cover = f'E:/code emil stephano/cover/0.4'
    # base_folder_LSB = f'E:/code emil stephano/LSB/0.4'
    # base_folder_HOGO = f'E:/code emil stephano/HOGO/0.4'
    base_folder_WOW = f'E:/code emil stephano/WOW/0.4'
    base_folder_UNIWARD = f'E:/code emil stephano/UNIWARD/0.4'

    file_list_cover = glob.glob(os.path.join(base_folder_cover, '*'))
    # file_list_LSB = glob.glob(os.path.join(base_folder_LSB, '*'))
    # file_list_HOGO = glob.glob(os.path.join(base_folder_HOGO, '*'))
    file_list_WOW = glob.glob(os.path.join(base_folder_WOW, '*'))
    file_list_UNIWARD = glob.glob(os.path.join(base_folder_UNIWARD, '*'))

    return file_list_cover, file_list_WOW, file_list_UNIWARD
    # return file_list_cover,file_list_LSB,file_list_HOGO,file_list_WOW,file_list_UNIWARD
