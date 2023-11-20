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

    ste_model_cover = 'cover'
    ste_model_LSB = 'LSB'
    ste_model_HOGO = 'HOGO'
    ste_model_WOW = 'WOW'
    ste_model_UNIWARD = 'UNIWARD'

    base_folder_cover = f'C:/Users/ns/Desktop/paksersht/Code Email/{ste_model_cover}/{payload}'
    base_folder_LSB = f'C:/Users/ns/Desktop/paksersht/Code Email/{ste_model_LSB}/{payload}'
    base_folder_HOGO = f'C:/Users/ns/Desktop/paksersht/Code Email/{ste_model_HOGO}/{payload}'
    base_folder_WOW = f'C:/Users/ns/Desktop/paksersht/Code Email/{ste_model_WOW}/{payload}'
    base_folder_UNIWARD = f'C:/Users/ns/Desktop/paksersht/Code Email/{ste_model_UNIWARD}/{payload}'

    file_list_cover = glob.glob(os.path.join(base_folder_cover, '*'))
    file_list_LSB = glob.glob(os.path.join(base_folder_LSB, '*'))
    file_list_HOGO = glob.glob(os.path.join(base_folder_HOGO, '*'))
    file_list_WOW = glob.glob(os.path.join(base_folder_WOW, '*'))
    file_list_UNIWARD = glob.glob(os.path.join(base_folder_UNIWARD, '*'))

    return file_list_cover,file_list_LSB,file_list_HOGO,file_list_WOW,file_list_UNIWARD
