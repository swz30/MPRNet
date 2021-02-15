"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from MPRNet import MPRNet
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_deblurring.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = MPRNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

dataset = args.dataset
rgb_dir_test = os.path.join(args.input_dir, dataset, 'test', 'input')
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

result_dir  = os.path.join(args.result_dir, dataset)
utils.mkdir(result_dir)

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input_    = data_test[0].cuda()
        filenames = data_test[1]

        # Padding in case images are not multiples of 8
        if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
            factor = 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)
        restored = torch.clamp(restored[0],0,1)

        # Unpad images to original dimensions
        if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
            restored = restored[:,:,:h,:w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
