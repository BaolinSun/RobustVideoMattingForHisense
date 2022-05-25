import os
import cv2
import shutil
import torch

import numpy as np
import pandas as pd


from skimage import measure
from tqdm import tqdm
from PIL import Image

from torchvision.transforms import Resize, Compose

import time

resize = Compose([Resize((720, 1280))])

color_masks = [
        np.random.randint(0, 256, (1, 3))
        for _ in range(20)
    ]

def result2composition(src, pha, img_path=None, output_source='data/output'):

    # pha = torch.zeros(1, 3, 1080, 1920)
    # pha = pha.resize_(1, 1, 720, 1280)

    start_time = time.time()

    target_size = pha.shape
    if target_size[-1] >= 1920:
        pha = resize(pha)

    # print(pha.shape)

    pha = pha.squeeze(0)
    
    if pha.is_floating_point():
        pha = pha.mul(255).byte()
    
    pha = np.transpose(pha.cpu().numpy(), (1, 2, 0))
    pha = pha[:, :, 0]    

    # pha = np.resize(pha, (720, 1280))   

    ret, binary = cv2.threshold(pha, 127, 255, cv2.THRESH_BINARY)
    label, num = measure.label(binary, connectivity=2, background=0, return_num=True)

    src = src.squeeze(0)
    if src.is_floating_point():
        src = src.mul(255).byte()
    src = np.transpose(src.cpu().numpy(), (1, 2, 0))


    if target_size[-1] > 1920:
        # binary = cv2.resize(binary, (target_size[-1], target_size[-2]))
        label = label.astype(np.uint8)
        label = cv2.resize(label, (target_size[-1], target_size[-2]))

    # print(np.max(label))
    img_show = src.copy()
    for i in range(num):
        cur_mask_bool = (label == (i+1)).astype(np.bool)
        # if cur_mask_bool.sum() < 5000:
        #     continue

        # img_show[cur_mask_bool] = label[cur_mask_bool] * 255
        img_show[cur_mask_bool] = src[cur_mask_bool] * 0.6 + color_masks[i%20] * 0.4
    
    img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
    # Image.fromarray(img_show).save(os.path.join(output_source, img_path))
    cv2.imwrite(os.path.join(output_source, img_path), img_show)

    


    # src = src.squeeze(0)
    # if src.is_floating_point():
    #     src = src.mul(255).byte()
    # src = np.transpose(src.cpu().numpy(), (1, 2, 0))

def result2mask(pha, img_path=None, output_source='data/output'):

    target_size = pha.shape
    if target_size[-1] >= 1920:
        pha = resize(pha)

    pha = pha.squeeze(0)
    
    if pha.is_floating_point():
        pha = pha.mul(255).byte()
    
    pha = np.transpose(pha.cpu().numpy(), (1, 2, 0))
    pha = pha[:, :, 0]    

    # pha = np.resize(pha, (720, 1280))   

    ret, binary = cv2.threshold(pha, 127, 255, cv2.THRESH_BINARY)
    label, num = measure.label(binary, connectivity=2, background=0, return_num=True)

    if target_size[-1] > 1920:
        binary = cv2.resize(binary, (target_size[-1], target_size[-2]))
    
    cv2.imwrite(os.path.join(output_source, img_path), binary)

def com_speed_test(pha):
    target_size = pha.shape
    if target_size[-1] >= 1920:
        pha = resize(pha)

    pha = pha.squeeze(0)
    
    if pha.is_floating_point():
        pha = pha.mul(255).byte()
    
    pha = np.transpose(pha.cpu().numpy(), (1, 2, 0))
    pha = pha[:, :, 0]    

    # pha = np.resize(pha, (720, 1280))   

    ret, binary = cv2.threshold(pha, 127, 255, cv2.THRESH_BINARY)
    label, num = measure.label(binary, connectivity=2, background=0, return_num=True)

    if target_size[-1] > 1920:
        binary = cv2.resize(binary, (target_size[-1], target_size[-2]))
        


def result2image_pro(src, pha, img_path=None, output_source='data/output'):

    start_time = time.time()

    src = src.squeeze(0)
    pha = pha.squeeze(0)

    if src.is_floating_point():
        src = src.mul(255).byte()
    if pha.is_floating_point():
        pha = pha.mul(255).byte()

    src = np.transpose(src.cpu().numpy(), (1, 2, 0))
    pha = np.transpose(pha.cpu().numpy(), (1, 2, 0))
    src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    pha = cv2.cvtColor(pha, cv2.COLOR_RGB2BGR)
    pha = cv2.cvtColor(pha, cv2.COLOR_BGR2GRAY)

    # pha = np.zeros((2160, 3840), dtype=np.uint8)

    ret, binary = cv2.threshold(pha, 127, 255, cv2.THRESH_BINARY)

    
    label, num = measure.label(binary, connectivity=2, background=0, return_num=True)

    color_masks = [
        np.random.randint(0, 256, (1, 3))
        for _ in range(num)
    ]

    print(time.time() - start_time)

    img_show = src.copy()
    # img_show = np.zeros_like(label, dtype=np.uint8)
    for i in range(num):
        cur_mask_bool = (label == (i+1)).astype(np.bool)
        if cur_mask_bool.sum() < 10000:
            continue
        
        img_show[cur_mask_bool] = src[cur_mask_bool] * 0.5 + color_masks[i] * 0.5

    # img_show = cv2.resize(img_show, (src.shape[1], src.shape[0]))
    # print(time.time() - start_time)
    

    cv2.imwrite(os.path.join(output_source, img_path), img_show)


def run_eval_miou(pha_file, mask_file):
    pha_list = os.listdir(pha_file)
    mask_list = os.listdir(mask_file)
    pha_list.sort()
    mask_list.sort()

    iou = []
    for fmask in tqdm(mask_list):
        pha = cv2.imread(os.path.join(pha_file, fmask))
        mask = cv2.imread(os.path.join(mask_file, fmask))

        intersection = np.sum(np.logical_and(mask, pha))
        union = np.sum(np.logical_or(mask, pha))

        if (union == 0) or (intersection == 0):
            continue
        iou_score = intersection / union
        iou.append(iou_score)

    iou = pd.DataFrame(columns = ['iou'], data = iou)
    print(iou)
    print('....  ...')
    print('miou:', iou['iou'].mean())
    print('....  ...')
    print('stab(iou>90%):', len(iou.values[iou.values>0.9])  / len(iou))
    print('....  ...')

def run_eval_miou_pro(pha_file, mask_file):
    pha_list = os.listdir(pha_file)
    mask_list = os.listdir(mask_file)
    pha_list.sort()
    mask_list.sort()

    iou = []
    for fmask in tqdm(mask_list):
        pha = cv2.imread(os.path.join(pha_file, fmask))
        mask = cv2.imread(os.path.join(mask_file, fmask))

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # cv2.imwrite('tmp.png', mask)
        # break

        intersection = np.sum(np.logical_and(mask, pha))
        union = np.sum(np.logical_or(mask, pha))

        if (union == 0) or (intersection == 0):
            continue
        iou_score = intersection / union
        iou.append(iou_score)

    iou = pd.DataFrame(columns = ['iou'], data = iou)
    print(iou)
    print('....  ...')
    print('miou:', iou['iou'].mean())
    print('....  ...')
    print('stab(iou>90%):', len(iou.values[iou.values>0.9])  / len(iou))
    print('....  ...')


def to_pil_image(src, pha, mode=None):

    target_size = pha.shape
    if target_size[-1] >= 1920:
        pha = resize(pha)

    # print(pha.shape)

    # pha = pha.squeeze(0)
    
    if pha.is_floating_point():
        pha = pha.mul(255).byte()
    
    pha = np.transpose(pha.cpu().numpy(), (1, 2, 0))
    pha = pha[:, :, 0]    

    # pha = np.resize(pha, (720, 1280))   

    ret, binary = cv2.threshold(pha, 127, 255, cv2.THRESH_BINARY)
    label, num = measure.label(binary, connectivity=2, background=0, return_num=True)

    color_masks = [
        np.random.randint(0, 256, (1, 3))
        for _ in range(num)
    ]

    src = src.squeeze(0)
    if src.is_floating_point():
        src = src.mul(255).byte()
    src = np.transpose(src.cpu().numpy(), (1, 2, 0))


    if target_size[-1] > 1920:
        # binary = cv2.resize(binary, (target_size[-1], target_size[-2]))
        label = label.astype(np.uint8)
        label = cv2.resize(label, (target_size[-1], target_size[-2]))

    # print(np.max(label))
    img_show = src.copy()
    for i in range(num):
        cur_mask_bool = (label == (i+1)).astype(np.bool)
        # if cur_mask_bool.sum() < 5000:
        #     continue

        # img_show[cur_mask_bool] = label[cur_mask_bool] * 255
        img_show[cur_mask_bool] = src[cur_mask_bool] * 0.6 + color_masks[i] * 0.4

    return Image.fromarray(img_show)

if __name__ == '__main__':
    run_eval_miou('data/val2022/VideoMatte240K/label', 'data/output')

    