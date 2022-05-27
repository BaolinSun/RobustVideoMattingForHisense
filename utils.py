import os
import av
import cv2
import shutil
import torch
import time
import re
import numpy as np
import pandas as pd


from skimage import measure
from tqdm import tqdm
from PIL import Image
from typing import Optional, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, Compose



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


container = av.open('output.mp4', mode='w')
stream = container.add_stream('h264', rate=round(30))
stream.pix_fmt = 'yuv420p'
stream.bit_rate = 1000000


class VideoWriter:
    def __init__(self, path, frame_rate, bit_rate=1000000):
        self.container = av.open(path, mode='w')
        self.stream = self.container.add_stream('h264', rate=round(frame_rate))
        self.stream.pix_fmt = 'yuv420p'
        self.stream.bit_rate = bit_rate

        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # cap_fps = 30
        # size = (1920, 1080)
        # self.video = cv2.VideoWriter(path, fourcc, cap_fps, size)
    
    def write(self, src, pha):

        # target_size = pha.shape
        # if target_size[-1] >= 1920:
        #     pha = resize(pha)

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


        # if target_size[-1] > 1920:
        #     # binary = cv2.resize(binary, (target_size[-1], target_size[-2]))
        #     label = label.astype(np.uint8)
        #     label = cv2.resize(label, (target_size[-1], target_size[-2]))

        # print(np.max(label))
        img_show = src.copy()
        for i in range(num):
            cur_mask_bool = (label == (i+1)).astype(np.bool)
            # if cur_mask_bool.sum() < 5000:
            #     continue

            # img_show[cur_mask_bool] = label[cur_mask_bool] * 255
            img_show[cur_mask_bool] = src[cur_mask_bool] * 0.6 + color_masks[i%20] * 0.4

        # img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
        # self.video.write(img_show)

        
        # # frames: [T, C, H, W]
        self.stream.width = src.shape[1]
        self.stream.height = src.shape[0]

        # # print('==========================')
        # # print(src.shape)
        # # print('==========================')

        # frames = frames.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
        frame = av.VideoFrame.from_ndarray(img_show, format='rgb24')
        self.container.mux(self.stream.encode(frame))
        # time.sleep(0.01)

                
    def close(self):
        self.container.mux(self.stream.encode())
        self.container.close()
        # self.video.release()

def result2video(src, pha, img_path=None, output_source='data/output'):

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
    
    # img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
    # Image.fromarray(img_show).save(os.path.join(output_source, img_path))
    # cv2.imwrite(os.path.join(output_source, img_path), img_show)

    


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
        


def result2image(src, pha):

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
    
    # img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)

    return img_show, binary


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

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)

def interact_segmentation(model, input_source: str, device: Optional[str] = None, downsample_ratio: Optional[float] = None, dtype: Optional[torch.dtype] = None):
    mTime = []
    mIOU = []

    def isegm(input_img, input_mask=None):
        img_ori = cv2.imread(input_img)
        start_time = 0
        last_time = 0

        transform = transforms.ToTensor()

        def drawbbox(pos):     
            ix, iy, x, y = pos[0], pos[1], pos[2], pos[3]    

            cv2.rectangle(img_ori, (ix, iy), (x, y), (0, 255, 0), 1)
            template = img_ori[iy:y, ix:x, :]
            src = transform(template)
            src = src.unsqueeze(0)
        
            start_time = time.time()
            with torch.no_grad():
                downsample_ratio = auto_downsample_ratio(*src.shape[2:])
                rec = [None] * 4
                src = src.to(device, dtype, non_blocking=True).unsqueeze(0)
                fgr, pha, *rec = model(src, *rec, downsample_ratio)
                img_show, pred_mask = result2image(src[0], pha[0])

                img_ori[iy:y, ix:x, :] = img_show

            last_time = time.time()


            if input_mask is not None:
                img_mask = cv2.imread(input_mask, cv2.IMREAD_GRAYSCALE)
                img_mask = img_mask[iy:y, ix:x]
                intersection = np.sum(np.logical_and(pred_mask, img_mask))
                union = np.sum(np.logical_or(pred_mask, img_mask))
                iou_score = intersection / union
            else:
                print("Don't input mask label !!!")

            print('....  ...')
            print('time:', last_time - start_time)
            print('IOU:', iou_score)
            print('....  ...')

            mIOU.append(iou_score)
            mTime.append(last_time - start_time)


            # cv2.destroyAllWindows()
            cv2.imshow('img', img_ori)
            # cv2.imwrite('result.png', img_ori)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

        def mouseEvent(event, x, y, flags, param):
            global ix, iy, drawing, mode, cap, template, tempFlag
            if event == cv2.EVENT_LBUTTONDOWN:
                tempFlag = True
                drawing = True
                ix, iy = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                if drawing == True:
                    drawing = False
                    drawbbox(pos=[ix, iy, x, y])

        # img = cv2.imread(input_img)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouseEvent, img_ori)
        cv2.imshow("image", img_ori)

        cv2.waitKey()
        cv2.destroyAllWindows()

    if os.path.isdir(input_source):

        input_imgs = []
        input_masks = []
        for ffile in os.listdir(input_source):
            for item in os.listdir(os.path.join(input_source, ffile)):
                if ffile == 'img':
                    input_imgs.append(os.path.join(input_source, ffile, item))
                elif ffile == 'label':
                    input_masks.append(os.path.join(input_source, ffile, item))
        
        for input_img in input_imgs:
            input_mask = re.sub('img', 'label', input_img)
            isegm(input_img, input_mask)

        mTime = pd.DataFrame(columns = ['time'], data = mTime)
        mIOU = pd.DataFrame(columns = ['iou'], data = mIOU)
        print('....  ...')
        print('mTime:', mTime['time'].mean())
        print('mIOU:', mIOU['iou'].mean())
        print('stab(iou>90%):', len(mIOU.values[mIOU.values>0.9])  / len(mIOU))
        print('....  ...')

    else:
        isegm(input_source)

    # if os.path.isdir(input_source):
    #     for item in os.listdir(input_source):
    #         isegm(os.path.join(input_source, item))
    # else:
    #     isegm(input_source)



    # transform = transforms.ToTensor()
    # source = ImageSequenceReader(input_source, transform)
    # reader = DataLoader(source, batch_size=1, pin_memory=True, num_workers=2)

    # for src, img_path in reader:


    # pass

if __name__ == '__main__':
    run_eval_miou('data/val2022/VideoMatte240K/label', 'data/output')

    