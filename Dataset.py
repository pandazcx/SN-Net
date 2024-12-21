import os
import os.path
import numpy as np
import random
import torch
import cv2
import torch.utils.data as udata

class Dataset(udata.Dataset):
    def __init__(self,win,path,aug_mode,train=True):
        super(Dataset, self).__init__()
        self.train = train
        self.win = win
        self.aug_mode = aug_mode
        LQ_dir = os.path.join(path,"input")
        HQ_dir = os.path.join(path,"groundtruth")

        self.LQ_list = []
        self.HQ_list = []
        path = os.listdir(LQ_dir)
        for j in range(len(path)):
            path[j] = os.path.join(LQ_dir, path[j])
        self.LQ_list += path  # os.listdir(i)
        self.LQ_list.sort()

        path = os.listdir(HQ_dir)
        for j in range(len(path)):
            path[j] = os.path.join(HQ_dir, path[j])
        self.HQ_list += path  # os.listdir(i)
        self.HQ_list.sort()

    def argument(self, lq,hq, mode):
        if mode == 1:
            lq_aug = lq
            hq_aug = hq
        elif mode == 2:
            lq_aug = lq[:, ::-1, :]  # hflip
            hq_aug = hq[:, ::-1, :]  # hflip
        elif mode == 3:
            lq_aug = lq[::-1, :, :]  # vflip
            hq_aug = hq[::-1, :, :]  # vflip
        elif mode == 4:
            lq_aug = lq[::-1, ::-1, :]
            hq_aug = hq[::-1, ::-1, :]

        return lq_aug, hq_aug

    def get_patch_random(self, lq, hq):
        win = self.win
        h, w = hq.shape[:2]
        x = random.randrange(0, w - win + 1)
        y = random.randrange(0, h - win + 1)
        lq = lq[y:y + win, x:x + win, :]
        hq = hq[y:y + win, x:x + win, :]
        return lq,hq

    def get_patch_fixed(self, lq, hq):
        win = self.win
        h, w = hq.shape[:2]
        midh = (h - win) // 2
        midw = (w - win) // 2
        lq = lq[midh:midh + win, midw:midw + win, :]
        hq = hq[midh:midh + win, midw:midw + win, :]
        return lq, hq

    def load_file(self,idx):
        LQ_data = cv2.imread(self.LQ_list[idx])
        LQ_data = cv2.cvtColor(LQ_data, cv2.COLOR_BGR2RGB)
        HQ_data = cv2.imread(self.HQ_list[idx])
        HQ_data = cv2.cvtColor(HQ_data, cv2.COLOR_BGR2RGB)

        return LQ_data,HQ_data

    def totensor(self, img):
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor / 255.
        return img_tensor

    def __len__(self):
        return len(self.HQ_list)

    def __getitem__(self, idx):
        LQ_data,HQ_data = self.load_file(idx)
        LQ_data,HQ_data = self.argument(LQ_data,HQ_data, np.random.randint(1, self.aug_mode))  # h,w,c
        if self.train:
            LQ_data,HQ_data = self.get_patch_random(LQ_data,HQ_data)  # h,w,c
        LQ = self.totensor(LQ_data)
        HQ = self.totensor(HQ_data)
        return LQ,HQ
