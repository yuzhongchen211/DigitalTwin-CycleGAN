import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json 
import torch
import random

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train',rate=1.0):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/real_block' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/simu_pure_1k' % mode) + '/*.*'))
        self.files_A = random.sample(self.files_A,int(len(self.files_A)*rate))
        self.files_B = random.sample(self.files_B,int(len(self.files_B)*rate))

        with open(os.path.join(root, '%s/label_pure_1k.json'%mode),'r', encoding='UTF-8') as f:
            self.label_B = json.load(f)

    def __getitem__(self, index):
        item_A = np.load(self.files_A[index % len(self.files_A)])
        item_B = np.load(self.files_B[index % len(self.files_B)],allow_pickle=True)
        file_name = self.files_B[index%len(self.files_B)]
        file_name = file_name.split('/')[-1]
        label_b = self.label_B[file_name]
        return {'A': item_A, 'B': item_B, 'B_label':label_b}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
