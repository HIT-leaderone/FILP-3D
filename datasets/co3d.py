import os
import torch
import numpy as np
import torch.utils.data as data
import h5py
from typing import Tuple
import collections
from pytorch3d.io import load_obj
from .utils import plyread
import random
from torchvision.transforms import Normalize, ToTensor
from PIL import Image


def select_by_index(list, idx):
    ret = []
    for i, x in enumerate(list):
        if i in idx:
            ret.append(list[i])
    return ret

class CO3D(data.Dataset):
    train_idx = []
    test_idx = []
    def __init__(self, partition='train', num_points=1024):
        assert partition in ['train', 'test']
        self.data_root = '/data/CO3D'
        self.index_root = './data/CO3D'
        self.partition = partition
        self.npoints = 2000
        self.sample_points_num = num_points
        
        self.label2name = []
        self.name2label = {}
        cat_cnt = []
        self.file_list = []
        for label, cat in enumerate(os.listdir(self.data_root)):
            print(label, cat)
            cat_cnt.append(0)
            self.name2label[cat] = label
            self.label2name.append(cat)
            for tax in [f for f in os.listdir(os.path.join(self.data_root, cat)) if not os.path.isfile(os.path.join(self.data_root, cat, f))]:
                if os.path.exists(os.path.join(self.data_root, cat, tax, f"{self.npoints}.ply")):
                    cat_cnt[-1] += 1
                    self.file_list.append({
                        'label': label,
                        'model_id': cat_cnt[-1],
                        'file_path': os.path.join(self.data_root, cat, tax, f"{self.npoints}.ply")
                    })
        self.permutation = np.arange(self.npoints)
        print(cat_cnt)
        if len(CO3D.train_idx) == 0:
            CO3D.make_partition(self.file_list, cat_cnt)
        
        if self.partition == 'train':
            self.file_list = select_by_index(self.file_list, CO3D.train_idx)
        elif self.partition == 'test':
            self.file_list = select_by_index(self.file_list, CO3D.test_idx)

    @classmethod
    def make_partition(cls, file_list, cat_cnt, train_ratio=0.8, fewshot=5):
        CO3D.train_idx = []
        CO3D.test_idx = []
        train_cat_cnt = [0] * len(cat_cnt)
        for idx, f in enumerate(file_list):
            if(train_cat_cnt[f['label']] < train_ratio*cat_cnt[f['label']]):
                CO3D.train_idx.append(idx)
                train_cat_cnt[f['label']] += 1
        if fewshot:
            fs_idx = []
            fs_cat_cnt = [0] * len(cat_cnt)
            for idx in CO3D.train_idx:
                if(fs_cat_cnt[file_list[idx]['label']] < fewshot):
                    fs_idx.append(idx)
                    fs_cat_cnt[file_list[idx]['label']] += 1
            CO3D.train_idx = fs_idx
        CO3D.test_idx = [i for i in range(len(file_list)) if i not in CO3D.train_idx]
    
    def get_label2name(self):
        return self.label2name

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        points = np.array(plyread(sample['file_path'], points_only=True)._vertices, dtype=np.float32)
        points = self.random_sample(points, self.sample_points_num)
        points = self.pc_norm(points)
        points = torch.from_numpy(points).float()
        
        label = sample['label']
        return points, label

    def __len__(self):
        return len(self.file_list)
    
    
if __name__ == '__main__':
    print("a")
    train_data = CO3D('train')
    test_data = CO3D('test')
    train_dataloader = data.DataLoader(dataset=train_data, shuffle=True)