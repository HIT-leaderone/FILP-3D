import os, sys, math, random
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from path import Path
import torch
import torchvision.transforms as Tr
import pandas as pd
from tqdm import tqdm
from collections import Counter
import random
from .utils import read_from_path, offread_uniformed
from .shapenet import ShapeNet
from .transforms import default_pc_transform
from pytorch3d.transforms import axis_angle_to_matrix
from collections import Counter


class SessionDataset(Dataset):
    def __init__(self, _data, transform=default_pc_transform):
        super().__init__()
        self.paths = []
        self.labels = []
        self.load_method = []
        for i, path_list in enumerate(_data):
            #print(i, len(path_list))
            for path, load_method in path_list:
                self.paths.append(path)
                self.labels.append(i)
                self.load_method.append(load_method)
        self.transform = transform
        
    def save(self, save_path):
        with open(save_path, 'w') as f:
            for path in self.paths:
                print(path, file=f)
                
    def check(self, cmp_path):
        cmp_set = set()
        self_set = set()
        with open(cmp_path, 'r') as f:
            for path in f.readlines():
                cmp_set.add(path.strip().split('/')[-1])
        for path in self.paths:
            self_set.add(path.strip().split('/')[-1])
        if cmp_set != self_set:
            print(len(cmp_set|self_set)-len(cmp_set&self_set))
            #print('cmp:', cmp_set - cmp_set&self_set)
            #print('self:', self_set - cmp_set&self_set)
        else:
            print('ok!')
            
    def get_cat_num(self): 
        return len(dict(Counter(self.labels)))
    
    def set_transform(self, transform):
        self.transform = transform
    
    def __getitem__(self, idx):
        if self.load_method[idx] is None:
            point_cloud = read_from_path(self.paths[idx])
        else:
            point_cloud = self.load_method[idx](self.paths[idx])
        if self.transform is not None:
            point_cloud = self.transform(point_cloud)
        return point_cloud, self.labels[idx]
        
    def __len__(self):
        return len(self.paths)

class SessionMaker:
    """
    input: dataset(data_path, label_id), label-idx map
    store: paths of instances(train, test) 
    memory: examplar set
    session config: [base sets, incremental sets] -- [equal new classes per session, session list]
    output: dataset(train, test) for each session
    """
    def __init__(self):
        self.id2name = []
        self.name2id = {}
        self.data_train = [] # [label_0: [...], label_1: [...], ...] (path, load_method)
        self.data_test = [] # [label_0: [...], label_1: [...], ...] (path, load_method)
        self.cat_tot = 0 # total number of categories
        self.cat_cnt_train = [] # [0: cnt0, 1: cnt1, ...]
        self.cat_cnt_test = [] # [0: cnt0, 1: cnt1, ...]
        self.session_cfg = [] # [session_0: [], session_1: [], ...]  example: [[0, 1, 2, 3, 4, 5], [6, 7], [8, 9]]
        self.base_few_shot = 0
        self.inc_few_shot = 0
        # TODO: support tensor memory  (data, target)
        self.memory = [] # [((path, load_method)/data, label), ...]
    
    def update_memory(self, examplar):
        # print(examplar['path'])
        self.memory.append(examplar) 
    
    def tot_session(self):
        return len(self.session_cfg)
    
    def make_session(self, session_id, update_memory=0):
        """ return train_dataset & test dataset of session_id """
        # train data [session_i + memory]
        data_train = [[] for i in range(self.cat_tot)]
        tmp_new_mem = []
        for label in self.session_cfg[session_id]:
            if session_id == 0:
                data_train[label] = self.data_train[label]
                if self.base_few_shot > 0:
                    data_train[label] = data_train[label][:self.base_few_shot]
            else:
                data_train[label] = self.data_train[label]
                if self.inc_few_shot > 0:
                    data_train[label] = data_train[label][:self.inc_few_shot]
            # new memory examplar
            for path, load_method in data_train[label][:update_memory]:
                tmp_new_mem.append({'path': path, 'load_method': load_method, 'label': label})
        # memory
        for examplar in self.memory:
            data_train[examplar['label']].append((examplar['path'], examplar['load_method']))
        for new_examplar in tmp_new_mem:
            self.update_memory(new_examplar)
        
        # test data [session_0 + session_1 + ... + session_i]
        data_test = [[] for i in range(self.cat_tot)]
        for session in range(session_id + 1):
            for label in self.session_cfg[session]:
                data_test[label] = self.data_test[label]
        return SessionDataset(data_train), SessionDataset(data_test)

    def get_id2name(self):
        """ may used for prompt """
        return self.id2name
    
    def set_session_list(self, session_list, base_few_shot=0, inc_few_shot=5):
        """ session list contains categories` name instead of id """
        self.session_cfg = [[self.name2id[name] for name in session] for session in session_list]
        self.base_few_shot = base_few_shot
        self.inc_few_shot = inc_few_shot
    
    def set_session(self, num_base_cat, num_inc_cat, base_few_shot=0, inc_few_shot=5):
        """ in the order of appending datasets(categories) """
        res = num_base_cat
        self.session_cfg = [[i for i in range(num_base_cat)]]
        while res < self.cat_tot:
            self.session_cfg.append([i for i in range(res, min(res + num_inc_cat, self.cat_tot))])
            res += num_inc_cat
        self.base_few_shot = base_few_shot
        self.inc_few_shot = inc_few_shot
        
    def merge_new_data(self, 
                       new_data_train,
                       new_data_test,
                       new_cat_cnt_train,
                       new_cat_cnt_test,
                       new_id2name,
                       new_cat_tot,
                       new_dataset_name
                       ):
        """ assign new category idx, empty categories will be ignore """
        merged_cat_num = 0
        for i in range(new_cat_tot):
            if new_cat_cnt_train[i] == 0 and new_cat_cnt_test[i] == 0:
                continue
            self.data_train.append(new_data_train[i])
            self.data_test.append(new_data_test[i])
            self.cat_cnt_train.append(new_cat_cnt_train[i])
            self.cat_cnt_test.append(new_cat_cnt_test[i])
            self.id2name.append(new_id2name[i])
            self.name2id[new_id2name[i]] = self.cat_tot
            self.cat_tot += 1
            merged_cat_num += 1
        print(f"{merged_cat_num} categories has been merged from '{new_dataset_name}'.")
        
        
    def append_dataset(self, new_dataset, new_id2name, load_method=None, split_ratio=0.8):
        """
        dataset before splitted to (train, test)
        Args:
            new_dataset (iterable): dataset that return (path, label_id) when iterate
            new_id2name (list): label name list 
            split_ratio (float): the ratio of train/test after split
        """
        new_cat_tot = len(new_id2name)
        new_cat_cnt = [0 for i in range(new_cat_tot)]
        new_data = [[] for i in range(new_cat_tot)]  # before split
        # load paths
        for path, label in new_dataset:
            new_data[label].append((path, load_method))
            new_cat_cnt[label] += 1
        # split train/test
        new_data_train = [[] for i in range(new_cat_tot)]
        new_data_test = [[] for i in range(new_cat_tot)]
        new_cat_cnt_train = [0 for i in range(new_cat_tot)] 
        new_cat_cnt_test = [0 for i in range(new_cat_tot)] 
        for label, path_list in enumerate(new_data):
            num_train = int(new_cat_cnt[label] * split_ratio) # round
            num_test = new_cat_cnt[label] - num_train
            new_data_train[label] = path_list[:num_train + 1]
            new_data_test[label] = path_list[num_train + 1:]
            new_cat_cnt_train[label] = num_train
            new_cat_cnt_test[label] = num_test
        for name in new_id2name:
            if new_cat_cnt[new_id2name.index(name)] > 0 and name in self.id2name:
                print('duplicated category:', name)
        # merge
        self.merge_new_data(new_data_train, new_data_test, new_cat_cnt_train, new_cat_cnt_test, new_id2name, new_cat_tot, type(new_dataset).__name__)
        
    def append_dataset_train_test(self, new_dataset_train, new_dataset_test, new_id2name, load_method=None):
        new_cat_tot = len(new_id2name)
        new_data_train = [[] for i in range(new_cat_tot)] 
        new_data_test = [[] for i in range(new_cat_tot)]
        new_cat_cnt_train = [0 for i in range(new_cat_tot)] 
        new_cat_cnt_test = [0 for i in range(new_cat_tot)] 
        # load paths
        for path, label in new_dataset_train:
            new_data_train[label].append((path, load_method))
            new_cat_cnt_train[label] += 1
        for path, label in new_dataset_test:
            new_data_test[label].append((path, load_method))
            new_cat_cnt_test[label] += 1
        for name in new_id2name:
            if (new_cat_cnt_test[new_id2name.index(name)] > 0 or new_cat_cnt_train[new_id2name.index(name)] > 0) and name in self.id2name:
                print('duplicated category:', name)
        # merge
        self.merge_new_data(new_data_train, new_data_test, new_cat_cnt_train, new_cat_cnt_test, new_id2name, new_cat_tot, type(new_dataset_train).__name__)
    
    def info(self):
        info_dict = {}
        info_dict['category_num'] = self.cat_tot
        info_dict['categories'] = {i: name for i, name in enumerate(self.id2name)}
        info_dict['train_instance_num'] = sum(self.cat_cnt_train)
        info_dict['train_cat_cnt'] = {i: (self.id2name[i], num) for i, num in enumerate(self.cat_cnt_train)}
        info_dict['test_instance_num'] = sum(self.cat_cnt_test)
        info_dict['test_cat_cnt'] = {i: (self.id2name[i], num) for i, num in enumerate(self.cat_cnt_test)}
        info_dict['session_num'] = len(self.session_cfg)
        info_dict['session_cfg'] = {i: s for i, s in enumerate(self.session_cfg)}
        info_dict['base_few_shot'] = self.base_few_shot
        info_dict['inc_few_shot'] = self.inc_few_shot
        return info_dict
        


class ShapeNetCIL(Dataset):
    cat_labels = {'02691156': 0, '02747177': 1, '02773838': 2, '02801938': 3, '02808440': 4, '02818832': 5, '02828884': 6, '02843684': 7, '02871439': 8, '02876657': 9, '02880940': 10, '02924116': 11, '02933112': 12, '02942699': 13, '02946921': 14, '02954340': 15, '02958343': 16, 
                  '02992529': 17, '03001627': 18, '03046257': 19, '03085013': 20, '03207941': 21, '03211117': 22, '03261776': 23, '03325088': 24, '03337140': 25, '03467517': 26, '03513137': 27, '03593526': 28, '03624134': 29, '03636649': 30, '03642806': 31, '03691459': 32, '03710193': 33, 
                  '03759954': 34, '03761084': 35, '03790512': 36, '03797390': 37, '03928116': 38, '03938244': 39, '03948459': 40, '03991062': 41, '04004475': 42, '04074963': 43, '04090263': 44, '04099429': 45, '04225987': 46, '04256520': 47, '04330267': 48, '04379243': 49, '04401088': 50, 
                  '04460130': 51, '04468005': 52, '04530566': 53, '04554684': 54}
    id2name = ['airplane', 'ashcan', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellular telephone', 'chair', 'clock', 'computer keyboard', 'dishwasher', 'display', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'vessel', 'washer']
    def __init__(self, root='./data/ShapeNet55', partition='train', banlist=[], whole=False):
        assert partition in ['train', 'test']
        self.data_root = root + '/ShapeNet-55'
        self.pc_path = root + '/shapenet_pc'
        self.subset = partition
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        self.whole = whole

        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            lines = test_lines + lines
        self.file_list = []
        check_list = ['03001627-udf068a6b', '03001627-u6028f63e', '03001627-uca24feec', '04379243-', '02747177-', '03001627-u481ebf18', '03001627-u45c7b89f', '03001627-ub5d972a1', '03001627-u1e22cc04', '03001627-ue639c33f']
        
        # flag = False
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            if ShapeNetCIL.id2name[ShapeNetCIL.cat_labels[taxonomy_id]] in banlist:
                continue
            if taxonomy_id + '-' + model_id not in check_list:
                self.file_list.append({
                    'taxonomy_id': taxonomy_id,
                    'model_id': model_id,
                    'file_path': line
                })
        self.partition = partition
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        path = os.path.join(self.pc_path, sample['file_path'])
        return path, ShapeNetCIL.cat_labels[sample['taxonomy_id']]




class ModelNet40AlignCIL(Dataset):
    cats = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}
    id2name = list(cats.keys())
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, root='./data/ModelNet40_manually_aligned', partition='train', banlist=[]):
        assert partition in ('test', 'train')
        super().__init__()
        self.root = root
        self.partition = partition
        self._load_data(banlist)

    def _load_data(self, banlist):
        self.paths = []
        self.labels = []
        for cat in os.listdir(self.root):
            if cat in banlist:
                continue
            cat_path = os.path.join(self.root, cat, self.partition)
            for case in os.listdir(cat_path):
                if case.endswith('.off'):
                    self.paths.append(os.path.join(cat_path, case))
                    self.labels.append(ModelNet40AlignCIL.cats[cat])
    
    def get_load_method(self):
        def load(path, pt_num=1024):
            points = torch.Tensor(offread_uniformed(path, sampled_pt_num=pt_num)).type(torch.FloatTensor)
            rota1 = axis_angle_to_matrix(torch.tensor([0.5 * np.pi, 0, 0]))
            rota2 = axis_angle_to_matrix(torch.tensor([0, -0.5 * np.pi, 0]))
            points = points @ rota1 @ rota2
            return points.numpy()
        return load
        
    def __getitem__(self, index):      
        return self.paths[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)


class CO3DCIL(Dataset):
    def __init__(self, root='/data/CO3D', banlist=[]):
        self.data_root = root
        self.npoints = 2000
        
        self.label2name = []
        self.name2label = {}
        cat_cnt = []
        self.file_list = []
        label = 0
        for cat in os.listdir(self.data_root):
            if cat in banlist:
                continue
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
            label += 1

    def get_label2name(self):
        return self.label2name
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        return sample['file_path'], sample['label']

    def __len__(self):
        return len(self.file_list)



