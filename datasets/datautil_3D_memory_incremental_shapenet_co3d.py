import os, sys, math, random
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from path import Path
import torch
import torchvision.transforms as Tr
import pandas as pd
from configs.shapenet_co3d_info import task_ids_total as tid, len_cls, label2name
from tqdm import tqdm
from collections import Counter


"""#Preprocess"""
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2    
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return  norm_pointcloud

class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        noise = np.random.normal(0, 0.02, (pointcloud.shape)) 
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        return torch.from_numpy(pointcloud)



"""#custom Dataset"""
class PointCloudData(Dataset):
    def __init__(self, root_dir, folder="train"):
        self.root_dir = root_dir
        folders = sorted([int(dir) for dir in os.listdir(root_dir) if os.path.isdir(root_dir/dir)])
        folders = [str(i) for i in folders]
        # {'airplane': 0, ....}
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.files = []
        self.file_class_count = {category: 0 for category in self.classes.keys()}
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.pt'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    sample['name'] = category
                    self.files.append(sample)
                    self.file_class_count[category] += 1
        del self.file_class_count

    def __len__(self):
        return len(self.files)


class iPointCloudData(PointCloudData):
    def __init__(self, root, task_num, classes, folder, fewshot=0, transform=None, phase=None):
        super(iPointCloudData, self).__init__(root_dir=root, folder=folder) 
        if not isinstance(classes, list):
            classes = [classes]
        self.phase = phase
        self.task_num = task_num
        self.fewshot = 0 if self.task_num == 0 else fewshot
        self.folder = folder
        if self.folder == 'train':
            if self.task_num == 0:
                self.class_mapping = {c:i for i,c in enumerate(classes)}
            else:
                self.class_mapping = {c:i+len_cls[self.task_num-1] for i,c in enumerate(classes)}
        elif self.folder == 'test':
            self.class_mapping = {c:i for i,c in enumerate(classes)}

        self.transforms = transform
        pointcloud = []
        labels = []
        names = []
        flag_task = []
        task_label = []

        train_class_file_count = {i:0 for i in classes}

        for i in range(len(self.files)):
            if self.classes[self.files[i]['category']] in classes:
                if self.fewshot>0 and self.folder == 'train' and self.phase == 'training':
                    if train_class_file_count[self.classes[self.files[i]['category']]] >= self.fewshot:
                        continue
                    else:
                        train_class_file_count[self.classes[self.files[i]['category']]] += 1

                # max instances for each class in test
                """# TODO:
                if self.folder == 'test' and self.phase == 'training':
                    if train_class_file_count[self.classes[self.files[i]['category']]] >= 50:
                        continue
                    else:
                        train_class_file_count[self.classes[self.files[i]['category']]] += 1"""
                
                pointcloud.append(self.files[i]['pcd_path']) 
                labels.append(self.class_mapping[self.classes[self.files[i]['category']]])
                names.append(label2name[int(self.files[i]['name'])])
                flag_task.append(self.task_num)
        if self.phase=='training' and self.folder=="train":
                print("len_data with_Out_mem: ", len(labels))

        self.pointcloud = pointcloud   #adress of data of task 
        self.labels = labels
        self.names = names
        self.flag_task = flag_task
    
    def add_memory(self, mem):
        """
        only in training
        """
        self.pointcloud.extend(mem['pcd_path'])
        self.labels.extend(mem['labels'])
        self.names.extend(mem['names'])
        self.flag_task.extend(mem['task_label'])
    
    def sample_for_memory(self, maxshot_per_class=5):
        """
        sample traning memory for following tasks
        """
        cls_cnt = Counter()
        new_mem = {'labels':[], 'pcd_path':[], 'names':[], 'task_label':[]}
        for i in range(len(self.labels)):
            if cls_cnt[self.labels[i]] >= maxshot_per_class:
                continue
            cls_cnt[self.labels[i]] += 1
            new_mem['labels'].append(self.labels[i])
            new_mem['pcd_path'].append(self.pointcloud[i])
            new_mem['names'].append(self.names[i])
            new_mem['task_label'].append(self.flag_task[i])
        return new_mem
            
    def __len__(self):
        return len(self.pointcloud)
  
    def preproc(self, file):
        pcld = torch.load(file)
        if self.transforms:
          pointclouds = self.transforms(pcld)
        return pointclouds.float()

    def __getitem__(self, index):
        pcd_path = self.pointcloud[index]
        pointclouds = self.preproc(pcd_path)
        pointclouds,labels,names,task = pointclouds,self.labels[index],self.names[index],self.flag_task[index]

        if self.folder=="test" :
            return {'pointclouds':pointclouds,'labels':labels,'pcd_path':pcd_path,'names':names,'task_label':task}
        else:
            return {'pointclouds':pointclouds,'labels':labels,'pcd_path':pcd_path,'names':names,'task_label':task}
    
        



"""#DataLoader"""
class DatasetGen(object):
    """docstring for DatasetGen"""
    def __init__(self, args, root, fewshot=0):
        super(DatasetGen, self).__init__()
        self.root = root
        self.fewshot = fewshot
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        self.pin_memory = True #True 
        self.num_tasks = args.ntasks
        self.num_classes =args.nclasses
        self.inputsize = [1024,3]
        self.transformation = Tr.Compose([           
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])
        self.default_transforms=Tr.Compose([
                                Normalize(),
                                ToTensor()
                              ])

        # task ids 
        self.task_ids_total = tid
        # memory
        self.use_memory = args.use_memory
        self.maxshot_per_class = args.maxshot_per_class
        self.memory = {'labels':[], 'pcd_path':[], 'names':[], 'task_label':[]}
    
    def update_memory(self, new_mem):
        self.memory['labels'].extend(new_mem['labels'])
        self.memory['pcd_path'].extend(new_mem['pcd_path'])
        self.memory['names'].extend(new_mem['names'])
        self.memory['task_label'].extend(new_mem['task_label'])
        print(f"{len(new_mem['labels'])} new instances added to memory, now the memory size is {len(self.memory['labels'])}")
    
    
    def get(self, task_id, phase):
        self.dataloaders = {}
        self.dataloaders[task_id] = {}

        task_id_test = task_id
        task_ids_test = []
        task_ids = [list(arr) for arr in self.task_ids_total]
        for i in range(task_id_test + 1):
            task_ids_test = self.task_ids_total[task_id_test] + task_ids_test
            task_id_test = task_id_test - 1

        self.train_set = {}
        self.test_set = {}

        self.train_set[task_id] = iPointCloudData(root=self.root, classes=task_ids[task_id], task_num=task_id, folder="train", 
                                                transform=self.default_transforms, phase=phase, fewshot=self.fewshot)
        self.test_set[task_id] = iPointCloudData(root=self.root, classes=task_ids_test, task_num=task_id, folder='test', 
                                                transform=self.default_transforms, phase=phase, fewshot=0)
        
        if self.use_memory:
            new_mem = self.train_set[task_id].sample_for_memory(self.maxshot_per_class)
            self.train_set[task_id].add_memory(self.memory)
            self.update_memory(new_mem)
            
        
        train_loader = torch.utils.data.DataLoader(self.train_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                    pin_memory=self.pin_memory,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                    pin_memory=self.pin_memory, shuffle=True)
        self.dataloaders[task_id]['train'] = train_loader  
        self.dataloaders[task_id]['test'] = test_loader 
        
        if phase =='training':    
            print ('Task ID: {} -> {}'.format(task_id, task_ids[task_id]))
            print ("Task Clases:", task_ids[task_id])
            print ("Training set size:   {} pointcloud of {}x{}".format(len(train_loader.dataset),self.inputsize[0],self.inputsize[1]))
            print ("Test set size:       {} pointcloud of {}x{}".format(len(test_loader.dataset),self.inputsize[0],self.inputsize[1])) 
        
        return self.dataloaders