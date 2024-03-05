import numpy as np
import torch
from plyfile import PlyData
import yaml
from datetime import datetime
import os
from collections import Counter
import torch.nn.functional as F
class Argument(object):
    def __init__(self, config_file):
        super(Argument, self).__init__()    
        config_file = open(config_file, 'r') # args.config_path
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        for key in config:
            setattr(self, key, config[key])
class IOStream():
    def __init__(self, path, timestamp=False):
        self.f = open(path, 'a')
        self.t = timestamp

    def cprint(self, text):
        print(text)
        if self.t is True:
            self.f.write('['+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+']' + text + '\n')
        else: 
            self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
        
class EXIOStream():
    """
    EXIOStream for loggings
    each instance has a main stream and several arbitary sub-stream
    """
    def __init__(self, dir_path, main_name='main.log'):
        self.dir = dir_path
        self.main_name = main_name
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.fmain = open(self.dir+'/'+main_name, 'a')
        
    def cprint(self, *args, name=None, to_main=True):
        """
        Args:
            name (str, optional): if needs to print to sub-stream, use name='filename'. Defaults to None.
            to_main (bool, optional): whether to also print to main-stream. Defaults to True.
        """
        if to_main == True:
            if name is None:
                print('['+self.main_name+']', *args)
            print('['+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+']', *args, file=self.fmain)
        if name is not None and name != self.main_name:
            print('['+name+']', *args)
            f = open(self.dir+'/'+name, 'a')
            print(*args, file=f)
            f.close()



def read_state_dict(path):
    ckpt = torch.load(path)
    base_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    for key in list(base_ckpt.keys()):
        if key.startswith('point_model.'):
            base_ckpt[key[len('point_model.'):]] = base_ckpt[key]
        del base_ckpt[key]
    return base_ckpt


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def centroid_conversion(centroids):
    U, S, Vh = torch.linalg.svd(centroids.transpose(0,1), full_matrices=False)
    ind = 0
    for i in range(centroids.shape[0]):
        if i==0:
            ind = i
            if float(torch.sum(S[0]))>float(0.95*torch.sum(S)):
                break
        else:
            if float(torch.sum(S[:i+1]))>float(0.95*torch.sum(S)):
                break
            else:
                ind = i
    cent = U[:, :ind] @ torch.diag(S[:ind]) @ Vh[:ind, :]
    return cent.transpose(0,1)

def svd_conversion(centroids):
    U, S, Vh = torch.linalg.svd(centroids.transpose(0,1), full_matrices=False)
    ind = 0
    for i in range(centroids.shape[0]):
        if i==0:
            ind = i
            if float(torch.sum(S[0]))>float(0.95*torch.sum(S)):
                break
        else:
            if float(torch.sum(S[:i+1]))>float(0.95*torch.sum(S)):
                break
            else:
                ind = i
    cent = U[:, :ind] # [d_vec, u]
    return cent


def get_cls_balance_weight(train_dataloader, num_classes): 
    """
    Args:
        train_dataloader (_type_): _description_
    Return:
        tensor[n_cls]
    """
    # TODO: 
    weight = torch.zeros(num_classes, requires_grad=False)
    for points, label in train_dataloader:
        label = F.one_hot(label, num_classes)
        weight += torch.sum(label, dim=0)
    norm = 0
    for l in range(num_classes):
        if weight[l] != 0:
            weight[l] = 1.0 / weight[l]
            norm += weight[l]
    weight /= norm
    return weight
    