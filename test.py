import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
import matplotlib.pyplot as plt

from pytorch_loss import FocalLossV1, FocalLossV2, FocalLossV3
import torchvision
import torch
import numpy as np
import random
from datasets.CILdataset import *

if __name__ == '__main__':
    img_feats = np.randn(32,512)
    prompt_feature = np.randn(55,512)
    session_maker = SessionMaker()
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    modelnet_train = ModelNet40AlignCIL(partition='train')
    modelnet_test = ModelNet40AlignCIL(partition='test')
    modelnet_id2name = ModelNet40AlignCIL.id2name
    co3d = CO3DCIL()
    co3d_id2name = co3d.get_label2name()
    
    #session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name)
    session_maker.append_dataset_train_test(modelnet_train, modelnet_test, modelnet_id2name)
    #session_maker.append_dataset(co3d, co3d_id2name)
    
    session_maker.set_session(num_base_cat=20, num_inc_cat=4)
    id2name = session_maker.get_id2name()
    cnt = 0
    info = session_maker.info()
    for item in info.items():
        print(item)
    
    for session in range(session_maker.tot_session()):
        data_train, data_test = session_maker.make_session(session)
        for path, label in tqdm(data_train):
            cnt += 1
        for path, label in tqdm(data_test):
            cnt += 1
            