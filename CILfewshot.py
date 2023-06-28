import numpy as np
import os, sys, time, copy, gc, argparse
from datetime import datetime
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
from path import Path
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
from torchmetrics.aggregation import MeanMetric
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import clip
from pytorch_loss import FocalLossV1, FocalLossV2, FocalLossV3
from datetime import datetime

from copy import deepcopy
from datasets.datautil_3D_memory_incremental_shapenet_co3d import *
from configs.shapenet_co3d_info import label2name, len_cls

from render.selector import Selector
from render.render import Renderer
from utils import *
from random import random

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torchpq.clustering import MinibatchKMeans
from models import ZeroShotCIL, FewShotCIL, focal_loss
#################################################### TConfigurations ############################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exp_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
"""
Arguments
"""
parser = argparse.ArgumentParser()


parser.add_argument('-a', '--exp_name', type=str, default='test', help='name of experiment')
parser.add_argument('-e0', '--epoch0', type=int, default=5, help='epochs of task_0')
parser.add_argument('-ei', '--epochi', type=int, default=20, help='epochs of task_i')
parser.add_argument('-p', '--use_new_feats_p', type=bool, default=False, help='whether to extract new feats_p')
parser.add_argument('-u', '--cluster_num', type=int, default=512, help='number of clusters when extracting feats_p')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batchsize')
parser.add_argument('-mem', '--use_memory', type=bool, default=True, help='whether to use memory in training data')
parser.add_argument('-memshot', '--maxshot_per_class', type=int, default=1, help='maxshot per class in training memory')
parser.add_argument('--loss_fn', type=str, default='bce', choices=['ce', 'bce', 'focal'],help='Loss_fn to use, [ce, focal]')
    
parser.add_argument('-d', '--dataset_path', type=str, default='./data/dataset/shapenet_co3d', help='path of dataset')
parser.add_argument('-n', '--ntasks', type=int, default=11, help='number of tasks')
parser.add_argument('-w', '--workers', type=int, default=8, help='number of dataloader workers')
parser.add_argument('-c', '--ckpt', type=str, default='/home/tudooh/qty/CLIP2PointCIL/CLIP2Point-main/pre_builts/vit32/best_eval.pth', help='path of ckpt')
parser.add_argument('-v', '--views', type=int, default=6, help='views of rendering')
parser.add_argument('-s', '--seed', type=int, default=42, help='seed')
parser.add_argument('-f', '--feats_p_path', type=str, default='./feats_p/feats_p.pth', help='path of feats_p')
parser.add_argument('-shot', '--fewshot', type=int, default=5, help='number of shots')
parser.add_argument('-nc', '--nclasses', type=int, default=89, help='number of total classes')


args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

"""main"""

# IO
# TODO:封装IO类，主log记录训练进程和关键点的结果，主类附属子类打印到对应log中
io = EXIOStream('exp_results/' + exp_time + ':' + args.exp_name)

# print configs
io.cprint('Configurations:', args.__dict__)

# Dataloader
dataloader_tot = DatasetGen(args, root=Path(args.dataset_path), fewshot=args.fewshot )

# CLIP model
clip_model, _ = clip.load('ViT-B/32', device='cpu')

def get_p(dataloader_1, model_depth):
    # aggrate all image features to 'image_feats'
    clip_feat_dim = 512
    selector = Selector(args.views, 0).to(device)
    render = Renderer(points_per_pixel=1, points_radius=0.02).to(device)
    image_feats = torch.empty([0, clip_feat_dim]).to(device)
    model_depth.eval()
    with torch.no_grad():
        for data in tqdm(dataloader_1):
            points = data['pointclouds'].to(device)
            c_views_azim, c_views_elev, c_views_dist = selector(points)
            images = render(points, c_views_azim, c_views_elev, c_views_dist, args.views, rot=False)
            b, n, c, h, w = images.shape
            images = images.reshape(-1, c, h, w)
            image_feat = model_depth(images)
            image_feats = torch.cat([image_feats, image_feat / image_feat.norm(dim=-1, keepdim=True)], dim=0) # [views*n_data, d_vec]
    # k-means for clustering
    kmeans = MinibatchKMeans(n_clusters=args.cluster_num, distance='cosine', init_mode='kmeans++')
    cluster_centers = kmeans.kmeanspp(image_feats.t()).float().t() # [num_cluster, d_vec]
    feats_p = svd_conversion(cluster_centers)
    # torch.save(feats_p, args.feats_p_path)
    return feats_p



def train_loop(dataloader, model, prompts_feats, optimizer, loss_fn, stat):
    # init metrics
    num_cls = prompts_feats.size(0)
    metrics = MetricCollection([
        MulticlassAccuracy(num_classes=num_cls, average="micro"),
        MulticlassPrecision(num_classes=num_cls, average="macro"),
        MulticlassRecall(num_classes=num_cls, average="macro")
    ]).to(device)
    train_loss = MeanMetric().to(device)
    if args.loss_fn == 'bce':
        cls_weight = get_cls_balance_weight(dataloader, num_cls).to(device)
    model.train()
    for data in tqdm(dataloader):
        label = data['labels'].unsqueeze(1).to(device)
        points = data['pointclouds'].to(device)
        
        # Compute prediction
        logits = model(points, prompts_feats)
        
        # Compute loss
        if args.loss_fn == 'ce':
            loss = loss_fn(logits, label)
        elif args.loss_fn == 'bce':
            loss = loss_fn(logits, label, weight=cls_weight)
        elif args.loss_fn == 'focal':
            loss_fn = focal_loss(num_classes=prompts_feats.size(0))
            loss = loss_fn(logits, label)
        pred = torch.max(logits, dim=1).indices
        
        # update metrics
        metrics.update(pred, label)
        train_loss.update(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = metrics.compute()
    _train_loss = train_loss.compute()
    io.cprint(f"[train] Task:{stat['task_id']}\tEpoch:{stat['epoch']}\tLoss:\t{_train_loss}\t\
                Accuracy:\t{(100*acc['MulticlassAccuracy']):>0.1f}\t\
                Precision:\t{(100*acc['MulticlassPrecision']):>0.1f}\t\
                Recall:\t{(100*acc['MulticlassRecall']):>0.1f}", name='epochs.log')
   
        

    


    
def test_loop(dataloader, model, prompts_feats, loss_fn, stat):
    num_batches = len(dataloader)
    num_cls = prompts_feats.size(0)
    # define metrics
    metrics = MetricCollection([
        MulticlassAccuracy(num_classes=num_cls, average="micro"),
        MulticlassPrecision(num_classes=num_cls, average="macro"),
        MulticlassRecall(num_classes=num_cls, average="macro")
    ]).to(device)
    confmat = MulticlassConfusionMatrix(num_classes=prompts_feats.size(0), normalize='true').to(device)
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            label = data['labels'].unsqueeze(1).to(device)
            points = data['pointclouds'].to(device)
            
            # Compute prediction and loss
            logits = model(points, prompts_feats)
            pred = torch.max(logits, dim=1).indices
            
            # update metrics
            confmat.update(pred, label)
            metrics.update(pred, label)
    # print metrics
    acc = metrics.compute()
    io.cprint(f"[Test] Task:{stat['task_id']}\tEpoch:{stat['epoch']}\t\
                Accuracy:\t{(100*acc['MulticlassAccuracy']):>0.1f}\t\
                Precision:\t{(100*acc['MulticlassPrecision']):>0.1f}\t\
                Recall:\t{(100*acc['MulticlassRecall']):>0.1f}", name='epochs.log')
    _fig, _ax = confmat.plot(add_text=False)
    plt.savefig('exp_results/' + exp_time + ':' + args.exp_name + '/task' + str(stat['task_id'])+'_epoch'+str(stat['epoch']) + '.png')
    
    


def train():
    # init models
    model_depth = deepcopy(clip_model.visual).to(device)
    model = None
    if args.ckpt is not None:
        print(args.ckpt)
        model_depth.load_state_dict(read_state_dict(args.ckpt))
    
    # loss functions (criteria)
    if args.loss_fn == 'ce' or args.loss_fn == 'bce':
        loss_fn = F.cross_entropy
    elif args.loss_fn == 'focal':
        loss_fn = FocalLossV1()
    
    
    # init prompts
    prompts = label2name
    prompts = ['image or projection or sketch of a ' + prompts[i] for i in range(len(prompts))]
    prompts = clip.tokenize(prompts)
    prompts = clip_model.encode_text(prompts)
    prompts_feats = (prompts / prompts.norm(dim=-1, keepdim=True)).to(device)
    
    # task_0 
    runtime_stat = {'task_id':0, 'epoch':0}
    dataloader_0 = dataloader_tot.get(0, 'training')
    train_loader_0 = dataloader_0[0]['train']
    test_loader_0 = dataloader_0[0]['test'] 

    # get feats_p
    if args.use_new_feats_p:
        feats_p = get_p(train_loader_0, model_depth).to(device)
        torch.save(feats_p, args.feats_p_path)
        model = FewShotCIL(args, feats_p=feats_p, p_mask=0b11).to(device)
    else:
        feats_p = torch.load(args.feats_p_path).to(device)
        model = FewShotCIL(args, feats_p=feats_p, p_mask=0b11).to(device)
    
    # freeze params
    for name, param in model.named_parameters():
        if 'rn' not in name and 'adapter' not in name and 'selector' not in name and 'renderer' not in name:
            param.requires_grad_(False)
    prompts_feats = prompts_feats.to(device).detach()
    
    # train task_0
    io.cprint("="*100, "Task 0", "="*100)
    optimizer_0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    for t in range(args.epoch0):
        io.cprint(f"---------------Epoch {t+1}-------------------")
        runtime_stat['epoch'] = t + 1
        train_loop(train_loader_0, model, prompts_feats[:len_cls[0]], optimizer_0, loss_fn, runtime_stat)
    test_loop(test_loader_0, model, prompts_feats[:len_cls[0]], loss_fn, runtime_stat)
    
    # freeze params
    for name, param in model.named_parameters():
        if 'rn' not in name:
            param.requires_grad_(False)
    
    # incremental tasks
    for task_id in range(1, args.ntasks):
        io.cprint("="*100, f"Task {task_id}", "="*100)
        runtime_stat['task_id'] = task_id
        # get dataloader_i
        dataloader_i = dataloader_tot.get(task_id, 'training')
        train_loader_i = dataloader_i[task_id]['train']
        test_loader_i = dataloader_i[task_id]['test']
        
        optimizer_i = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
        for t in range(args.epochi):
            io.cprint(f"-------------Epoch {t+1}--------------")
            runtime_stat['epoch'] = t + 1
            train_loop(train_loader_i, model, prompts_feats[:len_cls[task_id]], optimizer_i, loss_fn, runtime_stat)
        test_loop(test_loader_i, model, prompts_feats[:len_cls[task_id]], loss_fn, runtime_stat)
        

if __name__ == "__main__":
    io.cprint("Device:", device)
    train()

