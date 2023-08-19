import numpy as np
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm
import matplotlib.pyplot as plt
import clip
from pytorch_loss import FocalLossV1
from datetime import datetime
from sklearn.metrics import classification_report

from copy import deepcopy
from datasets.datautil_3D_memory_incremental_shapenet_co3d import *

from utils import *
from session_settings import shapenet2modelnet
from random import random
from datasets.CILdataset import *
from torch.nn.parallel import DataParallel

from models import FewShotCIL, focal_loss, FewShotCILwoRn, FewShotCILwoRn2, FewShotCILwPoint

import sys, importlib
sys.path.append('models')

#################################################### TConfigurations ############################################
exp_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
"""
Arguments
"""
parser = argparse.ArgumentParser()

parser.add_argument('-a', '--exp_name', type=str, default='shape2model_point_freeze_encoder', help='name of experiment')
parser.add_argument('-w', '--workers', type=int, default=4, help='number of dataloader workers')
parser.add_argument('--parallel', type=bool, default=False, help='whether to use multiple gpus')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--pin_memory', type=bool, default=True, help='whether to use pinned memory')
parser.add_argument('-s', '--seed', type=int, default=42, help='seed')

parser.add_argument('-e0', '--epoch0', type=int, default=10, help='epochs of task_0')
parser.add_argument('-ei', '--epochi', type=int, default=20, help='epochs of task_i')
parser.add_argument('-p', '--use_new_feats_p', type=bool, default=False, help='whether to extract new feats_p')
parser.add_argument('-u', '--cluster_num', type=int, default=512, help='number of clusters when extracting feats_p')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batchsize')
parser.add_argument('-mem', '--use_memory', type=bool, default=True, help='whether to use memory in training data')
parser.add_argument('-memshot', '--memory_shot', type=int, default=1, help='maxshot per class in training memory')
parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'bce', 'focal'],help='Loss_fn to use, [ce, focal]')
parser.add_argument('--model', type=str, default='FewShotCILwPoint', choices=['FewShotCILwPoint', 'FewShotCILwoRn', 'FewShotCIL'], help='which model to use')
    
parser.add_argument('-c', '--ckpt', type=str, default='/home/tudooh/qty/CLIP2PointCIL/CLIP2Point-main/pre_builts/vit32/best_eval.pth', help='path of ckpt')
parser.add_argument('-v', '--views', type=int, default=6, help='views of rendering')
parser.add_argument('-f', '--feats_p_path', type=str, default='./feats_p/feats_p.pth', help='path of feats_p')
parser.add_argument('-shot', '--fewshot', type=int, default=5, help='number of shots')

# point branchpoint_dpkg_path
parser.add_argument('--point_dpkg_path', type=str, default='./pre_builts/point/dgcnn_occo_cls.pth', help="path to pretrained weights [default: None]")
parser.add_argument('--point_emb_dims', type=int, default=1024, help='dimension of embeddings [default: 1024]')
parser.add_argument('--point_k', type=int, default=20, help='number of nearest neighbors to use [default: 20]')
    


args = parser.parse_args()

rank, world_size = setup_distributed(port=args.port)
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f'cuda:{args.local_rank}')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)

"""main"""

if rank == 0:
    # IO
    io = EXIOStream('exp_results/' + exp_time + ':' + args.exp_name)

    # print configs
    io.cprint('Configurations:', args.__dict__)


# CLIP model
clip_model, _ = clip.load('ViT-B/32', device='cpu')


""" ↓↓↓↓↓session setup↓↓↓↓↓ """
session_maker = shapenet2modelnet()
id2name = session_maker.get_id2name()
if rank == 0:
    io.cprint(session_maker.info())
""" ↑↑↑↑↑session setup↑↑↑↑↑ """




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
    for points, label in tqdm(dataloader):
        label = label.unsqueeze(1).to(device)
        points = points.to(device)
        
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
    if rank == 0:
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
    predict_list=[]
    ans_list=[]
    
    model.eval()
    with torch.no_grad():
        for points, label in tqdm(dataloader):
            label = label.unsqueeze(1).to(device)
            points = points.to(device)
            
            # Compute prediction and loss
            logits = model(points, prompts_feats)
            pred = torch.max(logits, dim=1).indices
            
            # update metrics
            confmat.update(pred, label)
            metrics.update(pred, label)
            predict_list.extend(pred.detach().cpu())
            ans_list.extend(label.detach().cpu())
    # print metrics
    acc = metrics.compute()
    if rank == 0:
        io.cprint(f"[Test] Task:{stat['task_id']}\tEpoch:{stat['epoch']}\t\
                    Accuracy:\t{(100*acc['MulticlassAccuracy']):>0.1f}\t\
                    Precision:\t{(100*acc['MulticlassPrecision']):>0.1f}\t\
                    Recall:\t{(100*acc['MulticlassRecall']):>0.1f}", name='epochs.log')
        _fig, _ax = confmat.plot(add_text=False)
        plt.savefig('exp_results/' + exp_time + ':' + args.exp_name + '/task' + str(stat['task_id'])+'_epoch'+str(stat['epoch']) + '.png')
        result = classification_report(ans_list, predict_list, target_names=id2name)             
        io.cprint(result)
    


def train():
    """ ==================initiation==================== """
    # init models
    model_depth = deepcopy(clip_model.visual).to(device)
    model = None
    if args.ckpt is not None:
        if rank == 0:
            print(args.ckpt)
        model_depth.load_state_dict(read_state_dict(args.ckpt))
    
    # loss functions (criteria)
    if args.loss_fn == 'ce' or args.loss_fn == 'bce':
        loss_fn = F.cross_entropy
    elif args.loss_fn == 'focal':
        loss_fn = FocalLossV1()
    
    # init prompts
    prompts = id2name
    prompts = ['image or projection or sketch of a ' + prompts[i] for i in range(len(prompts))]
    prompts = clip.tokenize(prompts)
    prompts = clip_model.encode_text(prompts)
    prompts_feats = (prompts / prompts.norm(dim=-1, keepdim=True)).to(device)
    
    # get feats_p
    if args.use_new_feats_p:
        feats_p = get_p(train_loader_0, model_depth).to(device)
        torch.save(feats_p, args.feats_p_path)
    else:
        feats_p = torch.load(args.feats_p_path).to(device)
        
    
    # init task_0 
    runtime_stat = {'task_id':0, 'epoch':0}
    dataset_train_0, dataset_test_0 = session_maker.make_session(session_id=0, update_memory=args.memory_shot)
    num_cat_0 = dataset_test_0.get_cat_num()
    train_sampler_0 = torch.utils.data.distributed.DistributedSampler(dataset_train_0)
    test_sampler_0 = torch.utils.data.distributed.DistributedSampler(dataset_test_0)
    train_loader_0 = torch.utils.data.DataLoader(dataset_train_0, batch_size=args.batch_size, num_workers=args.workers,
                                                    pin_memory=args.pin_memory, sampler=train_sampler_0)
    test_loader_0 = torch.utils.data.DataLoader(dataset_test_0, batch_size=args.batch_size, num_workers=args.workers,
                                                    pin_memory=args.pin_memory, sampler=test_sampler_0)
    
    model = getattr(importlib.import_module('models'), args.model)(args, feats_p=feats_p, p_mask=0b11).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)   
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)
    print("check1")
    # freeze params
    for name, param in model.named_parameters():
        if 'rn' not in name and 'adapter' not in name and 'selector' not in name and 'renderer' not in name and 'linear' not in name:
            param.requires_grad_(False)
    prompts_feats = prompts_feats.to(device).detach()
    
    
    """ ================Main Training============== """
    # train task_0
    if rank == 0:
        io.cprint("="*100, "Task 0", "="*100)
    optimizer_0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    for t in range(args.epoch0):
        if rank == 0:
            io.cprint(f"---------------Epoch {t+1}-------------------")
        runtime_stat['epoch'] = t + 1
        train_loop(train_loader_0, model, prompts_feats[:num_cat_0], optimizer_0, loss_fn, runtime_stat)
    test_loop(test_loader_0, model, prompts_feats[:num_cat_0], loss_fn, runtime_stat)
    
    # incremental tasks
    for task_id in range(1, session_maker.tot_session()):
        if rank == 0:
            io.cprint("="*100, f"Task {task_id}", "="*100)
        runtime_stat['task_id'] = task_id
        
        dataset_train_i, dataset_test_i = session_maker.make_session(session_id=task_id, update_memory=args.memory_shot)
        num_cat_i = dataset_test_i.get_cat_num()
        train_sampler_i = torch.utils.data.distributed.DistributedSampler(dataset_train_i)
        test_sampler_i = torch.utils.data.distributed.DistributedSampler(dataset_test_i)
        train_loader_i = torch.utils.data.DataLoader(dataset_train_i, batch_size=args.batch_size, num_workers=args.workers,
                                                        pin_memory=args.pin_memory, persistent_workers=True, sampler=train_sampler_i)
        test_loader_i = torch.utils.data.DataLoader(dataset_test_i, batch_size=args.batch_size, num_workers=args.workers,
                                                        pin_memory=args.pin_memory, persistent_workers=True, sampler=test_sampler_i)
        
        optimizer_i = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
        for t in range(args.epochi):
            if rank == 0:
                io.cprint(f"-------------Epoch {t+1}--------------")
            runtime_stat['epoch'] = t + 1
            train_loop(train_loader_i, model, prompts_feats[:num_cat_i], optimizer_i, loss_fn, runtime_stat)
        test_loop(test_loader_i, model, prompts_feats[:num_cat_i], loss_fn, runtime_stat)
        

if __name__ == "__main__":
    feats_p = torch.load(args.feats_p_path).to(device)
    if rank == 0:
        print(feats_p.shape)
        io.cprint("Device:", device)
    train()
