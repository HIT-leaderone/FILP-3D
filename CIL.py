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
from session_settings import shapenet2modelnet, shapenet2co3d
from random import random
from datasets.CILdataset import *
from torch.nn.parallel import DataParallel

from models import FewShotCIL, focal_loss, FewShotCILwoRn, FewShotCILwoRn2, FewShotCILwPoint

import sys, importlib
sys.path.append('models')

#################################################### TConfigurations ############################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_id = [0, 1]
exp_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
len_cls = [39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89] 
"""
Arguments
"""
parser = argparse.ArgumentParser()

parser.add_argument('-a', '--exp_name', type=str, default='shape2model_point_freeze_encoder', help='name of experiment')
parser.add_argument('-w', '--workers', type=int, default=8, help='number of dataloader workers')
parser.add_argument('--parallel', type=bool, default=False, help='whether to use multiple gpus')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--pin_memory', type=bool, default=True, help='whether to use pinned memory')
parser.add_argument('-s', '--seed', type=int, default=42, help='seed')

parser.add_argument('-e0', '--epoch0', type=int, default=10, help='epochs of task_0')
parser.add_argument('-ei', '--epochi', type=int, default=20, help='epochs of task_i')
parser.add_argument('--lr0', type=int, default=1e-3, help='epochs of task_0')
parser.add_argument('--lri', type=int, default=1e-3, help='epochs of task_i')
parser.add_argument('-p', '--use_new_feats_p', type=bool, default=False, help='whether to extract new feats_p')
parser.add_argument('-u', '--cluster_num', type=int, default=512, help='number of clusters when extracting feats_p')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batchsize')
parser.add_argument('-mem', '--use_memory', type=bool, default=True, help='whether to use memory in training data')
parser.add_argument('-memshot', '--memory_shot', type=int, default=1, help='maxshot per class in training memory')
parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'bce', 'focal'],help='Loss_fn to use, [ce, focal]')
parser.add_argument('--model', type=str, default='FewShotCILwPoint', choices=['FewShotCILwPoint', 'FewShotCILwoRn', 'FewShotCIL'], help='which model to use')
parser.add_argument('--save_model_name', type=str, default='save_model_wo', help='new name of model')
parser.add_argument('--model_name', type=str, default='save_model_wo', help='name of model')
    
parser.add_argument('-c', '--ckpt', type=str, default='/home/tudooh/qty/CLIP2PointCIL/CLIP2Point-main/pre_builts/vit32/best_eval.pth', help='path of ckpt')
parser.add_argument('-v', '--views', type=int, default=6, help='views of rendering')
parser.add_argument('-f', '--feats_p_path', type=str, default='./feats_p/feats_p.pth', help='path of feats_p')
parser.add_argument('-shot', '--fewshot', type=int, default=5, help='number of shots')
parser.add_argument('--use_FSCIL3D_dataset', type=bool, default=False, help='whether to FSCIL-3D benchmark')

# point branchpoint_dpkg_path
parser.add_argument('--point_dpkg_path', type=str, default='./pre_builts/point/dgcnn_occo_cls.pth', help="path to pretrained weights [default: None]")
parser.add_argument('--point_emb_dims', type=int, default=1024, help='dimension of embeddings [default: 1024]')
parser.add_argument('--point_k', type=int, default=20, help='number of nearest neighbors to use [default: 20]')
parser.add_argument('-cf', '--config_file', type=str, default='./configs/ShapeNet_CO3D_incremental_config.yaml', help='path of config file')

input_arguments = parser.parse_args()

# Args = Argument(config_file = input_arguments.config_file)
# print('Configurations:', Args.__dict__)    


args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
novel_acc_list_marco = []
novel_acc_list_mirco = []
"""main"""

# IO
io = EXIOStream('exp_results/' + exp_time + ':' + args.exp_name)

# print configs
io.cprint('Configurations:', args.__dict__)


# CLIP model
clip_model, _ = clip.load('ViT-B/32', device='cpu')


""" ↓↓↓↓↓session setup↓↓↓↓↓ """
session_maker = shapenet2co3d()
id2name = session_maker.get_id2name()
io.cprint(session_maker.info())
# dataloader_tot = DatasetGen(Args, root=Path(Args.dataset_path), fewshot=Args.fewshot)
""" ↑↑↑↑↑session setup↑↑↑↑↑ """


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
    fig = plt.figure()		# 创建图形实例
    ax = plt.subplot(111)		# 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        colorplt = 'brown'
        # if label[i] == 0:
        #     colorplt = 'brown'
        # elif label[i] == 5:
        #     colorplt = 'blue'
        # elif label[i] == 11:
        #     colorplt = 'cyan'
        # elif label[i] == 17:
        #     colorplt = 'green'
        # elif label[i] == 19:
        #     colorplt = 'black'
        # # elif label[i] == 23:
        # #     colorplt = 'magenta'
        # elif label[i] == 31:
        #     colorplt = 'red'
        # elif label[i] == 45:
        #     colorplt = 'yellow'
        # else: continue
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
        #          fontdict={'weight': 'bold', 'size': 7})
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=colorplt,
        #          fontdict={'weight': 'bold', 'size': 7})
        plt.scatter(data[i, 0], data[i, 1], c=colorplt, marker='o', s=5)
    plt.xticks()		# 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig

# 主函数，执行t-SNE降维
def tsne(vec, label):
    print('Starting compute t-SNE Embedding...')
    print(vec.shape)
    print(label.shape)
    ts = TSNE(n_components=2, init='pca', random_state=0)
    reslut = ts.fit_transform(vec)
    fig = plot_embedding(reslut, label, 't-SNE Embedding of digits')
    # 显示图像
    # plt.show()
    plt.savefig('./image/test.jpg')

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
    
    if args.use_FSCIL3D_dataset:
        for data in tqdm(dataloader):
            label = data['labels']
            points = data['pointclouds']
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
    else :
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
    io.cprint(f"[train] Task:{stat['task_id']}\tEpoch:{stat['epoch']}\tLoss:\t{_train_loss}\t\
                Accuracy:\t{(100*acc['MulticlassAccuracy']):>0.1f}\t\
                Precision:\t{(100*acc['MulticlassPrecision']):>0.1f}\t\
                Recall:\t{(100*acc['MulticlassRecall']):>0.1f}", name='epochs.log')
    if stat['task_id'] == 0:
        torch.save(model.state_dict(), '%s/best.pth' % (args.save_model_name))
        io.cprint('save the best test acc at %d' % (stat['epoch'] + 1))
    
last = 0
    
def test_loop(dataloader, model, prompts_feats, loss_fn, stat):
    global last
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
    novel_tot={}
    novel_acc={}
    for i in range(last,num_cls):
        # print("novel_class=",i)
        novel_tot[i] = 0
        novel_acc[i] = 0
    
    vec = []
    labs = []
    
    model.eval()
    with torch.no_grad():
        if args.use_FSCIL3D_dataset:
            for data in tqdm(dataloader):
                label = data['labels']
                points = data['pointclouds']
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
                for i in range(len(pred)):
                    if label[i]>=last:
                        novel_tot[label[i].item()] +=1
                        if pred[i] == label[i]:
                            novel_acc[label[i].item()] +=1
        else:
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
                for i in range(len(pred)):
                    if label[i]>=last:
                        novel_tot[label[i].item()] +=1
                        if pred[i] == label[i]:
                            novel_acc[label[i].item()] +=1
                Encoder = model.encoder(points).cpu().detach().numpy()
                for i in range(len(points)):    
                    vec.append(Encoder[i])
                    labs.append(int(label[i].cpu().detach()))
    
    vec = torch.tensor(vec)
    labs = torch.tensor(labs)
    tsne(vec,labs)
    
    # print metrics
    acc = metrics.compute()
    io.cprint(f"[Test] Task:{stat['task_id']}\tEpoch:{stat['epoch']}\t\
                Accuracy:\t{(100*acc['MulticlassAccuracy']):>0.1f}\t\
                Precision:\t{(100*acc['MulticlassPrecision']):>0.1f}\t\
                Recall:\t{(100*acc['MulticlassRecall']):>0.1f}", name='epochs.log')
    _fig, _ax = confmat.plot(add_text=False)
    plt.savefig('exp_results/' + exp_time + ':' + args.exp_name + '/task' + str(stat['task_id'])+'_epoch'+str(stat['epoch']) + '.png')
    #result = classification_report(ans_list, predict_list, target_names=id2name)             
    #io.cprint(result)
    macro_avg = 0
    micro_avg = 0
    tot_novel = 0
    for i in range(last,num_cls):
        macro_avg += novel_acc[i]/novel_tot[i]
        micro_avg += novel_acc[i]
        tot_novel += novel_tot[i]
    macro_avg /= (num_cls - last)
    micro_avg /= tot_novel
    io.cprint(f"[Test] Task:{stat['task_id']}\tEpoch:{stat['epoch']}\t\
                Novel Macro Accuracy:\t{(100*macro_avg):>0.1f}\t\
                Novel Micro Precision:\t{(100*micro_avg):>0.1f}", name='epochs.log')
    novel_acc_list_marco.append(macro_avg)
    novel_acc_list_mirco.append(micro_avg)
    io.cprint(f"[Test] Task:{stat['task_id']}\tEpoch:{stat['epoch']}\t\
                Ans Macro Accuracy:\t{(100*sum(novel_acc_list_marco)/len(novel_acc_list_marco)):>0.1f}\t\
                Ans Micro Precision:\t{(100*sum(novel_acc_list_mirco)/len(novel_acc_list_mirco)):>0.1f}", name='epochs.log')
    last = num_cls
    


def train():
    """ ==================initiation==================== """
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
    prompts = id2name
    prompts = ['image or projection or sketch of a ' + prompts[i] for i in range(len(prompts))]
    prompts = clip.tokenize(prompts)
    prompts = clip_model.encode_text(prompts)
    prompts_feats = (prompts / prompts.norm(dim=-1, keepdim=True)).to(device)
    

    
    # init task_0 
    runtime_stat = {'task_id':0, 'epoch':0}
    if args.use_FSCIL3D_dataset:
        task_id = 0
        dataloader_0 = dataloader_tot.get(task_id, 'training')
        train_loader_0 = dataloader_0[task_id]['train']
        test_loader_0 = dataloader_0[task_id]['test']
        num_cat_0 = len_cls[task_id]
    else :
        dataset_train_0, dataset_test_0 = session_maker.make_session(session_id=0, update_memory=args.memory_shot)
        num_cat_0 = dataset_test_0.get_cat_num()
        train_loader_0 = torch.utils.data.DataLoader(dataset_train_0, batch_size=args.batch_size, num_workers=args.workers,
                                                        pin_memory=args.pin_memory, shuffle=True, persistent_workers=True)
        test_loader_0 = torch.utils.data.DataLoader(dataset_test_0, batch_size=args.batch_size, num_workers=args.workers,
                                                    pin_memory=args.pin_memory, shuffle=True, persistent_workers=True)
    
    # get feats_p
    if args.use_new_feats_p:
        feats_p = get_p(args, train_loader_0, model_depth).to(device)
        torch.save(feats_p, args.feats_p_path)
    else:
        feats_p = torch.load(args.feats_p_path).to(device)
    
    
    model = getattr(importlib.import_module('models'), args.model)(args, feats_p=feats_p, p_mask=0b11).to(device)
    # freeze params
    for name, param in model.named_parameters():
        if 'rn' not in name and 'adapter' not in name and 'cross_attention' not in name and 'linear_point' not in name:
            param.requires_grad_(False)
    prompts_feats = prompts_feats.to(device).detach()
    
    
    """ ================Main Training============== """
    # train task_0
    io.cprint("="*100, "Task 0", "="*100)
    optimizer_0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr0, weight_decay=1e-4)
    for t in range(args.epoch0):
        io.cprint(f"---------------Epoch {t+1}-------------------")
        runtime_stat['epoch'] = t + 1
        train_loop(train_loader_0, model, prompts_feats[:num_cat_0], optimizer_0, loss_fn, runtime_stat)
    test_loop(test_loader_0, model, prompts_feats[:num_cat_0], loss_fn, runtime_stat)
    
    # incremental tasks`    `
    for task_id in range(1, session_maker.tot_session()):
        io.cprint("="*100, f"Task {task_id}", "="*100)
        runtime_stat['task_id'] = task_id
        
        if args.use_FSCIL3D_dataset:
            dataloader_i = dataloader_tot.get(task_id, 'training')
            train_loader_i = dataloader_i[task_id]['train']
            test_loader_i = dataloader_i[task_id]['test']
            num_cat_i = len_cls[task_id]
        else:   
            dataset_train_i, dataset_test_i = session_maker.make_session(session_id=task_id, update_memory=args.memory_shot)
            num_cat_i = dataset_test_i.get_cat_num()
            train_loader_i = torch.utils.data.DataLoader(dataset_train_i, batch_size=args.batch_size, num_workers=args.workers,
                                                            pin_memory=args.pin_memory, shuffle=True, persistent_workers=True)
            test_loader_i = torch.utils.data.DataLoader(dataset_test_i, batch_size=args.batch_size, num_workers=args.workers,
                                                        pin_memory=args.pin_memory, shuffle=True, persistent_workers=True)
        
        optimizer_i = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lri, weight_decay=1e-4)
        for t in range(args.epochi):
            io.cprint(f"-------------Epoch {t+1}--------------")
            runtime_stat['epoch'] = t + 1
            train_loop(train_loader_i, model, prompts_feats[:num_cat_i], optimizer_i, loss_fn, runtime_stat)
        test_loop(test_loader_i, model, prompts_feats[:num_cat_i], loss_fn, runtime_stat)
        

if __name__ == "__main__":
    feats_p = torch.load(args.feats_p_path).to(device)
    print(feats_p.shape)
    io.cprint("Device:", device)
    train()

