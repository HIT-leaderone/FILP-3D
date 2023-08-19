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
import torch.nn as nn
from sklearn.metrics import classification_report

from copy import deepcopy
from datasets.datautil_3D_memory_incremental_shapenet_co3d import *

from utils import *
from session_settings import shapenet2modelnet, shapenet2co3d
from random import random
from datasets.CILdataset import *
from torch.nn.parallel import DataParallel

from models import FewShotCIL, focal_loss, FewShotCILwoRn, FewShotCILwoRn2, FewShotCILwPoint, get_pointnet_encoder, PointNetCls300


import sys, importlib
sys.path.append('models')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#################################################### TConfigurations ############################################
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device_id = [0, 1]
exp_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
"""
Arguments
"""
parser = argparse.ArgumentParser()

parser.add_argument('-a', '--exp_name', type=str, default='shape2model_fscil3d', help='name of experiment')
parser.add_argument('-w', '--workers', type=int, default=4, help='number of dataloader workers')
parser.add_argument('--parallel', type=bool, default=False, help='whether to use multiple gpus')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--pin_memory', type=bool, default=True, help='whether to use pinned memory')
parser.add_argument('-s', '--seed', type=int, default=42, help='seed')

parser.add_argument('-e0', '--epoch0', type=int, default=20, help='epochs of task_0')
parser.add_argument('-ei', '--epochi', type=int, default=50, help='epochs of task_i')
parser.add_argument('-p', '--use_new_feats_p', type=bool, default=False , help='whether to extract new feats_p')
parser.add_argument('-u', '--cluster_num', type=int, default=1024, help='number of clusters when extracting feats_p')
parser.add_argument('--batch_size_base', type=int, default=64, help='batchsize')
parser.add_argument('--batch_size_inc', type=int, default=16, help='batchsize')
parser.add_argument('-mem', '--use_memory', type=bool, default=True, help='whether to use memory in training data')
parser.add_argument('-memshot', '--memory_shot', type=int, default=1, help='maxshot per class in training memory')
parser.add_argument('-f', '--feats_p_path', type=str, default='./feats_p/f_feats_p.pth', help='path of feats_p')
parser.add_argument('-shot', '--fewshot', type=int, default=5, help='number of shots')

    


args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

"""main"""

# IO
io = EXIOStream('exp_results/' + exp_time + ':' + args.exp_name)

# print configs
io.cprint('Configurations:', args.__dict__)


# CLIP model
clip_model, _ = clip.load('ViT-B/32', device='cpu')


""" ↓↓↓↓↓session setup↓↓↓↓↓ """
session_maker = shapenet2modelnet()
id2name = session_maker.get_id2name()
io.cprint(session_maker.info())
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
        if label[i] == 0:
            colorplt = 'brown'
        elif label[i] == 5:
            colorplt = 'blue'
        elif label[i] == 11:
            colorplt = 'cyan'
        elif label[i] == 17:
            colorplt = 'green'
        elif label[i] == 19:
            colorplt = 'black'
        # elif label[i] == 23:
        #     colorplt = 'magenta'
        elif label[i] == 31:
            colorplt = 'red'
        elif label[i] == 45:
            colorplt = 'yellow'
        else: continue
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RelationLoss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, num_class=20):
        super(RelationLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.n = num_class
        self.refresh_meter()

    def refresh_meter(self):
        self.ce_loss_meter = AverageMeter()

    def feature_transform_regularizer(self, trans):
        d = trans.size()[1]
        # batchsize = trans.size()[0]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.to(device)
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
        return loss


    def forward(self, pred, target, trans_feat):
        one_hot_labels = F.one_hot(target, num_classes=self.n)
        ce_loss = F.binary_cross_entropy(pred, one_hot_labels.float())
        mat_diff_loss = self.feature_transform_regularizer(trans_feat)
        total_loss = mat_diff_loss*self.mat_diff_loss_scale #+ v2_loss#+ args.lamda1*(dist_ps+dist_fs)
        total_loss+=ce_loss
        self.ce_loss_meter.update(ce_loss)
        return total_loss

def get_fp(args, dataloader_0, model_depth):
    """ aggrate all image features to 'image_feats' """
    device = next(model_depth.parameters()).device
    feat_dim = 1024
    feats = torch.empty([0, feat_dim]).to(device)
    model_depth.eval()
    with torch.no_grad():
        for points, label in tqdm(dataloader_0):
            points = points.to(device)
            feat = model_depth(points)
            feats = torch.cat([feats, feat / feat.norm(dim=-1, keepdim=True)], dim=0) # [views*n_data, d_vec]
    # k-means for clustering
    print(feats.shape)
    kmeans = MinibatchKMeans(n_clusters=args.cluster_num, distance='cosine', init_mode='kmeans++').to(device)
    cluster_centers = kmeans.kmeanspp(feats.t()).float().t() # [num_cluster, d_vec]
    feats_p = svd_conversion(cluster_centers)
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
    model.train()
    for points, label in tqdm(dataloader):
        label = label.to(device)
        points = points.transpose(1, 2).to(device)
        
        # Compute prediction
        outputs = model(points, prompts_feats)
        outputs_student, trans_feat = outputs['pred'], outputs['trans_feat']
        
        # Compute loss
        loss = loss_fn(outputs_student, label, trans_feat)
        pred = torch.max(outputs_student, dim=1).indices
        
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

last = 0
    
def test_loop(dataloader, model, prompts_feats, loss_fn, stat):
    global last
    num_batches = len(dataloader)
    task_id = stat['task_id']
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
        for points, label in tqdm(dataloader):
            label = label.to(device)
            points = points.transpose(1, 2).to(device)
            
            # Compute prediction and loss
            outputs = model(points, prompts_feats)
            outputs_student, trans_feat, Encoder = outputs['pred'], outputs['trans_feat'], outputs['feats']
            # print(Encoder.shape)
            pred = torch.max(outputs_student, dim=1).indices
            
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
            Encoder = Encoder.cpu().detach().numpy()
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
    #result = classification_report(ans_list, predict_list, target_names=id2name[:num_cls])             
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
    last = num_cls

    


def train():
    """ ==================initiation==================== """
    # init models
    model_depth = get_pointnet_encoder().to(device)
    model = None
    
    
    # init prompts
    prompts = id2name
    prompts = clip.tokenize(prompts)
    prompts = clip_model.encode_text(prompts)
    prompts_feats = (prompts / prompts.norm(dim=-1, keepdim=True)).to(device)
    

    
    
    # init task_0 
    runtime_stat = {'task_id':0, 'epoch':0}
    dataset_train_0, dataset_test_0 = session_maker.make_session(session_id=0, update_memory=args.memory_shot)
    num_cat_0 = dataset_test_0.get_cat_num()
    train_loader_0 = torch.utils.data.DataLoader(dataset_train_0, batch_size=args.batch_size_base, num_workers=args.workers,
                                                    pin_memory=args.pin_memory, shuffle=True)
    test_loader_0 = torch.utils.data.DataLoader(dataset_test_0, batch_size=args.batch_size_base, num_workers=args.workers,
                                                    pin_memory=args.pin_memory, shuffle=True)
    
    # get feats_p
    if args.use_new_feats_p:
        feats_p = get_fp(args, train_loader_0, model_depth).to(device)
        torch.save(feats_p, args.feats_p_path)
        print(feats_p.shape)
    else:
        feats_p = torch.load(args.feats_p_path).to(device)
    
    
    model = PointNetCls300(feat_p=feats_p).to(device)
    model.feat = model_depth.feat
    prompts_feats = prompts_feats.to(device).detach()
    
    
    """ ================Main Training============== """
    # train task_0
    io.cprint("="*100, "Task 0", "="*100)
    optimizer_0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)
    loss_fn = RelationLoss(mat_diff_loss_scale=0.001, num_class=num_cat_0).to(device)
    for t in range(args.epoch0):
        io.cprint(f"---------------Epoch {t+1}-------------------")
        runtime_stat['epoch'] = t + 1
        train_loop(train_loader_0, model, prompts_feats[:num_cat_0], optimizer_0, loss_fn, runtime_stat)
    test_loop(test_loader_0, model, prompts_feats[:num_cat_0], loss_fn, runtime_stat)
    
    
    # freeze params
    for name, param in model.named_parameters():
        if 'rn' not in name:
            param.requires_grad_(False)
    
    # incremental tasks
    for task_id in range(1, session_maker.tot_session()):
        io.cprint("="*100, f"Task {task_id}", "="*100)
        runtime_stat['task_id'] = task_id
        
        dataset_train_i, dataset_test_i = session_maker.make_session(session_id=task_id, update_memory=args.memory_shot)
        num_cat_i = dataset_test_i.get_cat_num()
        train_loader_i = torch.utils.data.DataLoader(dataset_train_i, batch_size=args.batch_size_inc, num_workers=args.workers,
                                                        pin_memory=args.pin_memory, shuffle=True)
        test_loader_i = torch.utils.data.DataLoader(dataset_test_i, batch_size=args.batch_size_base, num_workers=args.workers,
                                                        pin_memory=args.pin_memory, shuffle=True)
        
        optimizer_i = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=(1e-4+1e-5)/2, weight_decay=(1e-6+1e-4)/2)
        loss_fn = RelationLoss(mat_diff_loss_scale=0.001, num_class=num_cat_i).to(device)
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

