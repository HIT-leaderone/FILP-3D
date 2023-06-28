import numpy as np
import os, sys, time, copy, gc, argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from path import Path
import torchmetrics
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import clip

from copy import deepcopy
from datasets.datautil_3D_memory_incremental_shapenet_co3d import *
from configs.shapenet_co3d_info import label2name, len_cls

from render.selector import Selector
from render.render import Renderer
from utils import *

from torchpq.clustering import MinibatchKMeans
from models import ZeroShotCIL
#################################################### TConfigurations ############################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
Arguments
"""
parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config_file', type=str, default='./configs/ShapeNet_CO3D_incremental_config.yaml', help='path of config file')

input_arguments = parser.parse_args()

args = Argument(config_file = input_arguments.config_file)
print('Configurations:', args.__dict__)

start_iter = 0
amsgrad = True
eps = 1e-8

np.random.seed(args.seed)
torch.manual_seed(args.seed)

"""main"""

# Dataloader
dataloader_tot = DatasetGen(args, root=Path(args.dataset_path), fewshot=args.fewshot )

# CLIP model
clip_model, _ = clip.load('ViT-B/32', device='cpu')

def get_p(dataloader_1, model_depth, selector, render):
    # aggrate all image features to 'image_feats'
    clip_feat_dim = 512
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




def inference():
    # init models
    model_depth = deepcopy(clip_model.visual).to(device)
    model = None
    if args.ckpt is not None:
        print(args.ckpt)
        model_depth.load_state_dict(read_state_dict(args.ckpt))
    selector = Selector(args.views, 0).to(device)
    render = Renderer(points_per_pixel=1, points_radius=0.02).to(device)
    
    # init prompts
    prompts = label2name
    prompts = ['image or projection or sketch of a ' + prompts[i] for i in range(len(prompts))]
    print(len(prompts))
    prompts = clip.tokenize(prompts)
    prompts = clip_model.encode_text(prompts)
    prompts_feats = (prompts / prompts.norm(dim=-1, keepdim=True)).to(device)

    # metric
    all_total = 0
    all_correct_num = 0
    for task_id in range(0, args.ntasks):
        # get dataloader_i
        dataloader_i = dataloader_tot.get(task_id, 'training')
        train_loader_i = dataloader_i[task_id]['train']
        test_loader_i = dataloader_i[task_id]['test']
        
        # get feats_p
        if task_id == 0:
            if args.use_new_feats_p:
                feats_p = get_p(train_loader_i, model_depth, selector, render).to(device)
                torch.save(feats_p, args.feats_p_path)
                model = ZeroShotCIL(args, feats_p=feats_p).to(device)
            else:
                feats_p = torch.load(args.feats_p_path).to(device)
                model = ZeroShotCIL(args, feats_p=feats_p).to(device)
        
        # infer
        model.eval()
        with torch.no_grad():
            correct_num = 0
            total = 0
            for data in tqdm(test_loader_i):
                label = data['labels'].cpu()
                points = data['pointclouds'].to(device)
                logits = model(points, prompts_feats[0 if task_id == 0 else len_cls[0] :len_cls[task_id]])
                index = torch.max(logits, dim=1).indices + (0 if task_id == 0 else len_cls[0])
                #print(label)
                #print(index)
                masked_num = torch.sum((label< (0 if task_id == 0 else len_cls[0]))).item() # except shapenet
                
                correct_num += torch.sum(torch.eq(index.detach().cpu(), label)).item()
                total += len(label) - masked_num
            test_acc = correct_num / total
            print("Task_ID:", task_id, "test_ACC:", test_acc,"(",correct_num,"/",total,")")
            if task_id > 0:
                all_correct_num += correct_num
                all_total += total
    print("total acc on co3d:", all_correct_num / all_total,"(",all_correct_num,"/",all_total,")")

        
        
if __name__ == "__main__":
    print(device)
    inference()
