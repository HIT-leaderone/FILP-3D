from copy import deepcopy
import torch.nn as nn
import torch
import clip
import matplotlib.pyplot as plt
import numpy as np

from .adapter import SimplifiedAdapter
from render import Renderer, Selector
from utils import read_state_dict
import torchvision.utils as tv

clip_model, _ = clip.load("ViT-B/32", device='cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ZeroShotCIL(nn.Module):
    def __init__(self, args, feats_p=None, eval=False):
        super().__init__()
        self.views = args.views
        self.selector = Selector(self.views, 0)
        self.renderer = Renderer(points_radius=0.02)
        self.pre_model = deepcopy(clip_model.visual)
        self.ori_model = deepcopy(clip_model.visual)
        if not eval and args.ckpt is not None:
            print('loading from %s' % args.ckpt)
            self.pre_model.load_state_dict(read_state_dict(args.ckpt))
        self.feats_p = feats_p
    
    def forward(self, points, prompts_feats):
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views, rot=False)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        img_feats = (self.pre_model(imgs) + self.ori_model(imgs)) * 0.5
        img_feats = img_feats.reshape(b, n, -1)
        img_feats = torch.sum(img_feats, dim=1) # [b, d_vec]
        # img_feats = (img_feats / img_feats.norm(dim=-1, keepdim=True)).to(device)
        if self.feats_p is not None:
            img_feats = img_feats @ self.feats_p.t() @ self.feats_p @ prompts_feats.t()
            # img_feats = img_feats @ self.feats_p.t()
            # prompts_feats = prompts_feats @ self.feats_p.t()
            # norm_img_feats = (img_feats / img_feats.norm(dim=-1, keepdim=True)).to(device)
            # norm_prompts_feats = (prompts_feats / prompts_feats.norm(dim=-1, keepdim=True)).to(device)
            # img_feats=norm_img_feats@norm_prompts_feats.t()
        else:
            img_feats = img_feats @ prompts_feats.t()
        return img_feats

   
class MYCIL(nn.Module):
    def __init__(self, args,feats_p=None, eval=False,use_dpa=False,NMFmodel=None):
        super().__init__()
        self.views = args.views
        self.use_dpa = use_dpa
        self.selector = Selector(self.views, 0)
        self.renderer = Renderer(points_radius=0.02)
        self.pre_model = deepcopy(clip_model.visual)
        self.ori_model = deepcopy(clip_model.visual)
        if not eval and args.ckpt is not None:
            print('loading from %s' % args.ckpt)
            self.pre_model.load_state_dict(read_state_dict(args.ckpt))
        self.adapter1 = SimplifiedAdapter(num_views=args.views, in_features=512)
        self.adapter2 = SimplifiedAdapter(num_views=args.views, in_features=512)
        self.feats_p = feats_p
        self.NMFmodel = NMFmodel
    
    def forward(self, points,prompts_feats):
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views, rot=False)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        if self.use_dpa == False: 
            img_feats = self.adapter1(self.pre_model(imgs))
        else : img_feats = (self.adapter1(self.pre_model(imgs)) + self.adapter2(self.ori_model(imgs))) * 0.5   
        # img_feats = self.adapter1(img_feats)
        if self.feats_p is not None:
            img_feats = img_feats @ self.feats_p.t() @ self.feats_p @ prompts_feats.t()
        else:
            img_feats = img_feats @ prompts_feats.t()
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True) 
        return img_feats
    
class BaseLine(nn.Module):
    def __init__(self, args, eval=False):
        super().__init__()
        self.views = args.views
        self.selector = Selector(self.views, 0)
        self.renderer = Renderer(points_radius=0.02)
        self.pre_model = deepcopy(clip_model.visual)
        if not eval and args.ckpt is not None:
            print('loading from %s' % args.ckpt)
            self.pre_model.load_state_dict(read_state_dict(args.ckpt))
        self.adapter1 = SimplifiedAdapter(num_views=args.views, in_features=512)
    
    def forward(self, points):
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views, rot=False)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        img_feats = self.pre_model(imgs) 
        img_feats = self.adapter1(img_feats)
        # img_feats = img_feats.reshape(b, n, -1)
        # img_feats = torch.sum(img_feats, dim=1) # [b, d_vec]
        # print("size=:",img_feats.size(),points.size())
        # img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True) 
        return img_feats