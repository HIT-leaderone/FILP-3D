from copy import deepcopy
import torch.nn as nn
import torch
import clip
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from .adapter import SimplifiedAdapter
from .dpa import DPA
from render import Renderer, Selector
from utils import read_state_dict
import torchvision.utils as tv
from .dgcnn_cls import get_point_encoder

clip_model, _ = clip.load("ViT-B/32", device='cpu')
clip_out_dim = 512


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
        tv.save_image(imgs, './depco.png')
        img_feats = (self.pre_model(imgs) + self.ori_model(imgs)) * 1.0
        img_feats = img_feats.reshape(b, n, -1)
        img_feats = torch.sum(img_feats, dim=1) # [b, d_vec]
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        if self.feats_p is not None:
            img_feats = img_feats @ self.feats_p @ self.feats_p.t() @ prompts_feats.t()
        else:
            img_feats = img_feats @ prompts_feats.t()
        return img_feats
    
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias != None:
            m.bias.data.fill_(0)
class FewShotCIL(nn.Module):
    def __init__(self, args, feats_p=None, eval=False, p_mask=0b11):
        super().__init__()
        # feats_p
        self.feats_p = feats_p
        self.p_mask = p_mask # mult feats_p on: 10->dep & img | 01->prompts
        self.get_dims() # get dims of different layers according to feats_p and p_mask
        
        # rendering
        self.views = args.views
        self.selector = Selector(self.views, 0)
        self.renderer = Renderer(points_radius=0.02)
        
        # encoding -- freezed after pre-training
        self.pre_model = deepcopy(clip_model.visual)
        self.ori_model = deepcopy(clip_model.visual)
        if not eval and args.ckpt is not None:
            print('loading from %s' % args.ckpt)
            self.pre_model.load_state_dict(read_state_dict(args.ckpt))
        
        # adapter -- freezed after task_0
        self.adapter1 = SimplifiedAdapter(num_views=args.views, in_features=self.dim_img)
        self.adapter2 = SimplifiedAdapter(num_views=args.views, in_features=self.dim_img)
        
        # Relation Module -- learnable
        self.rn = nn.Sequential(
            nn.Linear(self.dim_img + self.dim_prompt, 300),
            nn.LeakyReLU(), 
            nn.Linear(300, 600),
            nn.LeakyReLU(), 
            nn.Linear(600, 1),
        )
        self.rn.apply(init_weights)
    
    def get_dims(self):
        self.dim_img = self.dim_prompt = clip_out_dim
        if self.feats_p is not None:
            self.dim_feats_p = self.feats_p.size(1)
            if self.p_mask & 0b10:
                self.dim_img = self.dim_feats_p
            if self.p_mask & 0b01:
                self.dim_prompt = self.dim_feats_p
    
    def forward(self, points, prompts_feats):
        num_cls = prompts_feats.size(0)
        
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views, rot=False)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        # tv.save_image(imgs, './depco.png')
        img_feat1 = self.pre_model(imgs)
        img_feat2 = self.ori_model(imgs)
        
        if self.feats_p is not None:
            if self.p_mask & 0b10:
                img_feat1 = img_feat1 @ self.feats_p
                img_feat2 = img_feat2 @ self.feats_p
            if self.p_mask & 0b01:
                prompts_feats = prompts_feats @ self.feats_p # [b, n_cls, d_vec_p]
        
        img_feat1 = self.adapter1(img_feat1)
        img_feat2 = self.adapter2(img_feat2)
        
        img_feats = (img_feat1 + img_feat2) * 0.5
        
        prompts_feats = prompts_feats.unsqueeze(0).repeat(img_feats.size(0), 1 ,1) # [n_cls, d_vec_p] -> [b, n_cls, d_vec_p]
        img_feats = img_feats.unsqueeze(1).repeat(1, num_cls, 1) # [b, d_vec_i] -> [b, n_cls, d_vec_i]
        
        img_prom_pairs = torch.cat([prompts_feats, img_feats], dim=2) # [b, n_cls, d_dev_i + d_vec_p]
        img_prom_pairs.reshape(-1, self.dim_img + self.dim_prompt)
        
        logits = self.rn(img_prom_pairs)
        logits.reshape(-1, num_cls)
        
        return logits
    
class FewShotCILwoRn(FewShotCIL):
    def __init__(self, args, feats_p=None, eval=False, p_mask=0b11):
        assert p_mask in [0b11, 0b00]
        super().__init__(args, feats_p=feats_p, eval=eval, p_mask=p_mask)
        
    def forward(self, points, prompts_feats):
        num_cls = prompts_feats.size(0)
        
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views, rot=False)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        # tv.save_image(imgs, './depco.png')
        img_feat1 = self.pre_model(imgs)
        img_feat2 = self.ori_model(imgs)
        
        if self.feats_p is not None and self.p_mask == 0b11:
            img_feat1 = img_feat1 @ self.feats_p
            img_feat2 = img_feat2 @ self.feats_p
            prompts_feats = prompts_feats @ self.feats_p # [b, n_cls, d_vec]
        
        img_feat1 = self.adapter1(img_feat1)
        img_feat2 = self.adapter2(img_feat2)
        
        img_feats = (img_feat1 + img_feat2) * 0.5
        #img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True) # [b, d_vec]
        
        prompts_feats = prompts_feats.unsqueeze(0).repeat(img_feats.size(0), 1 ,1) # [n_cls, d_vec] -> [b, n_cls, d_vec]
        img_feats = img_feats.unsqueeze(2) # [b, d_vec] -> [b, d_vec, 1]
        
        logits = torch.bmm(prompts_feats, img_feats) # [b, n_cls]
        
        return logits

class FewShotCILwoRn2(FewShotCIL):
    """
        featp after adapter
    """
    def __init__(self, args, feats_p=None, eval=False, p_mask=0b11):
        p_mask = 0b11
        super().__init__(args, feats_p=feats_p, eval=eval, p_mask=p_mask)
        
    def forward(self, points, prompts_feats):
        num_cls = prompts_feats.size(0)
        
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views, rot=False)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        # tv.save_image(imgs, './depco.png')
        img_feat1 = self.adapter1(self.pre_model(imgs))
        img_feat2 = self.adapter2(self.ori_model(imgs))
        img_feats = (img_feat1 + img_feat2) * 0.5
        
        if self.feats_p is not None and self.p_mask == 0b11:
            img_feats = img_feats @ self.feats_p
            prompts_feats = prompts_feats @ self.feats_p # [b, n_cls, d_vec]
        
        # img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True) # [b, d_vec]
        # prompts_feats = prompts_feats / prompts_feats.norm(dim=-1, keepdim=True)
        
        prompts_feats = prompts_feats.unsqueeze(0).repeat(img_feats.size(0), 1 ,1) # [n_cls, d_vec] -> [b, n_cls, d_vec]
        img_feats = img_feats.unsqueeze(2) # [b, d_vec] -> [b, d_vec, 1]
        
        logits = torch.bmm(prompts_feats, img_feats) # [b, n_cls, 1]
        
        return logits
        
    
    def get_dims(self):
        self.dim_img = self.dim_prompt = clip_out_dim

   
class FewShotCILwPoint(nn.Module):
    def __init__(self, args, feats_p=None, eval=False, p_mask=0b11):
        assert p_mask in [0b11, 0b00]
        super().__init__()
        # feats_p
        self.feats_p = feats_p
        self.p_mask = p_mask # mult feats_p on: 10->dep & img | 01->prompts
        self.get_dims() # get dims of different layers according to feats_p and p_mask
        
        # rendering
        self.views = args.views
        self.selector = Selector(self.views, 0)
        self.renderer = Renderer(points_radius=0.02)
        
        # encoding -- freezed after pre-training
        self.pre_model = deepcopy(clip_model.visual)
        self.ori_model = deepcopy(clip_model.visual)
        if not eval and args.ckpt is not None:
            print('loading from %s' % args.ckpt)
            self.pre_model.load_state_dict(read_state_dict(args.ckpt))
    
        self.point_model = get_point_encoder(args)
        self.linear_point = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.dim_img),
            nn.BatchNorm1d(self.dim_img)
        )
        
        # adapter -- freezed after task_0
        self.adapter1 = SimplifiedAdapter(num_views=args.views, in_features=self.dim_img)
        self.adapter2 = SimplifiedAdapter(num_views=args.views, in_features=self.dim_img)
        
    def forward(self, points, prompts_feats):
        num_cls = prompts_feats.size(0)
        
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views, rot=False)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        # tv.save_image(imgs, './depco.png')
        img_feat1 = self.pre_model(imgs)
        img_feat2 = self.ori_model(imgs)
        
        points = torch.transpose(points, 1, 2)
        point_feat = self.point_model(points)
        point_feat = self.linear_point(point_feat)
        point_feat = torch.repeat_interleave(point_feat, repeats=self.views, dim=0)
        
        if self.feats_p is not None and self.p_mask == 0b11:
            img_feat1 = img_feat1 @ self.feats_p
            img_feat2 = img_feat2 @ self.feats_p
            prompts_feats = prompts_feats @ self.feats_p # [b, n_cls, d_vec]
        
        img_point_feat1 = self.adapter1(img_feat1 + point_feat)
        img_point_feat2 = self.adapter2(img_feat2 + point_feat)
        
        img_point_feats = (img_point_feat1 + img_point_feat2) * 0.5
        
        prompts_feats = prompts_feats.unsqueeze(0).repeat(img_point_feats.size(0), 1 ,1) # [n_cls, d_vec] -> [b, n_cls, d_vec]
        img_point_feats = img_point_feats.unsqueeze(2) # [b, d_vec] -> [b, d_vec, 1]
        
        logits = torch.bmm(prompts_feats, img_point_feats) # [b, n_cls]
        
        return logits
    
    def get_dims(self):
        self.dim_img = self.dim_prompt = clip_out_dim
        if self.feats_p is not None:
            self.dim_feats_p = self.feats_p.size(1)
            if self.p_mask & 0b10:
                self.dim_img = self.dim_feats_p
            if self.p_mask & 0b01:
                self.dim_prompt = self.dim_feats_p

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        alpha = [alpha] * num_classes
        self.size_average = size_average
        self.alpha = torch.Tensor(alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        # preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds.squeeze(), dim=1).gather(dim=1, index=labels) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax\

        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        # loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss