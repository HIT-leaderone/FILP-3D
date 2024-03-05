from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class PointNetfeatt(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, centroids=None, sim=None):
        super(PointNetfeatt, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # print(self.kmeans.predict(x.transpose(2,1)), self.kmeans.predict(x.transpose(2,1))[0].shape)
        # sys.exit()
        x = self.kmeans.cluster_sim(x.transpose(2,1)) # batch_size x n_samples x n_clusters
        x = x.transpose(2,1) # batch_size x n_clusters x n_samples
        # x = self.bn4(x)
        # maxpooling for permutation invariance symmetric function
        # x = torch.max(x, 2, keepdim=True)[0] # nn.MaxPool1d(x.size(-1))(x) # torch.max(x, 2, keepdim=True)[0]
        x = nn.AvgPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        # x = self.bn4(x)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetfeatV2(PointNetfeatt):
    def __init__(self, global_feat = True, feature_transform = True):
        super(PointNetfeatV2, self).__init__(global_feat, feature_transform)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # batch number x feature vector size x number of points
        x = nn.AvgPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        #x = torch.max(matrix1024x1024, 2, keepdim=True)[0]
       # x = nn.Flatten(1)(x)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class get_encoder(nn.Module):
    def __init__(self, feature_transform=True, att_size=512):
        super().__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeatV2(global_feat=True, feature_transform=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, att_size)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(att_size)
        self.relu = nn.ReLU()

    def forward(self, x, old_att=None):
        x = torch.transpose(x, 1, 2)
        # print(x.shape)
        x, trans, trans_feat = self.feat(x)
        return x
    
    

class PointNetCls300(nn.Module):
    def __init__(self, feature_transform=True, feat_p=None):
        super(PointNetCls300, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeatV2(global_feat=True, feature_transform=feature_transform)
        self.feat_p = feat_p
        self.fc1 = nn.Linear(103, 256)
        self.fc2 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU() # LeakyReLU, ReLU
        # self.rn = RelationNetwork(in_dim=self.fc2.out_features*2)
        self.rn = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(), 
            nn.Linear(512, 1024),
            nn.LeakyReLU(), 
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, input, attribute):
        outputs = {} 
        x, trans, trans_feat = self.feat(input)
        x = x @ self.feat_p
        x = self.relu(self.bn1(self.fc1(x))) 
        x = self.relu(self.bn2(self.fc2(x)))
        outputs['feats'] = x
        outputs['trans_feat'] = trans_feat
        
        b, n = x.shape[0], attribute.shape[0]
        feat_dim = x.shape[1] + attribute.shape[1]

        sample_features_ext = attribute.unsqueeze(0).repeat(b,1,1)
        # print(sample_features_ext.shape) # torch.Size([1, 3, 2048])

        batch_features_ext = x.unsqueeze(0).repeat(n,1,1)
        # print(batch_features_ext.shape) # torch.Size([3, 1, 2048])
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        # print(batch_features_ext.shape) # torch.Size([1, 3, 2048])
        
        # concat att->features + actual_feature
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext),2)
        # print(relation_pairs.shape) # torch.Size([1, 3, 4096])
        relation_pairs = relation_pairs.view(-1, feat_dim)
        # print(relation_pairs.shape) # torch.Size([3, 4096])
        
        # get relation score
        relations = self.rn(relation_pairs)
        # print(relations.shape) # torch.Size([3, 1])
        relations = relations.view(-1,n)

        outputs['pred'] = relations
        return outputs


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.to(device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.to(device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.to(device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

def copy_parameters(model, pretrained, verbose=True):
    # ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3

    model_dict = model.state_dict()
    pretrained_dict = pretrained
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

    if verbose:
        print('=' * 27)
        print('Restored Params and Shapes:')
        for k, v in pretrained_dict.items():
            print(k, ': ', v.size())
        print('=' * 68)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def get_pointnet_encoder(num_channel=3):
    pointnet_cls = get_encoder(att_size=512)
    checkpoint = torch.load('pre_builts/pointnet/cls.pth')
    return copy_parameters(pointnet_cls, checkpoint, verbose=False)


