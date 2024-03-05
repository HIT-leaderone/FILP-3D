import torch
import numpy as np
import torchvision as tv

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def random_sample(pc, num):
    permutation = np.arange(np.size(pc, 0))
    np.random.shuffle(permutation)
    pc = pc[permutation[:num]]
    return pc

def default_pc_transform(pc):
    pc = random_sample(pc, min(1024, np.size(pc, 0)))
    pc = pc_normalize(pc)
    return pc