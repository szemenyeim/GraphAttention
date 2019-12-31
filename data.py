import torch
import numpy as np
import torch.utils.data as data
import glob
import itertools
import os.path as osp

flatten = lambda l: [item for sublist in l for item in sublist]

def stack_padding(l):
    return np.column_stack(list(itertools.zip_longest(*l, fillvalue=0)))

def stack_matrix_padding(l,size,nodeCnts):

    maxRow = np.sqrt(l.shape[1]).astype('int')
    featCnt = l.shape[-1]

    l = [elem[:cnt*cnt].reshape((cnt,cnt,-1)) for elem,cnt in zip(l,nodeCnts)]
    l = np.stack([np.pad(elem,(0,maxRow-cnt), mode='constant')[...,:featCnt] for elem,cnt in zip(l,nodeCnts)])[:,:size,:size,:]

    return l

class GraphDataSet(data.Dataset):
    def __init__(self, path, nFeatures = 12, eFeatures = 1, scene = False):
        super().__init__()

        self.path = path

        if scene:
            nodeFiles = ["sceneNodes.csv"]
            edgeFiles = ["sceneEdges.csv"]
            labels = np.genfromtxt(osp.join(path,"sceneLabels.csv"),delimiter=',')
            self.M = torch.tensor(labels == 0).bool()
            self.L = torch.tensor(labels-1).long()
            self.L[self.L < 0] = 1
        else:
            nodeFiles = sorted(glob.glob1(path,"*nodes.csv"))[:-1]
            edgeFiles = sorted(glob.glob1(path,"*edges.csv"))[:-1]

        nodes = [np.genfromtxt(osp.join(path,file),delimiter=',') for file in nodeFiles]
        edges = [np.genfromtxt(osp.join(path,file),delimiter=',') for file in edgeFiles]
        labels = [nodes[i].shape[0] for i in range(len(nodes))]

        nodes = flatten(nodes)
        nodes = stack_padding(nodes)

        self.N = torch.tensor(nodes).float()
        self.N = self.N.view((self.N.shape[0],-1,nFeatures))

        if not scene:
            self.L = torch.cat([torch.ones((label,self.N.shape[1]))*i for i,label in enumerate(labels)]).long()
            self.M = torch.logical_not(self.N.sum(2).bool())

        nodeCnts = (self.N.shape[1]-self.M.sum(1).int()).numpy()

        edges = np.stack(flatten(edges))
        edges = edges.reshape((edges.shape[0],-1,eFeatures+1))
        edges = stack_matrix_padding(edges,nodes.shape[1]//nFeatures, nodeCnts)

        self.E = torch.tensor(edges[...,1]).float()
        self.D = torch.tensor(edges[...,0]).float()

    def __len__(self):
        return self.N.shape[0]

    def __getitem__(self, item):
        return (self.N[item],self.E[item],self.D[item],self.M[item]),self.L[item]
