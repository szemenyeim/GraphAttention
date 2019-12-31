import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class GraphClassifier(nn.Module):
    def __init__(self,baseFeatNum, edgeFeatNum):
        super().__init__()
        self.baseFeatNum = baseFeatNum

        self.Features = SingleHeadedGraphAttention(12,1,baseFeatNum,baseFeatNum,edgeFeatNum)
        self.Classifier = nn.Linear(baseFeatNum*2,5)

    def forward(self, nodes, edges, distances, mask):
        f = self.Features(nodes,edges,distances,mask)
        out = self.Classifier(f)
        return out

class SingleHeadedGraphAttention(nn.Module):
    def __init__(self,nFeatures,eFeatures,QKDims,nValDims,eValDims):
        super().__init__()
        self.QKDims = QKDims
        self.nValDims = nValDims
        self.eValDims = eValDims
        self.nFeatures = nFeatures
        self.eFeatures = eFeatures

        self.coeff = 1.0/math.sqrt(QKDims)

        self.QNet = nn.Linear(nFeatures,QKDims)
        self.KNet = nn.Linear(nFeatures,QKDims)

        self.NVNet = nn.Linear(nFeatures,nValDims)
        self.EVNet = nn.Linear(eFeatures,eValDims)

        self.Trans = nn.Linear(nValDims+eValDims, nValDims*2)
        self.bn = nn.BatchNorm1d(nValDims*2)

    def forward(self, nodes, edges, distances, mask = None):

        nInputs = nodes.view([-1,self.nFeatures])
        eInputs = edges.view([-1,self.eFeatures])

        nShape = (nodes.shape[0],nodes.shape[1],-1)
        eShape = (edges.shape[0],edges.shape[1],edges.shape[2],-1)

        Q = self.QNet(nInputs).view(nShape)
        K = self.KNet(nInputs).view(nShape)

        NV = self.NVNet(nInputs).view(nShape).repeat([1,nodes.shape[1],1]).reshape(eShape)
        EV = self.EVNet(eInputs).view(eShape)

        V = torch.cat([NV,EV],dim=-1)

        distWeights = 1.0/(distances+1)
        QKWeights = torch.einsum('bik,bjk->bij',[Q,K])
        weights = distWeights*QKWeights*self.coeff

        if mask is not None:
            weights[mask] = float('-inf')

        weights = F.softmax(weights,dim=-1).unsqueeze(-1)

        #with torch.no_grad():
        weights2 = weights.clone()
        weights2[torch.isnan(weights)] = 0

        vals = (weights2*V).sum(-2)

        vals = self.Trans(torch.tanh(vals))
        valsshape = vals.shape
        vals = vals.view((-1,valsshape[-1]))
        vals = self.bn(vals)
        vals = vals.view(valsshape)

        return vals