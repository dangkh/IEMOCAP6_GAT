import numpy as np
import dgl
import torch
import os
from dgl.data import DGLDataset
from torch.utils.data import Dataset
import random
from ultis import *
import pickle

class MeldDGL(DGLDataset):
    def __init__(self, Vid, videoLabels, normed_node_features, node_label, missing, training = False):
        self.Vid = Vid
        self.videoLabels = videoLabels
        self.node_features = normed_node_features
        self.node_label = node_label
        self.training = training
        self.mergeGraph = 8
        self.vid2Idx = []
        for idx, x in enumerate(self.Vid):
            self.vid2Idx.append(x)
        self.startNode = [0]
        for idx, x in enumerate(self.Vid):
            self.startNode.append(self.startNode[-1]+len(self.videoLabels[x]))
        self.maxNode = 900
        self.missing = missing
        self.listMask = []
        randSTR = random.randint(0, 1000)
        missingPath = f'missing_{self.missing}_rand_{randSTR}.npy'
        if os.path.isfile(missingPath):
            mask = np.load(missingPath, allow_pickle=True)
            currentUt = 0
            for idx, x in enumerate(self.Vid):
                numUtterance = len(self.videoLabels[x])
                self.listMask.append(mask[currentUt:currentUt+numUtterance])
        else:
            for idx, x in enumerate(self.Vid):
                numUtterance = len(self.videoLabels[x])
                self.listMask.append(genMissMultiModal((3, numUtterance), self.missing))
            np.save(missingPath, np.hstack(self.listMask))
        super().__init__(name='dataset_DGL')


    def __getitem__(self, idx):
        if self.training:
            start = min(idx * self.mergeGraph, len(self.Vid))
            end = min((idx+1) * self.mergeGraph, len(self.Vid))
            listGraph = []
            totalNode = 0
            for graphidx in range(start, end):
                features  = self.node_features[self.startNode[idx]: self.startNode[idx+1]]
                mask = self.listMask[idx]
                for ii in range(len(features)):
                    currentFeatures = features[ii]
                    tt, aa, vv  = 100, 442, 2024
                    text = currentFeatures[:tt]
                    audio = currentFeatures[tt: aa]
                    video = currentFeatures[aa: vv]
                    if mask[0][ii] == 1:
                        text[:] = 0
                    if mask[1][ii] == 1:
                        audio[:] = 0
                    if mask[2][ii] == 1:
                        video[:] = 0
                numNode = len(features)
                totalNode += numNode
                src = []
                dst = []

                for node in range(numNode):
                    for nodeAdj in range(numNode):
                        src.append(node)
                        dst.append(nodeAdj)
                features = torch.from_numpy(features).float()
                labels = np.asarray(self.videoLabels[self.vid2Idx[idx]])
                labels = torch.from_numpy(labels).float()
                g = dgl.graph((src, dst))
                g.ndata["x"] = features
                g.ndata["label"] = labels
                listGraph.append(g)

            numNode = totalNode
            compensationF = torch.zeros(self.maxNode-numNode, self.node_features.shape[-1])
            compensation = torch.ones(self.maxNode-numNode)*6
            src = []
            dst = []
            for ii in range(self.maxNode-numNode):
                src.append(ii)
                dst.append(ii)
            g = dgl.graph((src, dst))
            g.ndata["x"] = compensationF.float()
            g.ndata["label"] = compensation.float()
            listGraph.append(g)
            returnGraph = dgl.batch(listGraph)
            return returnGraph, returnGraph.ndata["x"], returnGraph.ndata["label"]
        else:
            features  = self.node_features[self.startNode[idx]: self.startNode[idx+1]]
            mask = self.listMask[idx]
            for ii in range(len(features)):
                currentFeatures = features[ii]
                tt, aa, vv  = 100, 442, 2024
                text = currentFeatures[:tt]
                audio = currentFeatures[tt: aa]
                video = currentFeatures[aa: vv]
                if mask[0][ii] == 1:
                    text[:] = 0
                if mask[1][ii] == 1:
                    audio[:] = 0
                if mask[2][ii] == 1:
                    video[:] = 0
            numNode = len(features)
            src = []
            dst = []

            for node in range(numNode):
                for nodeAdj in range(numNode):
                    src.append(node)
                    dst.append(nodeAdj)
            for ii in range(numNode, self.maxNode):
                src.append(ii)
                dst.append(ii)

            compensationF = torch.zeros(self.maxNode-numNode, features.shape[-1])
            compensation = torch.ones(self.maxNode-numNode)*6
            features = torch.from_numpy(features)
            features = torch.vstack((features, compensationF))
            labels = np.asarray(self.videoLabels[self.vid2Idx[idx]])
            labels = torch.from_numpy(labels)
            labels = torch.hstack((labels, compensation))
            g = dgl.graph((src, dst))
            g.ndata["x"] = features
            g.ndata["label"] = labels
            return g, features, labels

    def __len__(self):
        return ( len(self.Vid) // self.mergeGraph ) + 1



def missingParam(percent):
    al, be , ga = 0, 0, 0
    for aa in range(1, 200):
        for bb in range(1, 200):
            for gg in range(200):
                if (aa+bb+gg) != 0:
                    if abs(((bb*3 + gg * 6) * 100.0 / (aa*9 + bb*9 + gg*9)) - percent) <= 1.0:
                        return aa, bb, gg
    return al, be, ga


def genMissMultiModal(matSize, percent):
    index = (percent-10) // 10
    types = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    missPercent = 0
    batch_size = 1
    if matSize[0] != len(types[0]):
        return None
    al, be, ga = missingParam(percent)
    errPecent = 1.7
    if matSize[-1] <= 10:
        errPecent = 5
    if matSize[-1] <= 3:
        errPecent = 20
    listMask = []
    masks = [np.asarray([[0, 0, 0]]), np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), np.asarray([[0, 1, 1], [1, 1, 0], [1, 0, 1]])]
    if percent > 60:
        masks = [np.asarray([[0, 0, 0]]), np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), np.asarray([[0, 1, 1], [1, 1, 0], [1, 0, 1]]*7)]
    for mask, num in ([0, al], [1, be], [2, ga]):
        if num > 0:
            listMask.append(np.repeat(masks[mask], num, axis = 0))
    missType = np.vstack(listMask)
    counter = 0
    while np.abs(missPercent - percent) > 1.0:
        mat = np.zeros((matSize[0], matSize[-1]))
        for ii in range(matSize[-1]):
            tmp = random.randint(0, len(missType)-1)
            mat[:,ii] = missType[tmp]
        missPercent = mat.sum() / (matSize[0] * matSize[-1]) * 100
        print(missPercent, errPecent, matSize[-1])
        if (np.abs(missPercent - percent) < errPecent) & (np.abs(missPercent - percent) > 0):
            return mat
    return np.zeros((matSize[0], matSize[-1]))

class emotionDataset():
    def __init__(self, path = './MELD_features/MELD_features.pkl', missing = 0):
        super(emotionDataset, self).__init__()
        self.missing = missing
        self.path = path
        self.process()

    def extractNode(self, x1, x2, x3, x4):
        text = np.asarray(x1)
        audio = np.asarray(x2)
        video = np.asarray(x3)
        speakers = torch.FloatTensor([[1]*5 if x=='M' else [0]*5 for x in x4])
        # 100, 342, 1582, 5
        # 600, 342, 300, 5
        output = np.hstack([text, audio, video, speakers])
        return output    


    def process(self):
        self.subIdTrain, self.subIdTest = [], []
        inputData = pickle.load(open(self.path, 'rb'), encoding='latin1')
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
            self.testVid = inputData
        
        
        numSubGraph = len(self.trainVid) + len(self.testVid)
        numNodeTrain = sum([len(self.videoText[x]) for x in self.trainVid])
        numNodeTest = sum([len(self.videoText[x]) for x in self.testVid])
        numberNode = numNodeTest + numNodeTrain

        node_featuresTrain = np.vstack([self.extractNode(self.videoText[x], self.videoVisual[x], \
            self.videoAudio[x], self.videoSpeakers[x]) for x in self.trainVid])
        node_featuresTest = np.vstack([self.extractNode(self.videoText[x], self.videoVisual[x], \
            self.videoAudio[x], self.videoSpeakers[x]) for x in self.testVid])
        node_features = np.vstack([node_featuresTrain, node_featuresTest])
        # feature normalization
        # node_featuresTrain = normMat(node_featuresTrain, node_featuresTrain)
        # node_featuresTest = normMat(node_featuresTest, node_featuresTest)

        node_labelTrain = np.hstack([np.asarray(self.videoLabels[x]) for x in self.trainVid])
        node_labelTest = np.hstack([np.asarray(self.videoLabels[x]) for x in self.testVid])
        # node_labels = np.hstack([node_labelTrain, node_labelTest])



        self.trainSet = MeldDGL(self.trainVid, self.videoLabels, node_featuresTrain, node_labelTrain, self.missing, True)
        self.testSet = MeldDGL(self.testVid, self.videoLabels, node_featuresTest, node_labelTest, self.missing, False)
        self.in_size = node_features.shape[-1]
        self.out_size = len(np.unique(node_labelTrain))+1
        