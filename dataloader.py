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
    def __init__(self, Vid, videoLabels, normed_node_features, node_label, missing, modals):
        self.Vid = Vid
        self.videoLabels = videoLabels
        self.node_features = normed_node_features
        self.node_label = node_label
        self.vid2Idx = []
        for idx, x in enumerate(self.Vid):
            self.vid2Idx.append(x)
        self.startNode = [0]
        for idx, x in enumerate(self.Vid):
            self.startNode.append(self.startNode[-1]+len(self.videoLabels[x]))
        self.maxNode = 120
        self.missing = missing
        self.modals = modals
        self.listMask = []
        randSTR = random.randint(0, 1000)
        missingPath = f'./missingMask/{modals}_modals_missing_{self.missing}_rand_{randSTR}.npy'
        if os.path.isfile(missingPath):
            mask = np.load(missingPath, allow_pickle=True)
            currentUt = 0
            for idx, x in enumerate(self.Vid):
                numUtterance = len(self.videoLabels[x])
                self.listMask.append(mask[currentUt:currentUt+numUtterance])
        else:
            for idx, x in enumerate(self.Vid):
                numUtterance = len(self.videoLabels[x])
                mask = genMissMultiModal((len(modals), numUtterance), self.missing)
                matSize = mask.shape
                compensation = np.zeros((1,matSize[-1]))
                if modals == 'av':
                    mask = np.vstack([compensation, mask])
                elif modals == 'al':
                    mask = np.vstack([mask[0,:], mask[1,:], compensation])
                elif modals == 'vl':
                    mask = np.vstack([mask[0,:], compensation, mask[1,:]])
                self.listMask.append(mask)
            np.save(missingPath, np.hstack(self.listMask))
        super().__init__(name='dataset_DGL')


    def __getitem__(self, idx):
        features  = self.node_features[self.startNode[idx]: self.startNode[idx+1]]
        mask = self.listMask[idx]
        tt, aa, vv  = 100, 442, 2024
        for ii in range(len(features)):
            currentFeatures = features[ii]
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

        # select feature corresponding to choosed modals

        if self.modals == 'vl':
            featuresL = features[:, :tt]        
            featuresV = features[:, aa:vv]
            features = np.hstack([featuresL, featuresV, features[:, 2024:]])
        elif self.modals == 'al':
            featuresL = features[:, :tt]        
            featuresA = features[:, tt:aa]
            features = np.hstack([featuresL, featuresA, features[:, 2024:]])
        elif self.modals == 'av':
            featuresA = features[:, tt:aa]        
            featuresV = features[:, aa:vv]
            features = np.hstack([featuresA, featuresV, features[:, 2024:]])


        # generate graph

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
        return len(self.Vid)

class emotionDataset():
    def __init__(self, path = './IEMOCAP_features/IEMOCAP_features.pkl', missing = 0, modals = 'avl'):
        super(emotionDataset, self).__init__()
        self.missing = missing
        self.path = path
        self.modals = modals
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



        self.trainSet = MeldDGL(self.trainVid, self.videoLabels, node_featuresTrain, node_labelTrain, self.missing, self.modals)
        self.testSet = MeldDGL(self.testVid, self.videoLabels, node_featuresTest, node_labelTest, self.missing, self.modals)
        self.out_size = len(np.unique(node_labelTrain))+1
        if len(self.modals) == 3:
            self.in_size = node_features.shape[-1]
        else:
            tt, aa, vv = 100, 342, 1582
            if self.modals == 'vl':
                self.in_size = tt + vv + 5
            elif self.modals == 'al':
                self.in_size = tt + aa + 5
            elif self.modals == 'av':
                self.in_size = aa + vv + 5
            