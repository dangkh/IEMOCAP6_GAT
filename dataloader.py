import numpy as np
import dgl
import torch
import os
from dgl.data import DGLDataset
from torch.utils.data import Dataset
import random
from ultis import *
import pickle

import glob
import tqdm
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

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
        

def read_data(label_path, feature_root):

    ## gain (names, speakers)
    names = []
    videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, testVid = pickle.load(open(label_path, "rb"), encoding='latin1')
    for ii, vid in enumerate(videoIDs):
        uids_video = videoIDs[vid]
        names.extend(uids_video)

    ## (names, speakers) => features
    features = []
    feature_dim = -1
    for ii, name in enumerate(names):
        feature = []
        feature_path = os.path.join(feature_root, name+'.npy')
        feature_dir = os.path.join(feature_root, name)
        if os.path.exists(feature_path):
            single_feature = np.load(feature_path)
            single_feature = single_feature.squeeze() # [Dim, ] or [Time, Dim]
            feature.append(single_feature)
            feature_dim = max(feature_dim, single_feature.shape[-1])
        else: ## exists dir, faces
            facenames = os.listdir(feature_dir)
            for facename in sorted(facenames):
                facefeat = np.load(os.path.join(feature_dir, facename))
                feature_dim = max(feature_dim, facefeat.shape[-1])
                feature.append(facefeat)
        # sequeeze features
        single_feature = np.array(feature).squeeze()
        if len(single_feature) == 0:
            single_feature = np.zeros((feature_dim, ))
        elif len(single_feature.shape) == 2:
            single_feature = np.mean(single_feature, axis=0)
        features.append(single_feature)

    ## save (names, features)
    print (f'Input feature {os.path.basename(feature_root)} ===> dim is {feature_dim}; No. sample is {len(names)}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    name2feats = {}
    for ii in range(len(names)):
        name2feats[names[ii]] = features[ii]

    return name2feats, feature_dim


class IEMOCAP6DGL_GCNET(DGLDataset):
    def __init__(self, trainVids, videoIDs, videoLabels, name2audio, name2text, name2video, missing):
        
        self.trainVids = trainVids
        self.videoIDs = videoIDs
        self.videoLabels = videoLabels
        self.missing = missing
        self.name2audio, self.name2text, self.name2video, = name2audio, name2text, name2video
        self.listMask = []
        self.maxSize = 120
        randSTR = random.randint(0, 1000)
        self.listNumNode = []
        for ii in range(len(self.trainVids)):
            name = self.trainVids[ii]
            self.listNumNode.append(len(self.videoIDs[name]))

        tmpLb = []
        for ii, v in enumerate(videoLabels):
            tmpLb.extend(videoLabels[v])
        self.out_size = len(np.unique(np.asarray(tmpLb)))

        missingPath = f'./mmask/missing_{self.missing}_rand_{randSTR}.npy'
        if os.path.isfile(missingPath):
            mask = np.load(missingPath, allow_pickle=True)
            currentUt = 0
            for idx, numNode in enumerate(self.listNumNode):
                self.listMask.append(mask[:,currentUt:currentUt+numNode])
                currentUt += numNode
        else:
            for idx, numNode in enumerate(self.listNumNode):
                mask = genMissMultiModal((3, numNode), self.missing)
                self.listMask.append(mask)
            np.save(missingPath, np.hstack(self.listMask))
        super().__init__(name='dataset_DGL')


    def __getitem__(self, index):
        name = self.trainVids[index]
        textf = []
        audiof = []
        visionf = []
        for i, vid in enumerate(self.videoIDs[name]):
            textf.append(np.copy(self.name2text[vid]))
            audiof.append(np.copy(self.name2audio[vid]))
            visionf.append(np.copy(self.name2video[vid]))
        text = np.vstack(textf)
        audio = np.vstack(audiof)
        vision = np.vstack(visionf)
        meanText = np.mean(text)
        meanAudio = np.mean(audio)
        meanVision = np.mean(vision)
        numNode = len(text)
        missingMask = self.listMask[index]
        for ii in range(numNode):
            if missingMask[0][ii] == 1:
                text[ii] = meanText
            if missingMask[1][ii] == 1:
                audio[ii] = meanAudio
            if missingMask[2][ii] == 1:
                vision[ii] = meanVision

        labels = np.asarray(self.videoLabels[name])
        src = []
        dst = []

        for node in range(numNode):
            for nodeAdj in range(node, numNode+1):
                src.append(node)
                dst.append(nodeAdj)
        outSize = self.maxSize

        for ii in range(numNode, outSize):
            src.append(ii)
            dst.append(ii)

        def compensation(features, size):
            shape = features.shape
            compensationF = torch.zeros(size-shape[0], shape[1])
            features = torch.from_numpy(features)
            features = torch.vstack((features, compensationF))
            return features

        text = compensation(text, outSize)
        audio = compensation(audio, outSize)
        vision = compensation(vision, outSize)
        
        compensation = torch.ones(self.maxSize-numNode)*self.out_size
        labels = torch.from_numpy(labels)
        labels = torch.hstack((labels, compensation))


        g = dgl.graph((src, dst))
        g.ndata["text"] = text.to(torch.float64)
        g.ndata["audio"] = audio.to(torch.float64)
        g.ndata["vision"] = vision.to(torch.float64)
        g.ndata["label"] = labels.to(torch.float64)
        return g, labels

    def __len__(self):
        return len(self.trainVids) 


class Iemocap6_Gcnet_Dataset():

    def __init__(self, path = './IEMOCAP/IEMOCAP_features_raw_6way.pkl', missing = 0, info = None):
        super(Iemocap6_Gcnet_Dataset, self).__init__()
        self.missing = missing
        self.path = path
        self.info = info
        self.process()

    def process(self):
        videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, testVid = pickle.load(open(self.path, "rb"), encoding='latin1')
        self.trainVids = sorted(trainVid)
        self.testVids = sorted(testVid)

        tmpLb = []
        for ii, v in enumerate(videoLabels):
            tmpLb.extend(videoLabels[v])


        name2audio, adim = read_data(self.path, f'./IEMOCAP/features/wav2vec-large-c-UTT')
        name2text, tdim = read_data(self.path, f'./IEMOCAP/features/deberta-large-4-UTT')
        name2video, vdim = read_data(self.path, f'./IEMOCAP/features/manet_UTT')

        self.trainSet = IEMOCAP6DGL_GCNET(self.trainVids, videoIDs, videoLabels, name2audio, name2text, name2video, self.missing)
        self.testSet = IEMOCAP6DGL_GCNET(self.testVids, videoIDs, videoLabels, name2audio, name2text, name2video, self.missing)

        self.out_size = len(np.unique(np.asarray(tmpLb)))


