import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.nn import LabelPropagation
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dataloader import *
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
# torch.set_default_dtype(torch.float)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
from attentionModule import *
from dgl.nn import GraphConv, SumPooling, AvgPooling
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def checkMissing(data):
    if len(np.where(data==0)[0]) > 0:
        return True
    return False

class maskFilter(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        tt, aa, vv  = 64, 128, 192
        # self.testM = nn.Parameter(torch.rand(in_size, in_size))
        currentFeatures = np.asarray([0.0] * in_size)
        textMask = np.copy(currentFeatures)
        textMask[:tt] = 1.0
        audioMask = np.copy(currentFeatures)
        audioMask[tt: aa] = 1.0
        videoMask = np.copy(currentFeatures)
        videoMask[aa:] = 1.0
        self.textMask = torch.from_numpy(textMask) * torch.tensor(3.0)
        self.textMask = nn.Parameter(self.textMask).float().to(DEVICE)
        
        self.audioMask = torch.from_numpy(audioMask) * torch.tensor(2.0)
        self.audioMask = nn.Parameter(self.audioMask).float().to(DEVICE)
        
        self.videoMask = torch.from_numpy(videoMask) * torch.tensor(1.0)
        self.videoMask = nn.Parameter(self.videoMask).float().to(DEVICE)


    def forward(self, features):
        return features * self.textMask + features * self.audioMask + features * self.videoMask

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.textMask.item()} + {self.audioMask.item()} + {self.videoMask.item()}'

class GAT_FP(nn.Module):
    def __init__(self, out_size, wFP, probality = False):
        super().__init__()
        self.audioEncoder = nn.Linear(512, 64).to(torch.float64)
        self.dropAudio = nn.Dropout(0.5)
        self.visionEncoder = nn.Linear(1024, 64).to(torch.float64)
        self.dropVision = nn.Dropout(0.5)
        self.textEncoder = nn.Linear(1024, 64).to(torch.float64)
        self.in_size = 192
        self.outMMEncoder = 4
        self.MMEncoder = nn.LSTM(self.in_size, self.outMMEncoder, bidirectional = True).to(torch.float64)
        gcv = [self.in_size, 32, 4]
        self.maskFilter = maskFilter(self.in_size)
        self.num_heads = 16
        self.imputationModule = dglnn.GraphConv(self.in_size,  self.in_size, norm = 'both', weight=True)
        self.gat1 = nn.ModuleList()
        # two-layer GCN
        for ii in range(len(gcv)-1):
            self.gat1.append(
                dglnn.GATv2Conv(np.power(self.num_heads, ii) * gcv[ii],  gcv[ii+1], activation=F.relu,  residual=True, num_heads = self.num_heads)
            )
        coef = 1
        self.reconstruct = dglnn.GATv2Conv(np.power(self.num_heads, 1) * gcv[1],  self.in_size, activation=F.relu,  residual=True, num_heads = 1)
        self.gat2 = MultiHeadGATCrossModal(self.in_size,  gcv[-1], num_heads = self.num_heads)
        self.linear = nn.Linear(136, out_size).to(torch.float64)
        # self.linear = nn.Linear(gcv[-1] * self.num_heads * 7, out_size)
        self.dropout = nn.Dropout(0.5)
        self.probality = probality

    def forward(self, g):
        text = g.ndata["text"].to(torch.float64)
        audio = g.ndata["audio"]
        audio = audio.to(torch.float64)
        video = g.ndata["vision"]
        video = video.to(torch.float64)
        # text =norm(text)
        # audio =norm(audio)
        # video =norm(video)
        audioOuput = self.audioEncoder(audio)
        audioOuput = self.dropAudio(audioOuput)
        
        visionOutput = self.visionEncoder(video)
        
        visionOutput = self.dropVision(visionOutput)
        textOutput = self.textEncoder(text)
        stackFT = torch.hstack([textOutput, audioOuput, visionOutput]).to(torch.float64)
        newFeature = stackFT.view(-1, 120, self.in_size).to(torch.float64)
        newFeature = newFeature.permute(1, 0, 2)
        newFeature, _ = self.MMEncoder(newFeature)
        newFeature = newFeature.permute(1, 0, 2)
        newFeature = newFeature.reshape(-1, self.outMMEncoder*2)  

        # stackFT = torch.hstack([text, audio, video]).float()  
        # stackFT = stackFT.view(-1, 100, self.in_size).to(torch.float64)
        h = stackFT.float()
        h1 = self.imputationModule(g, h)
        h = 0.5 * (h + h1)
        # h = h + h1
        h = F.normalize(h, p=1)
        # h = self.maskFilter(h)
        h3 = self.gat2(g, h)
        for i, layer in enumerate(self.gat1):
            if i != 0:
                h = self.dropout(h)
            h = h.float()
            h = torch.reshape(h, (len(h), -1))
            h = layer(g, h)
            if i == 0 and self.probality:
                self.firstGCN = torch.sigmoid(h)
                self.data_rho = torch.mean(self.firstGCN.reshape(-1, 16*32), 0)
        
        h = torch.reshape(h, (len(h), -1))
        h = torch.cat((h,newFeature,h3), 1)
        h = self.linear(h)
        return h

    def rho_loss(self, rho, size_average=True):        
        dkl = - rho * torch.log(self.data_rho) - (1-rho)*torch.log(1-self.data_rho) # calculates KL divergence
        if size_average:
            self._rho_loss = dkl.mean()
        else:
            self._rho_loss = dkl.sum()
        return self._rho_loss


def train(trainLoader, testLoader, model, info, numLB):
    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=info['lr'], weight_decay=info['weight_decay'])
    highestAcc = 0
    # training loop
    for epoch in range(info['numEpoch']):
        model.train()
        totalLoss = 0
        for batch in tqdm(trainLoader):
            g, labels = batch
            g = g.to(DEVICE)
            labels = g.ndata["label"]
            labels = labels.type(torch.LongTensor)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(g)
            pos = torch.where(labels != numLB)
            labels = labels[pos]
            logits = logits[pos]
            loss = np.power((100 - info['missing']) * 0.01,2) * loss_fcn(logits, labels)
            totalLoss += loss.item()
            loss.backward()
            optimizer.step()
        acc = evaluate(trainLoader, model, numLB)
        acctest = evaluate(testLoader, model, numLB)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy_train {:.4f} | Accuracy_test {:.4f} ".format(
                epoch, totalLoss, acc, acctest
            )
        )
        highestAcc = max(highestAcc, acctest)

    return highestAcc
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--E', help='number of epochs', default=50, type=int)
    parser.add_argument('--seed', help='type of seed: random vs fix', default='random')
    parser.add_argument('--lr', help='learning rate', default=0.003, type=float)
    parser.add_argument('--rho', help='probality default', default=-1.0, type=float)
    parser.add_argument('--weight_decay', help='weight decay', default=0.00001, type=float)
    parser.add_argument('--edgeType', help='type of edge:0 for similarity and 1 for other', default=0, type=int)
    parser.add_argument('--missing', help='percentage of missing utterance in MM data', default=0, type=int)
    parser.add_argument('--wFP', action='store_true', default=False, help='edge direction type')
    parser.add_argument('--numTest', help='number of test', default=10, type=int)
    parser.add_argument('--batchSize', help='size of batch', default=16, type=int)
    parser.add_argument('--log', action='store_true', default=True, help='save experiment info in output')
    parser.add_argument('--output', help='savedFile', default='./log.txt')
    parser.add_argument('--prePath', help='prepath to directory contain DGL files', default='.')
    parser.add_argument('--numLabel', help='4label vs 6label', default='6')
    parser.add_argument( "--dataset",
        type=str,
        default="IEMOCAP",
        help="Dataset name ('IEMOCAP', 'MELD').",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")
    torch.cuda.empty_cache()
    info = {
            'numEpoch': args.E,
            'lr': args.lr, 
            'weight_decay': args.weight_decay,
            'missing': args.missing,
            'seed': args.seed,
            'numTest': args.numTest,
            'wFP': args.wFP,
            'numLabel': args.numLabel
        }
    for test in range(args.numTest):
        if args.seed == 'random':
            setSeed = seedList[test]
            info['seed'] = setSeed
        else:
            setSeed = int(args.seed)
        seed_everything(seed=setSeed)
        info['seed'] = setSeed
        if args.log:
            sourceFile = open(args.output, 'a')
            print('*'*10, 'INFO' ,'*'*10, file = sourceFile)
            print(info, file = sourceFile)
            sourceFile.close()
                 
        # dataPath  = './IEMOCAP_features/IEMOCAP_features.pkl'
        # data = emotionDataset(missing = args.missing, path = dataPath)
        # trainSet, testSet = data.trainSet, data.testSet
        # trainLoader = GraphDataLoader( dataset=trainSet, batch_size=args.batchSize, shuffle=True)
        # testLoader = GraphDataLoader( dataset=testSet, batch_size=args.batchSize)
        numLB = 6
        if args.numLabel =='4':
            numLB = 4
        dataPath  = f'./IEMOCAP/IEMOCAP_features_raw_{numLB}way.pkl'
        data = Iemocap6_Gcnet_Dataset(missing = args.missing, path = dataPath, info = info)
        trainSet, testSet = data.trainSet, data.testSet
        g = torch.Generator()
        g.manual_seed(setSeed)

        trainLoader = GraphDataLoader(  dataset=trainSet, 
                                        batch_size=args.batchSize, 
                                        shuffle=True, 
                                        generator=g)
        testLoader = GraphDataLoader(   dataset=testSet, 
                                        batch_size=args.batchSize,
                                        generator=g)

        # create GCN model
        out_size = data.out_size 
        model = GAT_FP(out_size, args.wFP, probality = True).to(DEVICE)    
        print(model)
        # model training
        print("Training...")
        highestAcc = train(trainLoader, testLoader, model, info, numLB)
        # test the model
        print("Testing...")
        acc = evaluate(testLoader, model, numLB)
        print("Final Test accuracy {:.4f}".format(acc))
        if args.log:
            sourceFile = open(args.output, 'a')
            print(f'Highest Acc: {highestAcc}, final Acc {acc}', file = sourceFile)
            print('*'*10, 'End' ,'*'*10, file = sourceFile)
            sourceFile.close()