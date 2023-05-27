import argparse
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.nn import LabelPropagation
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dataloader import *
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.double)
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
        tt, aa, vv  = 100, 442, 2024
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
        self.textMask = nn.Parameter(self.textMask).double().to(DEVICE)
        
        self.audioMask = torch.from_numpy(audioMask) * torch.tensor(2.0)
        self.audioMask = nn.Parameter(self.audioMask).double().to(DEVICE)
        
        self.videoMask = torch.from_numpy(videoMask) * torch.tensor(1.0)
        self.videoMask = nn.Parameter(self.videoMask).double().to(DEVICE)


    def forward(self, features):
        return features * self.textMask + features * self.audioMask + features * self.videoMask

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.textMask.item()} + {self.audioMask.item()} + {self.videoMask.item()}'


class maskFilter_2modals(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        if in_size == 447:
            f1 = 100
            f2 = 342           
        elif in_size == 1687:
            f1 = 100
            f2 = 1582
        elif in_size == 342 + 1582 + 5:
            f1 = 342
            f2 = 1582
        tt, aa, vv  = 100, 442, 2024
        # self.testM = nn.Parameter(torch.rand(in_size, in_size))
        currentFeatures = np.asarray([0.0] * in_size)
        f1Mask = np.copy(currentFeatures)
        f1Mask[:f1] = 1.0
        f2Mask = np.copy(currentFeatures)
        f2Mask[f1: f1+f2] = 1.0
        self.f1Mask = torch.from_numpy(f1Mask) * torch.tensor(2.0)
        self.f1Mask = nn.Parameter(self.f1Mask).double().to(DEVICE)
        
        self.f2Mask = torch.from_numpy(f2Mask) * torch.tensor(2.0)
        self.f2Mask = nn.Parameter(self.f2Mask).double().to(DEVICE)
        
    def forward(self, features):
        return features * self.f1Mask + features * self.f2Mask

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.f1Mask.item()} + {self.f2Mask.item()}'

class GAT_FP(nn.Module):
    def __init__(self, in_size, hid_size, out_size, wFP, numFP, modals):
        super().__init__()
        gcv = [in_size, 256, 8]
        self.modals = modals
        if len(self.modals) == 3:
            self.maskFilter = maskFilter(in_size)
        else:
            self.maskFilter = maskFilter_2modals(in_size)
        self.num_heads = 4
        self.GATFP = dglnn.GraphConv(in_size,  in_size, norm = 'both')
        self.gat1 = nn.ModuleList()
        # two-layer GCN
        for ii in range(len(gcv)-1):
            self.gat1.append(
                dglnn.GATv2Conv(np.power(self.num_heads, ii) * gcv[ii],  gcv[ii+1], activation=F.relu,  residual=True, num_heads = self.num_heads)
            )
        coef = 1
        if len(self.modals) == 3:
            self.gat2 = MultiHeadGATCrossModal(in_size,  gcv[-1], num_heads = self.num_heads)
        else:
            self.gat2 = MultiHeadGATCross2Modal(in_size,  gcv[-1], num_heads = self.num_heads)
        # self.layers.append(dglnn.GraphConv(hid_size, 16))
        self.linear = nn.Linear(gcv[-1] * self.num_heads * 2, out_size)
        # self.linear = nn.Linear(gcv[-1] * self.num_heads * 7, out_size)
        self.dropout = nn.Dropout(0.5)
        self.reset_parameters()

    def forward(self, g):
        features = g.ndata["x"]
        h = features.double()
        h1 = self.GATFP(g, h)
        h = 0.5 * (h + h1)
        h = F.normalize(h, p=1)
        h = self.maskFilter(h)
        h3 = self.gat2(g, h)
        for i, layer in enumerate(self.gat1):
            if i != 0:
                h = self.dropout(h)
            h = h.double()
            h = torch.reshape(h, (len(h), -1))
            h = layer(g, h)
        
        h = torch.reshape(h, (len(h), -1))
        h = torch.cat((h3,h), 1)
        h = self.linear(h)
        return h

    def reset_parameters(self):
        self.GATFP.reset_parameters()
        for i, layer in enumerate(self.gat1):
            layer.reset_parameters()
        init.xavier_uniform_(self.linear.weight, gain=1)
        nn.init.constant_(self.linear.bias, 0)

def train(trainLoader, testLoader, model, info):
    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=info['lr'], weight_decay=info['weight_decay'])
    highestAcc = 0
    # training loop
    for epoch in range(info['numEpoch']):
        model.train()
        totalLoss = 0
        for batch in tqdm(trainLoader):
            g, dataset_idx, labels = batch
            g = g.to(DEVICE)
            labels = g.ndata["label"]
            labels = labels.type(torch.LongTensor)
            dataset_idx =  dataset_idx.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(g)
            loss = loss_fcn(logits, labels)
            totalLoss += loss.item()
            loss.backward()
            optimizer.step()
        acc = evaluate(trainLoader, model)
        acctest = evaluate(testLoader, model)
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
    parser.add_argument('--weight_decay', help='weight decay', default=0.00001, type=float)
    parser.add_argument('--edgeType', help='type of edge:0 for similarity and 1 for other', default=0, type=int)
    parser.add_argument('--missing', help='percentage of missing utterance in MM data', default=0, type=int)
    parser.add_argument('--wFP', action='store_true', default=False, help='edge direction type')
    parser.add_argument('--numFP', help='number of FP layer', default=5, type=int)
    parser.add_argument('--numTest', help='number of test', default=10, type=int)
    parser.add_argument('--batchSize', help='size of batch', default=32, type=int)
    parser.add_argument('--log', action='store_true', default=True, help='save experiment info in output')
    parser.add_argument('--output', help='savedFile', default='./log.txt')
    parser.add_argument('--prePath', help='prepath to directory contain DGL files', default='F:/dangkh/work/test/')
    parser.add_argument( "--dataset",
        type=str,
        default="IEMOCAP",
        help="Dataset name ('IEMOCAP', 'MELD').",
    )
    parser.add_argument( "--modals",
        type=str,
        default="avl",
        help="Dataset name ('av', 'vl', 'al').",
    )
    args = parser.parse_args()

    if args.modals not in ['av', 'vl', 'al', 'avl']:

        raise Exception("Modals must be in list")

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
            'numFP': args.numFP
        }
    for test in range(args.numTest):
        if args.seed == 'random':
            setSeed = seedList[test]
            info['seed'] = setSeed
        else:
            setSeed = int(args.seed)
        seed_everything(seed=setSeed)
        if args.log:
            sourceFile = open('db.txt', 'a')
            print('*'*10, 'INFO' ,'*'*10, file = sourceFile)
            print(info, file = sourceFile)
                 
        dataPath  = './IEMOCAP_features/IEMOCAP_features.pkl'
        data = emotionDataset(missing = args.missing, path = dataPath, modals = args.modals)
        trainSet, testSet = data.trainSet, data.testSet
        trainLoader = GraphDataLoader( dataset=trainSet, batch_size=args.batchSize)
        testLoader = GraphDataLoader( dataset=testSet, batch_size=args.batchSize)


        # create GCN model
        in_size = data.in_size
        out_size = data.out_size 
        model = GAT_FP(in_size, 128, out_size, args.wFP, args.numFP, args.modals)
        for layer in model.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        model.to(DEVICE)
        print(model)
        # model training
        print("Training...")
        highestAcc = train(trainLoader, testLoader, model, info)
        # test the model
        print("Testing...")
        acc = evaluate(testLoader, model)
        print("Final Test accuracy {:.4f}".format(acc))
        if args.log:
            print(f'Highest Acc: {highestAcc}, final Acc {acc}', file = sourceFile)
            print('*'*10, 'End' ,'*'*10, file = sourceFile)
            sourceFile.close()