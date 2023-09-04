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
from torch.nn import init


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
    def __init__(self, out_size, wFP, probality = False, modality = "avl", lambd = 0.1):
        super().__init__()
        self.modality = modality
        if 'a' in self.modality:
            self.audioEncoder = nn.Linear(512, 64).to(torch.float64)
            self.dropAudio = nn.Dropout(0.5)
        if 'v' in self.modality:
            self.visionEncoder = nn.Linear(1024, 64).to(torch.float64)
            self.dropVision = nn.Dropout(0.5)
        if 'l' in self.modality:
            self.textEncoder = nn.Linear(1024, 64).to(torch.float64)
        self.in_size = len(self.modality)*64

        self.outMMEncoder = 8
        # <40 self.outMMencoder = 4
        self.MMEncoder = nn.LSTM(self.in_size, self.outMMEncoder, bidirectional = True).to(torch.float64)
        gcv = [self.in_size, 32, 4]
        if len(self.modality) == 3:
            self.maskFilter = maskFilter(self.in_size)
        self.num_heads = 16
        self.imputationModule = dglnn.GraphConv(self.in_size,  self.in_size, norm = 'both')
        self.decodeModule = nn.Linear(self.in_size, self.in_size)
        self.gat1 = nn.ModuleList()
        # two-layer GCN
        for ii in range(len(gcv)-1):
            self.gat1.append(
                dglnn.GATv2Conv(np.power(self.num_heads, ii) * gcv[ii],  gcv[ii+1], activation=F.relu,  residual=True, num_heads = self.num_heads)
            )
        if len(self.modality) > 1:
            self.gat2 = MultiHeadGATCrossModal(self.in_size,  gcv[-1], self.num_heads, merge = 'cat', modality = self.modality)
        if len(self.modality) != 1:
            self.linear = nn.Linear(144, out_size).to(torch.float64)
        else:
            self.linear = nn.Linear(80, out_size).to(torch.float64)
        # <40 self.linear = 136
        # self.linear = nn.Linear(gcv[-1] * self.num_heads * 7, out_size)
        self.dropout = nn.Dropout(0.75)
        self.probality = probality
        self.lambd = lambd
        # self.reset_parameters()

    def featureFusion(self, allF):
        if len(self.modality) == 3:
            tf, af, vf = allF
        elif self.modality == "av":
            af, vf = allF
        elif self.modality == "al":
            tf, af = allF
        elif self.modality == "vl":
            tf,  vf = allF
        elif self.modality == "a":
            af = allF[0]
        elif self.modality == "v":
            vf = allF[0]
        elif self.modality == "l":
            tf = allF[0]

        if 'a' in self.modality:
            audioOuput = self.audioEncoder(af)
            audioOuput = self.dropAudio(audioOuput)
        if 'v' in self.modality:
            visionOutput = self.visionEncoder(vf)
            visionOutput = self.dropVision(visionOutput)
        if 'l' in self.modality:
            textOutput = self.textEncoder(tf)
        listFt = []
        if 'l' in self.modality:
            listFt.append(textOutput)
        if 'a' in self.modality:
            listFt.append(audioOuput)
        if 'v' in self.modality:
            listFt.append(visionOutput)
        stackFT = torch.hstack(listFt).to(torch.float64)

        newFeature = stackFT.view(-1, 120, self.in_size).to(torch.float64)
        newFeature = newFeature.permute(1, 0, 2)
        newFeature, _ = self.MMEncoder(newFeature)
        newFeature = newFeature.permute(1, 0, 2)
        newFeature = newFeature.reshape(-1, self.outMMEncoder*2)  
        return newFeature, stackFT


    def forward(self, g):
        if 'l' in self.modality:
            text = g.ndata["text"].to(torch.float64)
            oText = g.ndata["oText"].to(torch.float64)
        if 'a' in self.modality:
            audio = g.ndata["audio"]
            audio = audio.to(torch.float64)
            oAudio = g.ndata["oAudio"]
            oAudio = oAudio.to(torch.float64)
        if 'v' in self.modality:
            video = g.ndata["vision"]
            video = video.to(torch.float64)
            oVideo = g.ndata["oVision"]
            oVideo = oVideo.to(torch.float64)

        listFt = []
        listFOt = []
        if 'l' in self.modality:
            listFt.append(text)
            listFOt.append(oText)
        if 'a' in self.modality:
            listFt.append(audio)
            listFOt.append(oAudio)
        if 'v' in self.modality:
            listFt.append(video)
            listFOt.append(oVideo)
        newFeature, stackFT = self.featureFusion(listFt)
        oFeature, oStackFT = self.featureFusion(listFOt)
        h = stackFT.float()
        h1 = self.imputationModule(g, h)
        h1 = self.decodeModule(h1)
        h = self.lambd * h + (1-self.lambd) * h1
        self.data_mse = h
        self.odata = oStackFT.float()
        # h = h + h1
        h = F.normalize(h, p=1)
        if len(self.modality) == 3:
            h = self.maskFilter(h)
        if len(self.modality) > 1:
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
        if len(self.modality) > 1:
            h = torch.cat((h,newFeature,h3), 1)
        else:
            h = torch.cat((h,newFeature), 1)
        h = self.linear(h)
        return h

    def reset_parameters(self):
        self.imputationModule.reset_parameters()
        for i, layer in enumerate(self.gat1):
            layer.reset_parameters()
        init.xavier_uniform_(self.linear.weight, gain=1)
        nn.init.constant_(self.linear.bias, 0)
        init.xavier_uniform_(self.audioEncoder.weight, gain=1)
        nn.init.constant_(self.audioEncoder.bias, 0)
        init.xavier_uniform_(self.visionEncoder.weight, gain=1)
        nn.init.constant_(self.visionEncoder.bias, 0)
        self.textEncoder.reset_parameters()
        self.MMEncoder.reset_parameters()

    def mseLoss(self):
        return self.data_mse, self.odata

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
    loss_imput = nn.MSELoss()
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
            # loss = loss_fcn(logits, labels)
            if info['reconstructionLoss'] == 'mse':
                data_mse, odata = model.mseLoss()
                loss = loss_fcn(logits, labels) + (info['missing']) * 0.01 * loss_imput(data_mse, odata)
            elif (info['reconstructionLoss'] == 'kl') and (int(info['rho']) != -1):
                loss = loss_fcn(logits, labels) + (info['missing']) * 0.01 * model.rho_loss(float(info['rho']))
                # loss = (100 - info['missing']) * 0.01 * loss_fcn(logits, labels) + (info['missing']) * 0.01 * model.rho_loss(float(info['rho']))
            else:
                loss = loss_fcn(logits, labels)

            totalLoss += loss.item()
            loss.backward()
            optimizer.step()
        # acc = evaluate(trainLoader, model, numLB)
        acc  = -1
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
    parser.add_argument('--batchSize', help='size of batch', default=64, type=int)
    parser.add_argument('--log', action='store_true', default=True, help='save experiment info in output')
    parser.add_argument('--output', help='savedFile', default='./log_v2.txt')
    parser.add_argument('--prePath', help='prepath to directory contain DGL files', default='.')
    parser.add_argument('--numLabel', help='4label vs 6label', default='6')
    parser.add_argument('--reconstructionLoss', 
        help='mse, kl, none. unless set rho number for kl loss, using none loss instead',
        default='none')
    parser.add_argument('--modality', 
        help='avl, av, al, vl, a, v, l',
        default='avl')
    parser.add_argument('--lambd', help='imputation contribution', default=0.5, type=float)
    parser.add_argument( "--dataset",
        type=str,
        default="IEMOCAP",
        help="Dataset name ('IEMOCAP', 'MELD').",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")
    torch.cuda.empty_cache()
    typeModal = ['avl', 'av', 'al', 'vl', 'a', 'v', 'l']
    if args.modality not in typeModal:
        raise "chosen modality not in available list"
    info = {
            'numEpoch': args.E,
            'lr': args.lr, 
            'weight_decay': args.weight_decay,
            'missing': args.missing,
            'seed': args.seed,
            'numTest': args.numTest,
            'wFP': args.wFP,
            'numLabel': args.numLabel,
            'reconstructionLoss': args.reconstructionLoss,
            'modality': args.modality,
            'lambd': args.lambd,
            'rho': args.rho
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
        model = GAT_FP(out_size, args.wFP, probality = True, modality = args.modality, lambd = args.lambd)
        for layer in model.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        model = model.to(DEVICE)
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