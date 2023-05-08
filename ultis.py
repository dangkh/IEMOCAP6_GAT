import random 
import torch 
import numpy as np 
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
seed = 1001
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seedList = [1001, 9138, 86503, 37949, 22627, 75258, 94877, 9829, 47702, 15908]

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

cos = nn.CosineSimilarity(dim=0, eps=1e-6)


def featureSimilarity(v1, v2):
    similar = 1.0 - torch.acos(cos(v1, v2))/ np.pi
    return similar

def convertNP2Tensor(listV):
    listR = []
    for xx in listV:
        listR.append(torch.from_numpy(xx))
    return listR

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

def norm(features):
    meanMat = np.mean(features, axis=0, keepdims=True)
    stdMat = np.std(features, axis=0, keepdims=True)
    stdMat[np.where(stdMat == 0)] = 1
    newFeatures = (features - meanMat) / stdMat
    # minMat = np.min(newFeatures, axis = 0, keepdims=True)
    # newFeatures = newFeatures- minMat
    return newFeatures

def vis(info):
    print('Visualize')
    X0, y0 = info
    visData = [[X0, y0]]
    embeddings = visData[0][0]
    tsne = TSNE(n_components=2)
    transformed = tsne.fit_transform(embeddings)

    palette = sns.color_palette("bright", len(np.unique(y0)))
    g = sns.scatterplot(
        x=transformed[:,0],
        y=transformed[:,1],
        hue=visData[0][1],
        legend='full',
        palette=palette
    )
    # _lg = g.get_legend()
    # _lg.remove()
    plt.show()

def evaluate(dataloader, model, numLB):
    counter = 0
    total = 0
    preds = []
    trueLabel = []
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        g, labels = batch
        labels = g.ndata["label"]
        labels = labels.type(torch.LongTensor)   
        trueLabel.extend(labels.cpu().numpy())
        g = g.to(DEVICE)
        with torch.no_grad():
            model.eval()
            pred = model(g)
            res = torch.argmax(pred, 1)
            res = res.to(DEVICE)
            preds.extend(res.cpu().numpy())
    trueLabel = np.asarray(trueLabel)
    preds = np.asarray(preds)
    pos = np.where(trueLabel != numLB)
    preds = preds[pos]
    trueLabel = trueLabel[pos]
    return f1_score(trueLabel, preds, average='weighted')


def normMat(X_train, refer, ax = 1):
    mean = np.mean(refer, axis=ax, keepdims=True)
    std = np.std(refer, axis=ax, keepdims=True)
    std[np.where(std == 0)] = 1
    X_train = (X_train - mean) / std
    return X_train

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P
 
