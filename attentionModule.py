import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np 
import dgl.function as fn

class GATInnerLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATInnerLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.qMask = nn.Linear(in_dim, out_dim, bias=False)
        self.kMask = nn.Linear(in_dim, out_dim, bias=False)
        self.vMask = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.qMask.weight, gain=gain)
        nn.init.xavier_normal_(self.kMask.weight, gain=gain)
        nn.init.xavier_normal_(self.vMask.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        qVal = self.qMask(edges.src['h'])
        kVal = self.kMask(edges.src['h'])
        qVal = torch.unsqueeze(qVal, 2)
        kVal = torch.unsqueeze(kVal, 1)
        score = (kVal.T @ qVal.T )  / np.sqrt(self.in_dim)
        score = F.softmax(score, dim=-1)
        origin = torch.unsqueeze(edges.src['h'], 2)
        attVal = torch.bmm(score, origin).sum(dim = 2)
        # print(attVal.shape)
        return {'a': attVal}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'a': edges.data['a']}

    def reduce_func(self, nodes):
        # self.counter += 1
        # print(f"debug counter at: {self.counter}")
        # print(nodes.mailbox)
        # datas =nodes.mailbox['a'] 

        # print(datas.shape)
        # print(score.shape)
        # lenFeature = datas.shape[-1]
        # datas = datas.reshape(-1, lenFeature)
        # print(datas.shape)
        out = torch.mean(nodes.mailbox['a'] , dim = 1)
        # print(out.shape)
        return {'h': out}

    def forward(self, g, h):
        qVal = self.qMask(h)
        kVal = self.kMask(h)
        vVal = self.vMask(h)
        qVal = torch.unsqueeze(qVal, 2)
        kVal = torch.unsqueeze(kVal, 1)
        score = (qVal @ kVal)  / np.sqrt(self.out_dim)
        score = F.softmax(score, dim=1)
        origin = torch.unsqueeze(vVal, 2)
        attVal = torch.bmm(score, origin).sum(dim = 2)
        return attVal
        # g.ndata['h'] = h
        # g.apply_edges(self.edge_attention)

        # g.update_all(self.message_func, self.reduce_func)
        # return g.ndata.pop('h')

class GATInnerLayer_v2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATInnerLayer_v2, self).__init__()
        currentFeatures = np.asarray([0.0] * in_dim)
        tt, aa, vv  = 100, 442, 2029
        if in_dim == 1247:
            tt, aa, vv  = 600, 942, 1247
        self.tt, self.aa, self.vv = tt, aa, vv
        textMask = np.copy(currentFeatures)
        textMask[:tt] = 1.0
        audioMask = np.copy(currentFeatures)
        audioMask[tt: vv] = 1.0
        videoMask = np.copy(currentFeatures)
        videoMask[vv:] = 1.0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.qMaskT = nn.Linear(tt, out_dim, bias=False)
        self.qMaskA = nn.Linear(aa-tt, out_dim, bias=False)
        self.qMaskV = nn.Linear(vv-aa, out_dim, bias=False)
        self.kMaskT = nn.Linear(tt, out_dim, bias=False)
        self.kMaskA = nn.Linear(aa-tt, out_dim, bias=False)
        self.kMaskV = nn.Linear(vv-aa, out_dim, bias=False)
        self.vMaskT = nn.Linear(tt, out_dim, bias=False)
        self.vMaskA = nn.Linear(aa-tt, out_dim, bias=False)
        self.vMaskV = nn.Linear(vv-aa, out_dim, bias=False)
        self.ln = nn.Linear(out_dim * 3, out_dim, bias = True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.qMaskT.weight, gain=gain)
        nn.init.xavier_normal_(self.kMaskT.weight, gain=gain)
        nn.init.xavier_normal_(self.vMaskT.weight, gain=gain)
        nn.init.xavier_normal_(self.qMaskA.weight, gain=gain)
        nn.init.xavier_normal_(self.kMaskA.weight, gain=gain)
        nn.init.xavier_normal_(self.vMaskA.weight, gain=gain)
        nn.init.xavier_normal_(self.qMaskV.weight, gain=gain)
        nn.init.xavier_normal_(self.kMaskV.weight, gain=gain)
        nn.init.xavier_normal_(self.vMaskV.weight, gain=gain)

    def unitAtt(self, q, k, v, feature):
        qVal = q(feature)
        kVal = k(feature)
        vVal = v(feature)
        qVal = torch.unsqueeze(qVal, 2)
        kVal = torch.unsqueeze(kVal, 1)
        score = (qVal @ kVal)  / np.sqrt(self.out_dim)
        score = F.softmax(score, dim=1)
        origin = torch.unsqueeze(vVal, 2)
        attVal = torch.bmm(score, origin).sum(dim = 2)
        return attVal

    def forward(self, g, h):
        attT = self.unitAtt(self.qMaskT, self.kMaskT, self.vMaskT, h[:,:self.tt])
        attA = self.unitAtt(self.qMaskA, self.kMaskA, self.vMaskA, h[:,self.tt:self.aa])
        attV = self.unitAtt(self.qMaskV, self.kMaskV, self.vMaskV, h[:,self.aa:])
        att = torch.cat((attT, attA, attV), dim = 1)
        # att = torch.mean(torch.stack([attT,attA,attV], 1), 1)
        att = self.ln(att)
        return att
       


class MultiHeadGATInnerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATInnerLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATInnerLayer_v2(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class crossModal(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(crossModal, self).__init__()
        currentFeatures = np.asarray([0.0] * in_dim)
        tt, aa, vv  = 100, 442, 2029
        if in_dim == 1247:
            tt, aa, vv  = 600, 942, 1247
        self.tt, self.aa, self.vv = tt, aa, vv
        textMask = np.copy(currentFeatures)
        textMask[:tt] = 1.0
        audioMask = np.copy(currentFeatures)
        audioMask[tt: vv] = 1.0
        videoMask = np.copy(currentFeatures)
        videoMask[vv:] = 1.0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.qMaskT = nn.Linear(tt, out_dim, bias=False)
        self.qMaskA = nn.Linear(aa-tt, out_dim, bias=False)
        self.qMaskV = nn.Linear(vv-aa, out_dim, bias=False)
        self.kMaskT = nn.Linear(tt, out_dim, bias=False)
        self.kMaskA = nn.Linear(aa-tt, out_dim, bias=False)
        self.kMaskV = nn.Linear(vv-aa, out_dim, bias=False)
        self.vMaskT = nn.Linear(tt, out_dim, bias=False)
        self.vMaskA = nn.Linear(aa-tt, out_dim, bias=False)
        self.vMaskV = nn.Linear(vv-aa, out_dim, bias=False)
        self.ln = nn.Linear(out_dim * 3, out_dim, bias = True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.qMaskT.weight, gain=gain)
        nn.init.xavier_normal_(self.kMaskT.weight, gain=gain)
        nn.init.xavier_normal_(self.vMaskT.weight, gain=gain)
        nn.init.xavier_normal_(self.qMaskA.weight, gain=gain)
        nn.init.xavier_normal_(self.kMaskA.weight, gain=gain)
        nn.init.xavier_normal_(self.vMaskA.weight, gain=gain)
        nn.init.xavier_normal_(self.qMaskV.weight, gain=gain)
        nn.init.xavier_normal_(self.kMaskV.weight, gain=gain)
        nn.init.xavier_normal_(self.vMaskV.weight, gain=gain)

    def unitAtt(self, q, k, v, feature, referFt):
        qVal = q(feature)
        kVal = k(referFt)
        vVal = v(referFt)
        qVal = torch.unsqueeze(qVal, 2)
        kVal = torch.unsqueeze(kVal, 1)
        score = (qVal @ kVal)  / np.sqrt(self.out_dim)
        score = F.softmax(score, dim=1)
        origin = torch.unsqueeze(vVal, 2)
        attVal = torch.bmm(score, origin).sum(dim = 2)
        return attVal

    def forward(self, g, h):
        attT = self.unitAtt(self.qMaskT, self.kMaskA, self.vMaskA, h[:,:self.tt], h[:,self.tt:self.aa])
        attA = self.unitAtt(self.qMaskA, self.kMaskT, self.vMaskT, h[:,self.tt:self.aa], h[:,:self.tt])
        attV = self.unitAtt(self.qMaskV, self.kMaskT, self.vMaskT, h[:,self.aa:], h[:,:self.tt])
        att = torch.cat((attT, attA, attV), dim = 1)
        # att = torch.mean(torch.stack([attT,attA,attV], 1), 1)
        att = self.ln(att)
        return att
       
class MultiHeadGATCrossModal(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATCrossModal, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(crossModal(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

# g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
# g = dgl.add_self_loop(g)
# feat = torch.rand(6, 10)
# gatconv = MultiHeadGATInnerLayer(10, 2, num_heads=3)
# res = gatconv(g, feat)
# print(res.shape)
# print(res)