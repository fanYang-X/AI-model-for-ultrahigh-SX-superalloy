import copy
import numpy as np
import torch
import torch.nn as nn

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, encode_layer, N):
        super(Encoder, self).__init__()
        self.layer_list = clones(encode_layer, N)
        
    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

class SublayeResConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayeResConnection, self).__init__()
        self.norm = nn.LayerNorm(size) 
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer):
        return self.norm(x + self.Dropout(sub_layer(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, multi_attention, feed_forward, dropout, init):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        if init:
            multi_attention.apply(weights_init)
        self.attention = multi_attention
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.sublayer = clones(SublayeResConnection(self.d_model, self.dropout), 2)

    def forward(self, x):
        x = self.sublayer[0](x, self.attention)
        x = self.sublayer[1](x, self.feed_forward)
        return x

def SelfAttention(q, k, v, scale_factor=0.04):
    k_t = torch.transpose(k, dim0=-2, dim1=-1)
    q_k_t = torch.matmul(q, k_t)
    # mask 
    q_k_t = q_k_t.masked_fill(q_k_t == 0, -1e9)
    soft_value = nn.Softmax(dim=-1)(q_k_t/scale_factor)
    z = torch.matmul(soft_value, v)
    return z, soft_value

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_forward, head, dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_forward = d_forward
        self.head = head
        self.dropout = dropout
        self.atten_score = None
        self.d_k = self.d_forward//self.head
        self.LinearList0 = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.head)])
        self.LinearList1 = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.head)])
        self.LinearList2 = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.head)])
        self.Linear = nn.Linear(self.d_forward, self.d_model)

    def forward(self, x):
        n_batch = x.size(0)
        q = torch.zeros((n_batch, self.head, x.size(1), self.d_k))
        k = torch.zeros((n_batch, self.head, x.size(1), self.d_k))
        v = torch.zeros((n_batch, self.head, x.size(1), self.d_k))
        for i in range(self.head):
            q[:, i, :, :] = self.LinearList0[i](x)
            k[:, i, :, :] = self.LinearList1[i](x)
            v[:, i, :, :] = self.LinearList2[i](x)
        z, self.atten_score = SelfAttention(q, k, v, dropout=self.dropout)
        z = z.transpose(1, 2).contiguous() .view(n_batch, -1, self.head*self.d_k)
        return self.Linear(z)

class AtomFeedForward(nn.Module):
    def __init__(self, d_model, d_forward, dropout=0):
        super(AtomFeedForward, self).__init__()
        self.d_model = d_model
        self.Linear1 = nn.Linear(d_model, d_forward)
        self.Linear2 = nn.Linear(d_forward, d_model)
        self.ReLu = nn.LeakyReLU()
        self.Dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        r = self.Linear1(x)
        r = self.ReLu(r)
        r = self.Dropout(r)
        r = self.Linear2(r)
        return r

class LpBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(LpBlock, self).__init__()
        self.Linear = nn.Linear(input_size, output_size)
        self.BatchNormal = nn.BatchNorm1d(output_size)
        self.ReLu = nn.ReLU()

    def forward(self, x): 
        x = self.Linear(x)
        x = self.BatchNormal(x)
        x = self.ReLu(x)
        return x

def weights_init(m, key='xavier_uniform'):
    classname = m.__class__.__name__
    if classname.find('ModuleList') != -1:
        for l in m:
            if key == 'normal':
                nn.init.normal_(l.weight.data, 1, 0.02)
            elif key == 'orthogonal': 
                nn.init.orthogonal_(l.weight.data)
            elif key == 'eye':
                nn.init.eye_(l.weight.data)
            elif key == 'xavier_uniform':
                nn.init.xavier_uniform_(l.weight.data)
            elif key == 'kaiming_uniform':
                nn.init.kaiming_uniform_(l.weight.data)


class SaTNC(nn.Module):
    def __init__(self, d_model=6, d_forward=16, head=4, dropout=0, attention_num=2, d_forward1=64, scale_factor=0.04):
        super(SaTNC, self).__init__()
        self.d_model = d_model
        self.d_forward = d_forward
        self.head = head
        self.dropout = dropout
        self.attention_num = attention_num
        self.d_forward1 = d_forward1
        self.scale_factor = scale_factor
        c = copy.deepcopy
        self.element = torch.LongTensor(np.array(list(range(17))))
        self.Embeddings = nn.Embedding(17, d_model)
        self.encoder_layer = EncoderLayer(self.d_model, c(MultiHeadAttention(self.d_model, self.d_forward, self.head, self.dropout)), 
                                                            c(AtomFeedForward(self.d_model, self.d_forward1)), self.dropout, True)
        self.TransFormerEncoder = Encoder(self.encoder_layer, self.attention_num)
        self.Dropout = nn.Dropout(p=0.1)
        self.lp_layer = nn.ModuleList([LpBlock(108, 256), LpBlock(256, 512)])
        self.Linear = nn.Linear(512,1)

    def forward(self, x):
        atom_embedding = x[0][:, 0, :, :]*(x[0][:, 1, :, :] + self.Embeddings(self.element)*self.scale_factor)
        r = self.TransFormerEncoder(atom_embedding)
        r = r.view(r.size(0), -1)
        r = self.Dropout(r)
        c = x[-1]
        r = torch.cat([r,c], dim=-1)
        for index in range(len(self.lp_layer)):
            r = self.lp_layer[index](r)
        r = self.Linear(r)
        return r