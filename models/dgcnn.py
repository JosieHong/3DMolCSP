'''
Date: 2021-11-30 13:55:12
LastEditors: yuhhong
LastEditTime: 2022-08-06 12:58:02
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import FCResDecoder, TRNet



class EdgeConv(nn.Module):
    def __init__(self, in_dim, out_dim, k, device):
        super(EdgeConv, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.conv = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_dim), 
                                   nn.LeakyReLU(negative_slope=0.2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x): 
        # generate edge features
        feature = self._get_graph_feature(x, device=self.device, k=self.k) # get k-nearest neighbors
        feature = self.conv(feature)
        return feature.max(dim=-1, keepdim=False)[0]

    def _knn(self, x, k):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
        return idx

    def _get_graph_feature(self, x, device, k=20, idx=None): 
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None: 
            idx = self._knn(x, k=k)   # (batch_size, num_points, k)
        # device = torch.device('cuda')
        device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims) 
        # batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous() # torch.Size([32, 42, 300, 5])
        return feature
    
    def __repr__(self):
        return self.__class__.__name__ + '(k=' + str(self.k) + ', dims=('+str(self.in_dim) + '->' + str(self.out_dim) + '))'



class Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim, k, device):
        super(Encoder, self).__init__()
        
        self.conv1 = EdgeConv(in_dim=in_dim, out_dim=64, k=k, device=device)
        self.conv2 = EdgeConv(in_dim=64, out_dim=64, k=k, device=device)
        self.conv3 = EdgeConv(in_dim=64, out_dim=128, k=k, device=device)
        self.conv4 = EdgeConv(in_dim=128, out_dim=256, k=k, device=device)
        self.conv5 = EdgeConv(in_dim=256, out_dim=512, k=k, device=device)
        self.conv6 = EdgeConv(in_dim=512, out_dim=1024, k=k, device=device)

        self.merge = nn.Sequential(nn.Linear(4096, emb_dim), 
                                   nn.BatchNorm1d(emb_dim), 
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x): 
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1) 

        p1 = F.adaptive_max_pool1d(x, 1).squeeze()
        p2 = F.adaptive_avg_pool1d(x, 1).squeeze()
        x = torch.cat((p1, p2), 1) # torch.Size([64, 2048]) + torch.Size([64, 2048]) -> torch.Size([64, 4096])
        x = self.merge(x) # torch.Size([64, 4096]) -> torch.Size([64, 2048])
        return x



class DGCNN(nn.Module): 
    def __init__(self, args, device): 
        super(DGCNN, self).__init__()
        self.num_add = args['num_add']
        self.num_atoms = args['num_atoms']
        self.device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")

        self.tr_net = TRNet(device)
        self.encoder = Encoder(in_dim=args['in_channels'], 
                                    emb_dim=args['emb_dim'], 
                                    k=args['k'], 
                                    device=device)
        
        self.decoder = FCResDecoder(in_dim=args['emb_dim']+args['num_add'], 
                                    layers=args['decoder_layers'], 
                                    out_dim=args['out_channels'], 
                                    dropout=args['dropout'])

        for m in self.modules(): 
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x, mask, env): 
        batch_size = x.size(0)

        # init xyzw by xyz
        w = torch.ones(batch_size, 1, self.num_atoms).to(self.device) 
        xyz = x[:, :3, :]
        xyzw = torch.cat((xyz, w), dim=1)

        # predict transformation matrix by TRNet
        tr_matrix = self.tr_net(xyz)
        
        # translation and rotation transform
        xyzw = torch.bmm(tr_matrix, xyzw)
        
        # convert xyzw to xyz
        w = xyzw[:, 3, :]
        w = torch.stack([w, w, w], dim=1)
        xyz = xyzw[:, :3, :]
        xyz = torch.div(xyz, w)
        
        # concat transformed xyz to input
        x = torch.cat((xyz, x[:, 3:, :]), dim=1)

        # encoder
        x = self.encoder(x) # torch.Size([batch_size, emb_dim])

        if self.num_add == 1:
            x = torch.cat((x, torch.unsqueeze(env, 1)), 1)
        elif self.num_add > 1:
            x = torch.cat((x, env), 1)

        # decoder
        x = self.decoder(x)
        return x
