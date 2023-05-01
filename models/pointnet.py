'''
Date: 2021-11-30 13:55:12
LastEditors: yuhhong
LastEditTime: 2022-08-06 12:58:06
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import FCResDecoder, TRNet



class Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.conv5 = nn.Conv1d(256, 512, 1)
        self.conv6 = nn.Conv1d(512, 1024, 1)

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



class PointNet(nn.Module): 
    def __init__(self, args, device): 
        super(PointNet, self).__init__()
        self.num_add = args['num_add']
        self.num_atoms = args['num_atoms']
        self.device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")

        self.tr_net = TRNet(device)
        self.encoder = Encoder(in_dim=args['in_channels'], 
                                    emb_dim=args['emb_dim'])
        
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
