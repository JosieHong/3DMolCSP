'''
Date: 2022-12-13 15:47:53
LastEditors: yuhhong
LastEditTime: 2022-12-13 15:47:53
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .utils import FCResDecoder



class MolConv(nn.Module):
    def __init__(self, in_dim, out_dim, k, remove_xyz=False):
        super(MolConv, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.remove_xyz = remove_xyz

        self.dist_ff = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1, bias=False),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid())
        self.gm2m_ff = nn.Sequential(nn.Conv2d(k, 1, kernel_size=1, bias=False),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid())

        if remove_xyz: 
            self.update_ff = nn.Sequential(nn.Conv2d(in_dim-3, out_dim, kernel_size=1, bias=False),
                                nn.BatchNorm2d(out_dim),
                                nn.LeakyReLU(negative_slope=0.02))
        else:
            self.update_ff = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                                nn.BatchNorm2d(out_dim),
                                nn.LeakyReLU(negative_slope=0.02))

        self._reset_parameters()

    def _reset_parameters(self): 
        for m in self.modules(): 
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, 
                        idx_base: torch.Tensor) -> torch.Tensor: 
        dist, gm2, feat_c, feat_n = self._generate_feat(x, idx_base, k=self.k, remove_xyz=self.remove_xyz) 
        '''Returned features: 
        dist: torch.Size([batch_size, 1, point_num, k])
        gm2: torch.Size([batch_size, k, point_num, k]) 
        feat_c: torch.Size([batch_size, in_dim, point_num, k]) 
        feat_n: torch.Size([batch_size, in_dim, point_num, k])
        '''
        w1 = self.dist_ff(dist)
        w2 = self.gm2m_ff(gm2)
        
        feat = torch.mul(w1, w2) * feat_n + (1 - torch.mul(w1, w2)) * feat_c
        feat = self.update_ff(feat)
        feat = feat.mean(dim=-1, keepdim=False)
        return feat

    def _generate_feat(self, x: torch.Tensor, 
                                idx_base: torch.Tensor, 
                                k: int, 
                                remove_xyz: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        batch_size, num_dims, num_points = x.size()
        
        # local graph (knn)
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        dist, idx = pairwise_distance.topk(k=k, dim=2) # (batch_size, num_points, k)
        dist = - dist

        # device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
        # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1).contiguous() # (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims) 
        # print('_double_gram_matrix (x):', torch.any(torch.isnan(x)))
        graph_feat = x.view(batch_size*num_points, -1)[idx, :]
        # print('_double_gram_matrix (graph_feat):', torch.any(torch.isnan(graph_feat)))
        graph_feat = graph_feat.view(batch_size, num_points, k, num_dims)

        # gram matrix
        gm_matrix = torch.matmul(graph_feat, graph_feat.permute(0, 1, 3, 2))
        # gm_matrix = F.normalize(gm_matrix, dim=1) 
        # print('_double_gram_matrix (gm_matrix):', torch.any(torch.isnan(gm_matrix)))

        # double gram matrix
        sub_feat = gm_matrix[:, :, :, 0].unsqueeze(3)
        sub_gm_matrix = torch.matmul(sub_feat, sub_feat.permute(0, 1, 3, 2))
        sub_gm_matrix = F.normalize(sub_gm_matrix, dim=1) 
        # print('_double_gram_matrix (sub_gm_matrix):', torch.any(torch.isnan(sub_gm_matrix)))
        
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        if remove_xyz:
            return dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(), \
                    sub_gm_matrix.permute(0, 3, 1, 2).contiguous(), \
                    x[:, :, :, 3:].permute(0, 3, 1, 2).contiguous(), \
                    graph_feat[:, :, :, 3:].permute(0, 3, 1, 2).contiguous()
        else:
            return dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(), \
                    sub_gm_matrix.permute(0, 3, 1, 2).contiguous(), \
                    x.permute(0, 3, 1, 2).contiguous(), \
                    graph_feat.permute(0, 3, 1, 2).contiguous()
    
    def __repr__(self):
        return self.__class__.__name__ + ' k = ' + str(self.k) + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'



class Encoder(nn.Module):
    def __init__(self, in_dim, layers, emb_dim, k, device):
        super(Encoder, self).__init__()
        self.hidden_layers = nn.ModuleList([MolConv(in_dim=in_dim, out_dim=layers[0], k=k, remove_xyz=True)])
        for i in range(1, len(layers)): 
            if i == 1:
                self.hidden_layers.append(MolConv(in_dim=layers[i-1], out_dim=layers[i], k=k, remove_xyz=False))
            else:
                self.hidden_layers.append(MolConv(in_dim=layers[i-1], out_dim=layers[i], k=k, remove_xyz=False))
        
        self.merge = nn.Sequential(nn.Linear(emb_dim*2, emb_dim), 
                                   nn.BatchNorm1d(emb_dim), 
                                   nn.LeakyReLU(negative_slope=0.2))
        self._reset_parameters()

    def _reset_parameters(self): 
        for m in self.merge: 
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, 
                        idx_base: torch.Tensor) -> torch.Tensor: 
        '''
        x:      set of points, torch.Size([32, 21, 300]) 
        mask:   mask of real atom numner (without padding), torch.Size([32, 300])
        ''' 
        xs = []
        for i, hidden_layer in enumerate(self.hidden_layers): 
            if i == 0: 
                tmp_x = hidden_layer(x, idx_base)
            else: 
                tmp_x = hidden_layer(xs[-1], idx_base)
            # tmp_x = torch.mul(tmp_x.permute(1, 0, 2), mask).permute(1, 0, 2) # apply the mask
            xs.append(tmp_x)

        x = torch.cat(xs, dim=1)
        p1 = F.adaptive_max_pool1d(x, 1).squeeze()
        p2 = F.adaptive_avg_pool1d(x, 1).squeeze()
        
        x = torch.cat((p1, p2), 1)
        x = self.merge(x)
        return x



class MolNet(nn.Module): 
    def __init__(self, args, device): 
        super(MolNet, self).__init__()
        self.num_add = args['num_add']
        self.num_atoms = args['num_atoms']
        # self.device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")

        self.encoder = Encoder(in_dim=args['in_channels'], 
                                layers=args['encoder_layers'], 
                                    emb_dim=args['emb_dim'], 
                                    k=args['k'], 
                                    device=device)
        self.decoder = FCResDecoder(in_dim=args['emb_dim']+args['num_add'], 
                                    layers=args['decoder_layers'], 
                                    out_dim=args['out_channels'], 
                                    dropout=args['dropout'])
        
    def forward(self, x: torch.Tensor, 
                        env: torch.Tensor, 
                        idx_base: torch.Tensor) -> torch.Tensor: 
        '''
        Input: 
            x:      point set, torch.Size([batch_size, 21, num_atoms])
            env:    experimental condiction
            idx_base:   idx for local knn
        '''
        x = self.encoder(x, idx_base) # torch.Size([batch_size, emb_dim])
        
        if self.num_add == 1:
            x = torch.cat((x, torch.unsqueeze(env, 1)), 1)
        elif self.num_add > 1:
            x = torch.cat((x, env), 1)

        # decoder
        x = self.decoder(x)
        return torch.squeeze(x)
