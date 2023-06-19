'''
Date: 2021-11-30 13:55:12
LastEditors: yuhhong
LastEditTime: 2022-08-01 16:39:53
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import math



class  FCResBlock(nn.Module): 
    def __init__(self, in_dim, out_dim, dropout=None): 
        super(FCResBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # hid_dim = int(in_dim / 4)

        self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
        # self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn1 = nn.LayerNorm(out_dim)

        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.LayerNorm(out_dim)

        self.linear3 = nn.Linear(out_dim, out_dim)
        self.bn3 = nn.LayerNorm(out_dim)

        self.dp = nn.Dropout(dropout) if dropout != None else None 

        self._reset_parameters()

    def _reset_parameters(self): 
        for m in self.modules(): 
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        
        x = self.bn1(self.linear1(x))
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn2(self.linear2(x))
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn3(self.linear3(x))
        
        x = x + F.interpolate(identity.unsqueeze(1), size=x.size()[1]).squeeze()
        
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dp(x) if self.dp != None else x
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'



class FCResDecoder(nn.Module): 
    def __init__(self, in_dim, layers, out_dim, dropout): 
        super(FCResDecoder, self).__init__()
        self.blocks = nn.ModuleList([FCResBlock(in_dim=in_dim, out_dim=layers[0])])
        for i in range(len(layers)-1): 
            if len(layers) - i > 3:
                self.blocks.append(FCResBlock(in_dim=layers[i], out_dim=layers[i+1]))
            else:
                self.blocks.append(FCResBlock(in_dim=layers[i], out_dim=layers[i+1], dropout=dropout))

        self.fc = nn.Linear(layers[-1], out_dim)
        
        self._reset_parameters()

    def _reset_parameters(self): 
        nn.init.kaiming_normal_(self.fc.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        # x = self.fc(x)
        # return F.glu(torch.cat((x, x), dim=1))
        return self.fc(x)



class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, set_size):
        super(MLPDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim * set_size),
        )

    def forward(self, x): 
        x = x.view(x.size(0), -1).float()
        return self.decoder(x)
    


class TRNet(nn.Module):
    def __init__(self, emb_dim): 
        super(TRNet, self).__init__()
        self.emb_dim = emb_dim
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, emb_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(emb_dim)

        self.fc1 = nn.Linear(emb_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 6)
        self.sigmoid = nn.Sigmoid()
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self._reset_parameters()

    def _reset_parameters(self): 
        for m in self.modules(): 
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

            elif isinstance(m, nn.Linear): 
                sigmoid_gain = nn.init.calculate_gain('sigmoid')
                nn.init.xavier_uniform_(m.weight.data, gain=sigmoid_gain)

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.emb_dim)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.sigmoid(self.fc3(x)) # in range [0, 1]

        # create the rotation and translation transformation matrix
        return self._assemble_transformation(x)

    def _assemble_transformation(self, x):
        # batch_size = x.size(0)
        # tensor0 = torch.zeros(batch_size).to(self.device) 
        # tensor1 = torch.ones(batch_size).to(self.device) 
        tensor0 = torch.zeros_like(x[:, 0]) 
        tensor1 = torch.ones_like(x[:, 0]) 

        alpha = 2 * math.pi * x[:, 0] # in range [0, 2pi]
        pheta = 2 * math.pi * x[:, 1] # in range [0, 2pi]
        gamma = 2 * math.pi * x[:, 2] # in range [0, 2pi]
        rx = torch.permute(torch.stack([
                torch.stack([tensor1, tensor0, tensor0, tensor0]),
                torch.stack([tensor0, torch.cos(alpha), -torch.sin(alpha), tensor0]),
                torch.stack([tensor0, torch.sin(alpha), torch.cos(alpha), tensor0]),
                torch.stack([tensor0, tensor0, tensor0, tensor1])]), (2, 0, 1))
        ry = torch.permute(torch.stack([
                torch.stack([torch.cos(pheta), tensor0, torch.sin(pheta), tensor0]),
                torch.stack([tensor0, tensor1, tensor0, tensor0]),
                torch.stack([-torch.sin(pheta), tensor0, torch.cos(pheta), tensor0]),
                torch.stack([tensor0, tensor0, tensor0, tensor1])]), (2, 0, 1))
        rz = torch.permute(torch.stack([
                torch.stack([torch.cos(gamma), -torch.sin(gamma), tensor0, tensor0]),
                torch.stack([torch.sin(gamma), torch.cos(gamma), tensor0, tensor0]),
                torch.stack([tensor0, tensor0, tensor1, tensor0]),
                torch.stack([tensor0, tensor0, tensor0, tensor1])]), (2, 0, 1))
        t = torch.permute(torch.stack([
                torch.stack([tensor1, tensor0, tensor0, x[:, 3]]),
                torch.stack([tensor0, tensor1, tensor0, x[:, 4]]),
                torch.stack([tensor0, tensor0, tensor1, x[:, 5]]),
                torch.stack([tensor0, tensor0, tensor0, tensor1])]), (2, 0, 1))
        return torch.bmm(torch.bmm(torch.bmm(rx, ry), rz), t)



class ShiftedSoftPlus(nn.Softplus):
    def __init__(self, beta=1, origin=0.5, threshold=20):
        super(ShiftedSoftPlus, self).__init__(beta, threshold)
        self.origin = origin
        self.sp0 = F.softplus(torch.zeros(1) + self.origin, self.beta, self.threshold).item()

    def forward(self, input):
        return F.softplus(input + self.origin, self.beta, self.threshold) - self.sp0