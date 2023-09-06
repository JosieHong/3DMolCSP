'''
Date: 2021-11-30 13:55:12
LastEditors: yuhhong
LastEditTime: 2022-08-01 16:39:53
'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class  FCResBlock(nn.Module): 
    def __init__(self, in_dim, out_dim, dropout=None): 
        super(FCResBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
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

        return self.fc(x)

