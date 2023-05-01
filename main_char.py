'''
Date: 2022-11-23 11:29:36
LastEditors: yuhhong
LastEditTime: 2022-12-12 12:57:28
'''
import os
import argparse
import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import random

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.metrics import roc_auc_score, accuracy_score

from datasets.dataset_cls import ChiralityDataset
from models.dgcnn import DGCNN
from models.molnet import MolNet 
from models.pointnet import PointNet
from models.schnet import SchNet



def cls_criterion(outputs, targets):
    if len(outputs.size()) == 1:
        outputs = torch.unsqueeze(outputs, dim=1)
    if len(targets.size()) == 1:
        targets = torch.unsqueeze(targets, dim=1)

    loss = nn.BCELoss(reduction="mean")(outputs, targets)
    return loss

def train(model, device, loader, optimizer, accum_iter, batch_size, num_points, out_cls): 
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")): 
        _, x, mask, y, env = batch
        x = x.to(device).to(torch.float32)
        x = x.permute(0, 2, 1)
        mask = mask.to(device).to(torch.float32)
        y = y.to(device)
        y = F.one_hot(y, num_classes=out_cls).to(torch.float32)
        env = env.to(device).to(torch.float32)
        # env = F.normalize(env)

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        model.train()
        pred = model(x, env, idx_base)
        pred = F.softmax(pred, dim=0)
          
        loss = cls_criterion(pred, y)
        # normalize loss to account for batch accumulation
        loss = loss / accum_iter 
        loss.backward()

        # optimizer.step()
        # optimizer.zero_grad()
        # weights update
        if ((step + 1) % accum_iter == 0) or (step + 1 == len(loader)):
            optimizer.step()
            optimizer.zero_grad()

        y_true.append(y.detach().cpu())
        y_pred.append(pred.detach().cpu())
    
    y_true = torch.cat(y_true, dim = 0) 
    y_pred = torch.cat(y_pred, dim = 0)
    return y_true, y_pred

def eval(model, device, loader, batch_size, num_points, out_cls): 
    model.eval()
    y_true = []
    y_pred = []
    names = []
    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        name, x, mask, y, env = batch
        x = x.to(device).to(torch.float32)
        x = x.permute(0, 2, 1)
        mask = mask.to(device).to(torch.float32)
        y = y.to(device)
        y = F.one_hot(y, num_classes=out_cls).to(torch.float32)
        env = env.to(device).to(torch.float32)
        # env = F.normalize(env)

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        with torch.no_grad(): 
            pred = model(x, env, idx_base)
            pred = F.softmax(pred, dim=0)

        y_true.append(y.detach().cpu())
        y_pred.append(pred.detach().cpu())
        names.extend(name)
        
    y_true = torch.cat(y_true, dim = 0) # torch.Size([856, 2000])
    y_pred = torch.cat(y_pred, dim = 0)
    return names, y_true, y_pred

def batch_filter(supp): 
    for mol in supp: # remove empty molecule
        if mol is None:
            continue
        if len(Chem.MolToMolBlock(mol).split("\n")) <= 6: 
            continue
        yield mol



if __name__ == "__main__": 
    # Training settings
    parser = argparse.ArgumentParser(description='Molecular Properties Prediction')
    parser.add_argument('--config', type=str, default = './configs/molnet_bbbp.yaml',
                        help='Path to configuration')
    parser.add_argument('--csp_no', type=int, default=0,
                        help='charility phase number [0, 19]')
    parser.add_argument('--log_dir', type=str, default="./logs/molnet_bbbp/", 
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint', type=str, default = '', 
                        help='path to save checkpoint')
    parser.add_argument('--resume_path', type=str, default='', 
                        help='Pretrained model path')
    parser.add_argument('--transfer', action='store_true', 
                        help='Whether to load the pretrained encoder')

    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if config['model'] == 'molnet': 
        model = MolNet(config['model_para'], args.device).to(device)
    elif config['model'] == 'dgcnn':
        model = DGCNN(config['model_para'], args.device).to(device) 
    elif config['model'] == 'pointnet': 
        model = PointNet(config['model_para'], args.device).to(device) 
    elif config['model'] == 'schnet': 
        model = SchNet(config['model_para'], args.device).to(device)
    else:
        raise ValueError('Not implemented model')
    num_params = sum(p.numel() for p in model.parameters())
    # print(f'{str(model)} #Params: {num_params}')
    print('#Params: {}'.format(num_params))
    
    print("Loading the data...")
    supp = Chem.SDMolSupplier(config['paths']['train_data'])
    train_set = ChiralityDataset([item for item in batch_filter(supp)], 
                                num_points=config['model_para']['num_atoms'], 
                                csp_no=args.csp_no, 
                                data_augmentation=False)
    train_indices = train_set.balance_indices(list(range(len(train_set)))) # j0sie: please use this line to make balance sampling
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(train_set,
                                batch_size=config['train_para']['batch_size'],
                                num_workers=config['train_para']['num_workers'],
                                drop_last=True,
                                sampler=train_sampler)
    print('Load {} balanced training data from {}.'.format(len(train_set), config['paths']['train_data']))

    supp = Chem.SDMolSupplier(config['paths']['valid_data'])
    valid_set = ChiralityDataset([item for item in batch_filter(supp)], 
                                num_points=config['model_para']['num_atoms'], 
                                csp_no=args.csp_no, 
                                data_augmentation=False)
    valid_loader = DataLoader(valid_set,
                                batch_size=config['train_para']['batch_size'],
                                num_workers=config['train_para']['num_workers'],
                                drop_last=True)
    print('Load {} test data from {}.'.format(len(valid_set), config['paths']['valid_data']))
    
    optimizer = optim.Adam(model.parameters(), 
                            lr=config['train_para']['lr'], 
                            weight_decay=config['train_para']['weight_decay'])
    scheduler = MultiStepLR(optimizer, 
                            milestones=config['train_para']['scheduler']['milestones'], 
                            gamma=config['train_para']['scheduler']['gamma'])
    best_valid_auc = 0
    best_valid_acc = 0
    
    if args.resume_path != '': 
        if args.transfer: 
            print("Load the pretrained encoder...")
            state_dict = torch.load(args.resume_path)['model_state_dict']
            encoder_dict = {}
            for name, param in state_dict.items():
                if name.startswith("encoder"): 
                    encoder_dict[name] = param
            model.load_state_dict(encoder_dict, strict=False)
        else:
            print("Load the checkpoints...")
            model.load_state_dict(torch.load(args.resume_path)['model_state_dict'])
            optimizer.load_state_dict(torch.load(args.resume_path)['optimizer_state_dict'])
            scheduler.load_state_dict(torch.load(args.resume_path)['scheduler_state_dict'])
            best_valid_auc = torch.load(args.resume_path)['best_val_auc']

    model.to(device) 

    if args.checkpoint != '':
        checkpoint_dir = "/".join(args.checkpoint.split('/')[:-1])
        os.makedirs(checkpoint_dir, exist_ok = True)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    for epoch in range(1, config['train_para']['epochs'] + 1): 
        print("\n=====Epoch {}".format(epoch))

        print('Training...')
        y_true, y_pred = train(model, device, train_loader, optimizer, config['train_para']['accum_iter'], config['train_para']['batch_size'], config['model_para']['num_atoms'], config['model_para']['out_channels'])
        try: 
            train_auc = roc_auc_score(np.array(y_true), y_pred, multi_class='ovo',)
        except:
            train_auc = 0

        y_true = torch.argmax(y_true, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        train_acc = accuracy_score(y_true, y_pred)
        
        print('Evaluating...')
        names, y_true, y_pred = eval(model, device, valid_loader, config['train_para']['batch_size'], config['model_para']['num_atoms'], config['model_para']['out_channels'])
        try: 
            valid_auc = roc_auc_score(np.array(y_true), y_pred, multi_class='ovo',)
        except:
            valid_auc = 0

        y_true = torch.argmax(y_true, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        valid_acc = accuracy_score(y_true, y_pred)
        
        print("Train ACC: {} Train AUC: {}\nValid ACC: {} Valid AUC: {}\n".format(train_acc, train_auc, valid_acc, valid_auc))

        if args.log_dir != '':
            writer.add_scalar('valid/auc', valid_auc, epoch)
            writer.add_scalar('train/auc', train_auc, epoch)

        # if valid_auc > best_valid_auc: 
        #     best_valid_auc = valid_auc
        if valid_acc > best_valid_acc: 
            best_valid_acc = valid_acc
            best_valid_auc = valid_auc
            if args.checkpoint != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_auc': best_valid_auc, 'num_params': num_params}
                torch.save(checkpoint, args.checkpoint)
        
        scheduler.step()
        
        print('Best ACC so far: {}'.format(best_valid_acc))
        print('Best AUC so far: {}'.format(best_valid_auc))

    if args.log_dir != '':
        writer.close()

