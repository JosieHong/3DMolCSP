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
        _, x, mask, y = batch
        x = x.to(device).to(torch.float32)
        x = x.permute(0, 2, 1)
        mask = mask.to(device).to(torch.float32)
        y = y.to(device)
        y = F.one_hot(y, num_classes=out_cls).to(torch.float32)

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        model.train()
        pred = model(x, None, idx_base)
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
        name, x, mask, y = batch
        x = x.to(device).to(torch.float32)
        x = x.permute(0, 2, 1)
        mask = mask.to(device).to(torch.float32)
        y = y.to(device)
        y = F.one_hot(y, num_classes=out_cls).to(torch.float32)

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        with torch.no_grad(): 
            pred = model(x, None, idx_base)
            pred = F.softmax(pred, dim=0)

        y_true.append(y.detach().cpu())
        y_pred.append(pred.detach().cpu())
        names.extend(name)

    y_true = torch.cat(y_true, dim = 0) 
    y_pred = torch.cat(y_pred, dim = 0)
    return names, y_true, y_pred

def batch_filter(supp): 
    for mol in supp: # remove empty molecule
        if mol is None:
            continue
        if len(Chem.MolToMolBlock(mol).split("\n")) <= 6: 
            continue
        yield mol

def load_data_fold(dataset, split_indices, fold_i, num_workers, batch_size): 
    train_indices = []
    valid_indices = []
    for i, indices in enumerate(split_indices): 
        if i != fold_i:
            train_indices += indices
        else:
            valid_indices += indices
    
    train_indices = dataset.balance_indices(train_indices) # j0sie: please use this line to make balance sampling
    print('# train: {}, # valid: {}'.format(len(train_indices), len(valid_indices)))

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                drop_last=True,
                                sampler=train_sampler)
    valid_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                drop_last=True,
                                sampler=valid_sampler)
    return train_loader, valid_loader 



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

    # freezing the encode model
    # for name, param in model.named_parameters():
    #     if name.split('.')[0] in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv']: # encode parts
    #         param.requires_grad = False

    # freezing the trnet
    # for name, param in model.named_parameters():
    #     if name.split('.')[0] == 'tr_net': # TRNet
    #         param.requires_grad = False

    # --------------- K-Fold Validation --------------- # 
    print("Loading the data...") 
    supp = Chem.SDMolSupplier(config['paths']['all_data'])
    dataset = ChiralityDataset([item for item in batch_filter(supp)], 
                                num_points=config['model_para']['num_atoms'], 
                                csp_no=args.csp_no, 
                                data_augmentation=False)
    print('Load {} data from {}.'.format(len(dataset), config['paths']['all_data']))
    # split the indices into K_FOLD
    K_FOLD = 10
    each_chunk = len(dataset) // K_FOLD
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    print('dataset size: {} \nchunk size: {}'.format(len(indices), each_chunk))
    split_indices = []
    for i in range(K_FOLD): 
        split_indices.append(indices[i*each_chunk: (i+1)*each_chunk])

    records = {'best_acc': [], 'best_auc': []}
    for fold_i in range(K_FOLD): 
        print('\n# --------------- Fold-{} --------------- #'.format(fold_i)) 
        train_loader, valid_loader = load_data_fold(dataset, 
                                                    split_indices, 
                                                    fold_i,
                                                    num_workers=config['train_para']['num_workers'], 
                                                    batch_size=config['train_para']['batch_size'],)
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

        early_stop_step = 10
        early_stop_patience = 0
        for epoch in range(1, config['train_para']['epochs'] + 1): 
            print("\n=====Epoch {}".format(epoch))

            print('Training...')
            y_true, y_pred = train(model, device, train_loader, optimizer, config['train_para']['accum_iter'], config['train_para']['batch_size'], config['model_para']['num_atoms'], config['model_para']['out_channels'])
            train_auc = roc_auc_score(np.array(y_true), y_pred, multi_class='ovo',)
            
            y_true = torch.argmax(y_true, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            train_acc = accuracy_score(y_true, y_pred)
            
            print('Evaluating...')
            names, y_true, y_pred = eval(model, device, valid_loader, config['train_para']['batch_size'], config['model_para']['num_atoms'], config['model_para']['out_channels'])
            try: 
                valid_auc = roc_auc_score(np.array(y_true), y_pred, multi_class='ovo',)
            except:
                valid_auc = np.nan
                
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
                early_stop_patience = 0
                print('Early stop patience reset')
            else:
                early_stop_patience += 1
                print('Early stop count: {}/{}'.format(early_stop_patience, early_stop_step))

            scheduler.step()
            print('Best ACC so far: {}'.format(best_valid_acc))
            print('Best AUC so far: {}'.format(best_valid_auc))

            if early_stop_patience == early_stop_step:
                print('Early stop!')
                break

        if args.log_dir != '':
            writer.close()

        records['best_acc'].append(best_valid_acc)
        records['best_auc'].append(best_valid_auc)

    print('\n# --------------- Final Results --------------- #')
    for i, (acc, auc) in enumerate(zip(records['best_acc'], records['best_auc'])):
        print('fold_{}: acc: {}, auc: {}'.format(i, acc, auc))
    print('mean acc: {}, mean auc: {}'.format(sum(records['best_acc'])/len(records['best_acc']), sum(records['best_auc'])/len(records['best_auc'])))

'''
5-fold >>> 

Without balance sample: 
fold_0: acc: 0.6354166666666666, auc: 0.6942516728595217
fold_1: acc: 0.6953125, auc: 0.7290257515429867
fold_2: acc: 0.78125, auc: 0.839265944881246
fold_3: acc: 0.8098958333333334, auc: 0.8988830091186358
fold_4: acc: 0.8385416666666666, auc: 0.9050540931693934
mean acc: 0.7520833333333333, mean auc: 0.8132960943143568

With balance sample: 
fold_0: acc: 0.9036458333333334, auc: 0.9309637185411722
fold_1: acc: 0.8854166666666666, auc: 0.9089343698228641
fold_2: acc: 0.90625, auc: 0.9062275656412213
fold_3: acc: 0.9088541666666666, auc: 0.9333536479849031
fold_4: acc: 0.8984375, auc: 0.8824923704281337
mean acc: 0.9005208333333332, mean auc: 0.9123943344836588

10-fold >>> 

Without balance sample: 
fold_0: acc: 0.6953125, auc: 0.7573797771933334
fold_1: acc: 0.734375, auc: 0.8146456646888997
fold_2: acc: 0.7578125, auc: 0.7916062983950916
fold_3: acc: 0.8359375, auc: 0.9089281024478056
fold_4: acc: 0.8359375, auc: 0.9519108310479468
fold_5: acc: 0.8828125, auc: 0.9720815456109574
fold_6: acc: 0.8359375, auc: 0.9139328327353985
fold_7: acc: 0.8828125, auc: 0.9719635703743864
fold_8: acc: 0.8359375, auc: 0.9194201354364931
fold_9: acc: 0.8515625, auc: 0.9592259395215583
mean acc: 0.81484375, mean auc: 0.8961094697451873

With balance sample: 
fold_0: acc: 0.8828125, auc: 0.9684653719676651
fold_1: acc: 0.90625, auc: 0.9715461902026584
fold_2: acc: 0.921875, auc: 0.937265721955329
fold_3: acc: 0.9453125, auc: 0.9919787613520855
fold_4: acc: 0.9375, auc: 0.9775647416835748
fold_5: acc: 0.9375, auc: 0.975049060835436
fold_6: acc: 0.9296875, auc: 0.9732568318069807
fold_7: acc: 0.9765625, auc: 0.9916803652022358
fold_8: acc: 0.9296875, auc: 0.996549814616826
fold_9: acc: 0.9140625, auc: 0.96843895452918
mean acc: 0.928125, mean auc: 0.9751795814151972
'''