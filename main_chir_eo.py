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
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import random

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset import ChiralityDataset_EO
from model import MolNet_CSP 
from utils import set_seed, cls_criterion



def train(model, device, loader, optimizer, batch_size, num_points): 
	y_true = []
	y_pred = []
	for step, batch in enumerate(tqdm(loader, desc="Iteration")): 
		_, _, x, mask, y = batch
		x = x.to(device).to(torch.float32)
		x = x.permute(0, 2, 1)
		mask = mask.to(device).to(torch.float32)
		y = y.to(device)
		idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

		model.train()
		pred = model(x, None, idx_base)
		# print('pred', pred.size())

		loss = cls_criterion(pred, y)
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

		y_true.append(y.detach().cpu())
		y_pred.append(pred.detach().cpu())

	y_true = torch.cat(y_true, dim=0)
	y_pred = torch.cat(y_pred, dim=0)
	return y_true, y_pred

def eval(model, device, loader, batch_size, num_points): 
	model.eval()
	y_true = []
	y_pred = []
	smiles_list = []
	id_list = []
	for _, batch in enumerate(tqdm(loader, desc="Iteration")):
		mol_id, smiles_iso, x, mask, y = batch
		x = x.to(device).to(torch.float32)
		x = x.permute(0, 2, 1)
		mask = mask.to(device).to(torch.float32)
		y = y.to(device)

		idx_base = torch.arange(0, 2, device=device).view(-1, 1, 1) * num_points

		with torch.no_grad(): 
			pred = model(x, None, idx_base)

		y_true.append(y.detach().cpu())
		y_pred.append(pred.detach().cpu())
		smiles_list.extend(smiles_iso)
		id_list.extend(mol_id)

	y_true = torch.cat(y_true, dim=0) 
	y_pred = torch.cat(y_pred, dim=0)
	return id_list, smiles_list, y_true, y_pred

def batch_filter(supp): 
	for mol in supp: # remove empty molecule
		if mol is None:
			continue
		if len(Chem.MolToMolBlock(mol).split("\n")) <= 6: 
			continue
		yield mol



if __name__ == "__main__": 
	# Training settings
	parser = argparse.ArgumentParser(description='3DMolCSP for elution order prediction')
	parser.add_argument('--config', type=str, default = '',
						help='Path to configuration')
	parser.add_argument('--log_dir', type=str, default="./logs/", 
						help='tensorboard log directory')
	parser.add_argument('--checkpoint', type=str, default = '', 
						help='path to save checkpoint')
	parser.add_argument('--transfer', action='store_true', 
						help='whether to load the pretrained encoder')
	parser.add_argument('--resume_path', type=str, default='', 
						help='pretrained model path or checkpoint path')
	parser.add_argument('--result_path', type=str, default='', 
						help='results path')

	parser.add_argument('--device', type=int, default=0,
						help='which gpu to use if any (default: 0)')
	parser.add_argument('--no_cuda', type=bool, default=False,
						help='enables CUDA training')

	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	set_seed(42)

	# load the configuration file
	with open(args.config, 'r') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	
	device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
	model = MolNet_CSP(config['model_para'], args.device).to(device)
	num_params = sum(p.numel() for p in model.parameters())
	# print(f'{str(model)} #Params: {num_params}')
	print('#Params: {}'.format(num_params))
	
	print("Loading the data...")
	supp = Chem.SDMolSupplier(config['paths']['train_data'])
	train_set = ChiralityDataset_EO([item for item in batch_filter(supp)], 
								num_points=config['model_para']['num_atoms'])
	train_loader = DataLoader(train_set,
								batch_size=config['train_para']['batch_size'],
								num_workers=config['train_para']['num_workers'],
								drop_last=True)

	supp = Chem.SDMolSupplier(config['paths']['valid_data'])
	valid_set = ChiralityDataset_EO([item for item in batch_filter(supp)], 
								num_points=config['model_para']['num_atoms'])
	valid_loader = DataLoader(valid_set,
								batch_size=2, 
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
			state_dict = torch.load(args.resume_path, map_location=device)['model_state_dict']
			encoder_dict = {}
			for name, param in state_dict.items():
				if name.startswith("encoder"): 
					encoder_dict[name] = param
			model.load_state_dict(encoder_dict, strict=False)
		else:
			print("Load the checkpoints...")
			model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])
			optimizer.load_state_dict(torch.load(args.resume_path, map_location=device)['optimizer_state_dict'])
			scheduler.load_state_dict(torch.load(args.resume_path, map_location=device)['scheduler_state_dict'])
			best_valid_auc = torch.load(args.resume_path, map_location=device)['best_val_auc']

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
		y_true, y_pred = train(model, device, train_loader, optimizer, 
								config['train_para']['batch_size'], 
								config['model_para']['num_atoms'])
		train_auc = roc_auc_score(np.array(y_true), y_pred, multi_class='ovr',)
		y_pred_binary = torch.where(y_pred > 0.5, 1., 0.)
		train_acc = accuracy_score(y_true, y_pred_binary)

		print('Evaluating...')
		id_list, smiles_list, y_true, y_pred = eval(model, device, valid_loader, 
														config['train_para']['batch_size'], 
														config['model_para']['num_atoms'])
		
		valid_auc = roc_auc_score(np.array(y_true), y_pred, multi_class='ovr',)
		y_pred_binary = torch.where(y_pred > 0.5, 1., 0.)
		valid_acc = accuracy_score(y_true, y_pred_binary)
		
		print("Train ACC: {} Train AUC: {}\nValid ACC: {} Valid AUC: {}\n".format(train_acc, train_auc, valid_acc, valid_auc))

		if args.log_dir != '':
			writer.add_scalar('valid/auc', valid_auc, epoch)
			writer.add_scalar('train/auc', train_auc, epoch)

		if (not np.isnan(valid_auc) and valid_auc > best_valid_auc) or \
				(np.isnan(valid_auc) and valid_acc >= best_valid_acc): 
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

	if args.result_path != '':
		print("Load the best checkpoints...")
		model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])

		id_list_test, smiles_list_test, y_true_test, y_pred_test = eval(model, device, valid_loader, 1, 
														config_model_elution_order['num_atoms'])
		y_pred_test_binary = torch.where(y_pred_test > 0.5, 1., 0.)
		test_res = {'SMILES': smiles_list, 'True': y_true, 
					'Pred Final': y_pred_test_binary, 
					'Pred': [';'.join(p.astype('str')) for p in y_pred_test.detach().numpy()]}
		df_test = pd.DataFrame.from_dict(test_res)
		df_test.to_csv(args.result_path)

	