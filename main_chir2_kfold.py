'''
Date: 2022-11-23 11:29:36
LastEditors: yuhhong
LastEditTime: 2022-12-12 12:57:28
'''
import os
import argparse
import numpy as np
import pandas as pd 
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
from sklearn.preprocessing import OneHotEncoder

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.metrics import roc_auc_score, accuracy_score

from datasets.dataset2_cls import ChiralityDataset
from models.dgcnn import DGCNN
from models.molnet2 import MolNet 
from models.pointnet import PointNet
from models.schnet import SchNet



def cal_roc_auc_score(y_true, y_pred, multi_class='ovr'): 
	y_true = y_true.reshape(-1, 1)
	enc = OneHotEncoder(categories=[[i for i in range(y_pred.shape[1])]])
	y_true = enc.fit_transform(y_true).toarray()
	# print('y_true', y_true.shape, 'y_pred', y_pred.shape)
	score = roc_auc_score(y_true, y_pred, multi_class=multi_class)
	return score1

def cls_criterion(outputs, targets): 
	# print('outputs', outputs.size(), 'targets', targets.size())
	targets = torch.squeeze(targets)
	
	# print('outputs', outputs.size(), 'targets', targets.size())
	# print(outputs[0, :], targets[0])
	loss = nn.CrossEntropyLoss()(outputs, targets.to(torch.int64))
	return loss

def train(model, device, loader, optimizer, accum_iter, batch_size, num_points, out_cls, csp_num):
	y_true = []
	y_pred = []
	for step, batch in enumerate(tqdm(loader, desc="Iteration")): 
		s1, s2, _, x1, x2, mask, y = batch
		x1 = x1.to(device).to(torch.float32)
		x1 = x1.permute(0, 2, 1) 
		x2 = x2.to(device).to(torch.float32)
		x2 = x2.permute(0, 2, 1) 
		mask = mask.to(device).to(torch.float32)
		y = y.to(device)
		idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

		model.train()
		pred = model(x1, x2, None, idx_base)
		# print('pred', pred.size())

		invalid_y = torch.isnan(y)
		if csp_num > 1: 
			invalid_pred = invalid_y.unsqueeze(2).repeat(1, 1, out_cls)
		else:
			invalid_pred = invalid_y
		loss = cls_criterion(pred[~invalid_pred].view(y.size(0), -1), y[~invalid_y])
		# normalize loss to account for batch accumulation
		loss = loss / accum_iter 
		loss.backward()

		# optimizer.step()
		# optimizer.zero_grad()
		# weights update
		if ((step + 1) % accum_iter == 0) or (step + 1 == len(loader)):
			optimizer.step()
			optimizer.zero_grad()

		y_true.append(y[~invalid_y].detach().cpu())
		y_pred.append(pred[~invalid_pred].view(y.size(0), -1).detach().cpu())

	y_true = torch.cat(y_true, dim=0)
	y_pred = torch.cat(y_pred, dim=0)
	return y_true, y_pred

def eval(model, device, loader, batch_size, num_points, out_cls, csp_num): 
	model.eval()
	y_true = []
	y_pred = []
	names1 = []
	names2 = []
	mbs = []
	for _, batch in enumerate(tqdm(loader, desc="Iteration")):
		name1, name2, mb, x1, x2, mask, y = batch
		x1 = x1.to(device).to(torch.float32)
		x1 = x1.permute(0, 2, 1)
		x2 = x2.to(device).to(torch.float32)
		x2 = x2.permute(0, 2, 1)
		mask = mask.to(device).to(torch.float32)
		y = y.to(device)

		idx_base = torch.arange(0, 2, device=device).view(-1, 1, 1) * num_points

		with torch.no_grad(): 
			pred = model(x1, x2, None, idx_base)

		invalid_y = torch.isnan(y)
		if csp_num > 1: 
			invalid_pred = invalid_y.unsqueeze(2).repeat(1, 1, out_cls)
		else:
			invalid_pred = invalid_y
		y_true.append(y[~invalid_y].detach().cpu())
		y_pred.append(pred[~invalid_pred].view(y.size(0), -1).detach().cpu())
		names1.extend(name1)
		names2.extend(name2)
		mbs.extend(mb.tolist())

	y_true = torch.cat(y_true, dim=0) 
	y_pred = torch.cat(y_pred, dim=0)
	return names1, names2, mbs, y_true, y_pred

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
								batch_size=2, 
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
						help='Charility phase number [0, 19]')
	parser.add_argument('--k_fold', type=int, default=10,
						help='k for k-fold validation')
	parser.add_argument('--log_dir', type=str, default="./logs/molnet_bbbp/", 
						help='Tensorboard log directory')
	parser.add_argument('--checkpoint', type=str, default = '', 
						help='Path to save checkpoint')
	parser.add_argument('--resume_path', type=str, default='', 
						help='Pretrained model path')
	parser.add_argument('--result_path', type=str, default='', 
						help='Results path')
	parser.add_argument('--transfer', action='store_true', 
						help='Whether to load the pretrained encoder')

	parser.add_argument('--device', type=int, default=0,
						help='Which gpu to use if any (default: 0)')
	parser.add_argument('--no_cuda', type=bool, default=False,
						help='Enables CUDA training')

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
								num_csp=config['model_para']['csp_num'], 
								csp_no=args.csp_no, 
								data_augmentation=False)
	print('Load {} data from {}.'.format(len(dataset), config['paths']['all_data']))
	# split the indices into k-fold
	each_chunk = len(dataset) // args.k_fold
	indices = list(range(len(dataset)))
	random.shuffle(indices)
	print('dataset size: {} \nchunk size: {}'.format(len(indices), each_chunk))
	split_indices = []
	for i in range(args.k_fold): 
		split_indices.append(indices[i*each_chunk: (i+1)*each_chunk])

	records = {'best_acc': [], 'best_auc': []}
	for fold_i in range(args.k_fold): 
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
		
		# josie: modify the path to check_point
		if args.checkpoint != '':
			check_point_fold = args.checkpoint.replace('.pt', '_{}.pt'.format(fold_i))
			print('Modify the path to checkpoint as: {}'.format(check_point_fold))

			checkpoint_dir = "/".join(args.checkpoint.split('/')[:-1])
			os.makedirs(checkpoint_dir, exist_ok = True)
			print('Create {}'.format(checkpoint_dir))

		if args.resume_path != '':
			resume_path_fold = args.resume_path.replace('.pt', '_{}.pt'.format(fold_i))
			print('Modify the path to resume_path as: {}'.format(resume_path_fold))
		if args.result_path != '':
			result_path_fold = args.result_path.replace('.csv', '_{}.csv'.format(fold_i))
			print('Modify the path to result_path as: {}'.format(result_path_fold))

			result_dir = "/".join(args.result_path.split('/')[:-1])
			os.makedirs(result_dir, exist_ok = True)
			print('Create {}'.format(result_dir))

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
				model.load_state_dict(torch.load(resume_path_fold)['model_state_dict'])
				optimizer.load_state_dict(torch.load(resume_path_fold)['optimizer_state_dict'])
				scheduler.load_state_dict(torch.load(resume_path_fold)['scheduler_state_dict'])
				best_valid_auc = torch.load(resume_path_fold)['best_val_auc']

		model.to(device) 

		if args.log_dir != '':
			writer = SummaryWriter(log_dir=args.log_dir)

		early_stop_step = 20
		early_stop_patience = 0
		for epoch in range(1, config['train_para']['epochs'] + 1): 
			print("\n=====Epoch {}".format(epoch))

			print('Training...')
			y_true, y_pred = train(model, device, train_loader, optimizer, 
									config['train_para']['accum_iter'], 
									config['train_para']['batch_size'], 
									config['model_para']['num_atoms'], 
									config['model_para']['out_channels'], 
									config['model_para']['csp_num'])
			train_auc = roc_auc_score(np.array(y_true), y_pred, multi_class='ovr',)
			# train_auc = cal_roc_auc_score(np.array(y_true), np.array(y_pred), multi_class='ovr',)
			y_pred = torch.argmax(y_pred, dim=1)
			train_acc = accuracy_score(y_true, y_pred)

			print('Evaluating...')
			names1, names2, mbs, y_true, y_pred = eval(model, device, valid_loader, 
												config['train_para']['batch_size'], 
												config['model_para']['num_atoms'], 
												config['model_para']['out_channels'],
												config['model_para']['csp_num'])
			try: 
				valid_auc = roc_auc_score(np.array(y_true), y_pred, multi_class='ovr',)
				# valid_auc = cal_roc_auc_score(np.array(y_true), np.array(y_pred), multi_class='ovr',)
			except: 
				valid_auc = np.nan
			
			y_pred = torch.argmax(y_pred, dim=1)
			valid_acc = accuracy_score(y_true, y_pred)
			
			print("Train ACC: {} Train AUC: {}\nValid ACC: {} Valid AUC: {}\n".format(train_acc, train_auc, valid_acc, valid_auc))

			if args.log_dir != '':
				writer.add_scalar('valid/auc', valid_auc, epoch)
				writer.add_scalar('train/auc', train_auc, epoch)

			if (not np.isnan(valid_auc) and valid_auc > best_valid_auc) or \
				(np.isnan(valid_auc) and valid_acc >= best_valid_acc): 
			# if valid_acc > best_valid_acc: 
			# if valid_acc >= best_valid_acc and valid_auc >= best_valid_auc: 
				best_valid_acc = valid_acc
				best_valid_auc = valid_auc
				if args.checkpoint != '':
					print('Saving checkpoint...')
					checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_auc': best_valid_auc, 'num_params': num_params}
					torch.save(checkpoint, check_point_fold)
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

		# output the best validation results
		if args.result_path: 
			print("Load the best results...")
			model.load_state_dict(torch.load(check_point_fold)['model_state_dict'])
			optimizer.load_state_dict(torch.load(check_point_fold)['optimizer_state_dict'])
			scheduler.load_state_dict(torch.load(check_point_fold)['scheduler_state_dict'])
			best_valid_auc = torch.load(check_point_fold)['best_val_auc']

			print('Evaluating...')
			names1, names2, mbs, y_true, y_pred = eval(model, device, valid_loader, 
												config['train_para']['batch_size'], 
												config['model_para']['num_atoms'], 
												config['model_para']['out_channels'],
												config['model_para']['csp_num'])
			# y_pred = torch.argmax(y_pred, dim=1) # we need to output the probabilities
			y_pred_out = []
			for y in y_pred:
				y_pred_out.append(','.join([str(i) for i in y.tolist()]))

			res_df = pd.DataFrame({'SMILES1': names1, 'SMILES2': names2, 'MB': mbs, 'Class': y_true, 'Pred': y_pred_out})
			res_df.to_csv(result_path_fold, sep='\t')
			print('Save the test results to {}'.format(result_path_fold))

	print('\n# --------------- Final Results --------------- #')
	for i, (acc, auc) in enumerate(zip(records['best_acc'], records['best_auc'])):
		print('fold_{}: acc: {}, auc: {}'.format(i, acc, auc))
	print('mean acc: {}, mean auc: {}'.format(sum(records['best_acc'])/len(records['best_acc']), sum(records['best_auc'])/len(records['best_auc'])))

