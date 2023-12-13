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
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import random
from sklearn.preprocessing import OneHotEncoder

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset import ChiralityDataset
from model import MolNet_CSP 
from utils import set_seed, average_results_on_enantiomers, CE_loss

TEST_BATCH_SIZE = 1 # global variable in validation



def train(model, device, loader, optimizer, batch_size, num_points):
	y_true = []
	y_pred = []
	for step, batch in enumerate(tqdm(loader, desc="Iteration")): 
		_, _, _, x, y = batch
		x = x.to(device).to(torch.float32)
		x = x.permute(0, 2, 1)
		y = y.to(device)
		idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

		model.train()
		pred = model(x, idx_base)
		# print('pred', pred.size())

		loss = CE_loss(pred, y)
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
	mbs = []
	for _, batch in enumerate(tqdm(loader, desc="Iteration")): 
		mol_id, smiles_iso, mb, x, y = batch
		x = x.to(device).to(torch.float32)
		x = x.permute(0, 2, 1)
		y = y.to(device)

		idx_base = torch.arange(0, TEST_BATCH_SIZE, device=device).view(-1, 1, 1) * num_points

		with torch.no_grad(): 
			pred = model(x, idx_base)

		y_true.append(y.detach().cpu())
		y_pred.append(pred.detach().cpu())
		smiles_list.extend(smiles_iso)
		id_list.extend(mol_id)
		mbs.extend(mb.tolist())

	y_true = torch.cat(y_true, dim=0) 
	y_pred = torch.cat(y_pred, dim=0)
	return id_list, smiles_list, mbs, y_true, y_pred

def batch_filter(supp): 
	for mol in supp: # remove empty molecule
		if mol is None:
			continue
		if len(Chem.MolToMolBlock(mol).split("\n")) <= 6: 
			continue
		yield mol

def load_data_fold(dataset, dataset_ena, split_indices, fold_i, num_workers, batch_size): 
	train_indices = []
	valid_indices = []
	for i, indices in enumerate(split_indices): 
		if i != fold_i:
			train_indices += indices
		else:
			valid_indices += indices
	
	train_indices = dataset.balance_indices(train_indices) # make balance sampling
	print('# train: {}, # valid: {}'.format(len(train_indices), len(valid_indices)))

	train_indices += [i+len(dataset) for i in train_indices] # add enantiomers (use the same indexes for two configurations prohibit data leaking)
	valid_indices += [i+len(dataset) for i in valid_indices]
								
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(valid_indices)

	all_dataset = ConcatDataset([dataset, dataset_ena]) # concat two configurations' datasets
	train_loader = DataLoader(all_dataset,
								batch_size=batch_size,
								num_workers=num_workers,
								drop_last=True,
								sampler=train_sampler)
	valid_loader = DataLoader(all_dataset,
								batch_size=TEST_BATCH_SIZE, 
								num_workers=num_workers,
								drop_last=True,
								sampler=valid_sampler)
	return train_loader, valid_loader 



if __name__ == "__main__":
	# Training settings
	parser = argparse.ArgumentParser(description='3DMolCSP (train in k-fold)')
	parser.add_argument('--config', type=str, default = './configs/molnet_train_s.yaml',
						help='Path to configuration')
	parser.add_argument('--csp_no', type=int, default=0,
						help='Charility phase number [0, 19]')
	parser.add_argument('--k_fold', type=int, default=10,
						help='k for k-fold validation')
	parser.add_argument('--log_dir', type=str, default="./logs/", 
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

	set_seed(42)

	# load the configuration file
	with open(args.config, 'r') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	
	device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

	# --------------- K-Fold Validation --------------- # 
	print("Loading the data...") 
	supp = Chem.SDMolSupplier(config['paths']['all_data'])
	dataset = ChiralityDataset([item for item in batch_filter(supp)], 
								num_points=config['model_para']['num_atoms'], 
								csp_no=args.csp_no, 
								flipping=False)
	supp_ena = Chem.SDMolSupplier(config['paths']['all_data'])
	dataset_ena = ChiralityDataset([item for item in batch_filter(supp_ena)], 
								num_points=config['model_para']['num_atoms'], 
								csp_no=args.csp_no, 
								flipping=True)
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
		model = MolNet_CSP(config['model_para'], args.device).to(device)
		num_params = sum(p.numel() for p in model.parameters())
		# print(f'{str(model)} #Params: {num_params}')
		print('#Params: {}'.format(num_params))

		train_loader, valid_loader = load_data_fold(dataset, dataset_ena, 
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
		
		# modify the path to check_point
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
				model.load_state_dict(torch.load(resume_path_fold, map_location=device)['model_state_dict'])
				optimizer.load_state_dict(torch.load(resume_path_fold, map_location=device)['optimizer_state_dict'])
				scheduler.load_state_dict(torch.load(resume_path_fold, map_location=device)['scheduler_state_dict'])
				best_valid_auc = torch.load(resume_path_fold, map_location=device)['best_val_auc']

		if args.log_dir != '':
			writer = SummaryWriter(log_dir=args.log_dir)

		early_stop_step = 5
		early_stop_patience = 0
		for epoch in range(1, config['train_para']['epochs'] + 1): 
			print("\n=====Epoch {}".format(epoch))

			print('Training...')
			y_true, y_pred = train(model, device, train_loader, optimizer, 
									config['train_para']['batch_size'], 
									config['model_para']['num_atoms'])
			train_auc = roc_auc_score(np.array(y_true), y_pred, multi_class='ovr',)
			y_pred = torch.argmax(y_pred, dim=1)
			train_acc = accuracy_score(y_true, y_pred)

			print('Evaluating...')
			id_list, smiles_list, mbs, y_true, y_pred = eval(model, device, valid_loader, 
															config['train_para']['batch_size'], 
															config['model_para']['num_atoms'])
			try: 
				valid_auc = roc_auc_score(np.array(y_true), y_pred, multi_class='ovr',)
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
				best_valid_acc = valid_acc
				best_valid_auc = valid_auc
				if args.checkpoint != '':
					print('Saving checkpoint...')
					checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 
									'optimizer_state_dict': optimizer.state_dict(), 
									'scheduler_state_dict': scheduler.state_dict(), 
									'best_val_auc': best_valid_auc, 
									'num_params': num_params}
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
			model.load_state_dict(torch.load(check_point_fold, map_location=device)['model_state_dict'])
			optimizer.load_state_dict(torch.load(check_point_fold, map_location=device)['optimizer_state_dict'])
			scheduler.load_state_dict(torch.load(check_point_fold, map_location=device)['scheduler_state_dict'])
			best_valid_auc = torch.load(check_point_fold, map_location=device)['best_val_auc']
			
			print('Evaluating...')
			id_list, smiles_list, mbs, y_true, y_pred = eval(model, device, valid_loader, 
															config['train_para']['batch_size'], 
															config['model_para']['num_atoms'])
			y_pred_out = []
			for y in y_pred:
				y_pred_out.append(','.join([str(i) for i in y.tolist()]))

			res_df = pd.DataFrame({'ID': id_list, 'SMILES': smiles_list, 'MB': mbs, 'Class': y_true, 'Pred': y_pred_out})
			print('Average the results of enantiomers...')
			res_df = average_results_on_enantiomers(res_df)
			print(res_df.head(), res_df.columns)
			res_df.to_csv(result_path_fold, sep='\t')
			print('Save the test results to {}'.format(result_path_fold))

		del model # remove the model from GPU

	print('\n# --------------- Final Results --------------- #')
	for i, (acc, auc) in enumerate(zip(records['best_acc'], records['best_auc'])):
		print('fold_{}: acc: {:.2f}, auc: {:.2f}'.format(i, acc, auc))
	print('mean acc: {:.2f}, mean auc: {:.2f}'.format(sum(records['best_acc'])/len(records['best_acc']), sum(records['best_auc'])/len(records['best_auc'])))
