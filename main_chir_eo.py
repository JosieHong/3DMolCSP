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

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset import ChiralityDataset_EO
from model import MolNet_CSP
from utils import set_seed, BCE_loss, get_lr, triplet_loss

def custom_replace(tensor, on_zero, on_non_zero): 
    # we create a copy of the original tensor, 
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

def train(model, device, loader, optimizer, batch_size, num_points): 
	y_true = []
	y_pred = []
	loss1_list = []
	# loss2_list = []
	cos1_list = []
	cos2_list = []
	with tqdm(total=len(loader)) as bar: 
		for step, batch in enumerate(loader): 
			_, _, pos, neg, anchor, y = batch
			pos = pos.to(device).to(torch.float32)
			pos = pos.permute(0, 2, 1)
			neg = neg.to(device).to(torch.float32)
			neg = neg.permute(0, 2, 1)
			anchor = anchor.to(device).to(torch.float32)
			anchor = anchor.permute(0, 2, 1)
			y = y.to(device)
			idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

			model.train()
			emb_pos, pred = model(pos, idx_base)
			emb_neg, pred2 = model(neg, idx_base)
			emb_anchor, pred3 = model(anchor, idx_base)
			# print(emb.size(), emb2.size(), COS(emb, emb2))
			# print('pred', pred.size())
			# print('y', y.size())

			loss = BCE_loss(pred, y.float()) + BCE_loss(pred2, custom_replace(y, on_zero=1., on_non_zero=0.).float())
			# loss2 = triplet_loss(emb_anchor, emb_pos, emb_neg, distance_metric='euclidean_normalized')
			# loss = loss1 + loss2
			loss.backward()

			optimizer.step()
			optimizer.zero_grad()

			y_true.append(y.detach().cpu())
			y_pred.append(pred.detach().cpu())

			bar.set_description('Train')
			bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
			bar.update(1)

			loss1_list.append(loss.item())
			# loss2_list.append(loss2.item())
			COS = nn.CosineSimilarity(dim=1, eps=1e-6)
			cos1_list.append(torch.mean(COS(emb_anchor, emb_neg)).item())
			cos2_list.append(torch.mean(COS(emb_anchor, emb_pos)).item())

	y_true = torch.cat(y_true, dim=0)
	y_pred = torch.cat(y_pred, dim=0)
	print('loss1 (cls): {}, loss2 (emb): {}'.format(np.mean(np.array(loss1_list)), None))
	print('cos1 (anchor-neg): {}, cos2 (anchor-pos): {}'.format(np.mean(np.array(cos1_list)), np.mean(np.array(cos2_list))))
	return y_true, y_pred

def eval(model, device, loader, batch_size, num_points): 
	model.eval()
	y_true = []
	y_pred = []
	smiles_list = []
	cos1_list = []
	cos2_list = []
	for _, batch in enumerate(tqdm(loader, desc="Iteration")):
		smiles_iso, smiles, pos, neg, anchor, y = batch
		pos = pos.to(device).to(torch.float32)
		pos = pos.permute(0, 2, 1)
		neg = neg.to(device).to(torch.float32)
		neg = neg.permute(0, 2, 1)
		anchor = anchor.to(device).to(torch.float32)
		anchor = anchor.permute(0, 2, 1)
		y = y.to(device)

		idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

		with torch.no_grad(): 
			emb_pos, pred = model(pos, idx_base)
			emb_neg, _ = model(neg, idx_base)
			emb_anchor, _ = model(anchor, idx_base)
			# print('pred', pred.size())
			# print('y', y.size())

		y_true.append(y.detach().cpu())
		y_pred.append(pred.detach().cpu())
		smiles_list.extend(smiles_iso)

		COS = nn.CosineSimilarity(dim=1, eps=1e-6)
		cos1_list.append(torch.mean(COS(emb_anchor, emb_neg)).item())
		cos2_list.append(torch.mean(COS(emb_anchor, emb_pos)).item())

	y_true = torch.cat(y_true, dim=0) 
	y_pred = torch.cat(y_pred, dim=0)
	print('cos1 (anchor-neg): {}, cos2 (anchor-pos): {}'.format(np.mean(np.array(cos1_list)), np.mean(np.array(cos2_list))))
	return smiles_list, y_true, y_pred



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
	model = MolNet_CSP(config['model_para'], args.device, out_emb=True).to(device)
	num_params = sum(p.numel() for p in model.parameters())
	# print(f'{str(model)} #Params: {num_params}')
	print('#Params: {}'.format(num_params))
	
	print("Loading the data...")
	train_set = ChiralityDataset_EO(config['paths']['train_data'], 
								num_points=config['model_para']['num_atoms'])
	train_loader = DataLoader(train_set,
								batch_size=config['train_para']['batch_size'],
								num_workers=config['train_para']['num_workers'],
								drop_last=True,
								shuffle=True)
	print('Load {} training data from {}.'.format(len(train_set), config['paths']['train_data']))

	valid_set = ChiralityDataset_EO(config['paths']['valid_data'], 
								num_points=config['model_para']['num_atoms'])
	valid_loader = DataLoader(valid_set,
								batch_size=1, 
								num_workers=config['train_para']['num_workers'],
								drop_last=True,
								shuffle=True)
	print('Load {} test data from {}.'.format(len(valid_set), config['paths']['valid_data']))
	
	optimizer = optim.AdamW(model.parameters(), lr=config['train_para']['lr'], weight_decay=1e-5)
	# optimizer = optim.SGD(model.parameters(), lr=config['train_para']['lr'], weight_decay=1e-5, momentum=0.5)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

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

	early_stop_step = 40
	early_stop_patience = 0
	for epoch in range(1, config['train_para']['epochs'] + 1): 
		print("\n=====Epoch {}".format(epoch))

		print('Training...')
		y_true, y_pred = train(model, device, train_loader, optimizer, 
								config['train_para']['batch_size'], 
								config['model_para']['num_atoms'])
		print(y_true[:8])
		print(y_pred[:8])
		if config['model_para']['out_channels'] == 1: 
			y_pred_binary = torch.where(y_pred > 0.5, 1., 0.)
		else: 
			y_pred_binary = torch.argmax(y_pred, dim=1)
		train_auc = roc_auc_score(y_true, y_pred, multi_class='ovr',)
		train_acc = accuracy_score(y_true, y_pred_binary)

		print('Evaluating...')
		smiles_list, y_true, y_pred = eval(model, device, valid_loader, 1, 
											config['model_para']['num_atoms'])
		print(y_true[:8])
		print(y_pred[:8])
		if config['model_para']['out_channels'] == 1:
			y_pred_binary = torch.where(y_pred > 0.5, 1., 0.)
		else:
			y_pred_binary = torch.argmax(y_pred, dim=1)
		valid_auc = roc_auc_score(y_true, y_pred, multi_class='ovr',)
		valid_acc = accuracy_score(y_true, y_pred_binary)
		
		print("Train ACC: {} Train AUC: {}\nValid ACC: {} Valid AUC: {}\n".format(train_acc, train_auc, valid_acc, valid_auc))

		if args.log_dir != '':
			writer.add_scalar('valid/auc', valid_auc, epoch)
			writer.add_scalar('train/auc', train_auc, epoch)

		# if valid_auc > best_valid_auc or valid_acc >= best_valid_acc:
		if valid_auc > best_valid_auc: 
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

		# scheduler.step()
		scheduler.step(valid_auc) # ReduceLROnPlateau
		print('Best ACC so far: {}'.format(best_valid_acc))
		print('Best AUC so far: {}'.format(best_valid_auc))

		if early_stop_patience == early_stop_step:
			print('Early stop!')
			break

	if args.log_dir != '':
		writer.close()

	if args.result_path != '':
		print("Load the best checkpoints...")
		model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])

		print("Inference on test set:")
		smiles_list_test, y_true_test, y_pred_test = eval(model, device, valid_loader, 1, 
														config['model_para']['num_atoms'])
		if config['model_para']['out_channels'] == 1:
			y_pred_test_binary = torch.where(y_pred_test > 0.5, 1., 0.)
			test_res = {'SMILES': smiles_list, 'True': y_true, 
						'Pred Final': y_pred_test_binary.squeeze().tolist(), 
						'Pred': y_pred_test.squeeze().tolist()}
		else:
			y_pred_test_binary = torch.argmax(y_pred_test, dim=1)
			y_pred_out = []
			for y in y_pred_test:
				y_pred_out.append(','.join([str(i) for i in y.tolist()]))
			test_res = {'SMILES': smiles_list_test, 'True': y_true_test, 
						'Pred Final': y_pred_test_binary.squeeze().tolist(), 
						'Pred': y_pred_out}
		df_test = pd.DataFrame.from_dict(test_res)
		df_test.to_csv(args.result_path)
		print('Save the results to {}'.format(args.result_path))

		print("Inference on training set:")
		train_loader = DataLoader(train_set,
								batch_size=1,
								num_workers=config['train_para']['num_workers'],
								drop_last=True)
		smiles_list_train, y_true_train, y_pred_train = eval(model, device, train_loader, 1, 
															config['model_para']['num_atoms'])
		if config['model_para']['out_channels'] == 1: 
			y_pred_train_binary = torch.where(y_pred_train > 0.5, 1., 0.)
			train_res = {'SMILES': smiles_list_train, 'True': y_true_train, 
						'Pred Final': y_pred_train_binary.squeeze().tolist(), 
						'Pred': y_pred_train.squeeze().tolist()}
		else:
			y_pred_train_binary = torch.argmax(y_pred_train, dim=1)
			y_pred_out = []
			for y in y_pred_train:
				y_pred_out.append(','.join([str(i) for i in y.tolist()]))
			train_res = {'SMILES': smiles_list_train, 'True': y_true_train, 
						'Pred Final': y_pred_train_binary.squeeze().tolist(), 
						'Pred': y_pred_out}
		df_train = pd.DataFrame.from_dict(train_res)
		df_train.to_csv(args.result_path.replace('.csv', '_train.csv'))
		print('Save the results to {}'.format(args.result_path.replace('.csv', '_train.csv')))
	