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
import pandas as pd

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

from datasets.dataset_cls import ChiralityDataset_infer
from models.dgcnn import DGCNN
from models.molnet import MolNet 
from models.pointnet import PointNet
from models.schnet import SchNet
from utils import set_seed, average_results_on_enantiomers

TEST_BATCH_SIZE = 1 # global variable in inference

def inference(model, device, loader, num_points, out_cls, csp_num): 
	model.eval()
	y_pred = []
	smiles_list = []
	id_list = []
	mbs = []
	for _, batch in enumerate(tqdm(loader, desc="Iteration")): 
		mol_id, smiles_iso, mb, x, mask = batch
		x = x.to(device).to(torch.float32)
		x = x.permute(0, 2, 1)
		mask = mask.to(device).to(torch.float32)

		idx_base = torch.arange(0, TEST_BATCH_SIZE, device=device).view(-1, 1, 1) * num_points

		with torch.no_grad(): 
			pred = model(x, None, idx_base)

		y_pred.append(pred.view(TEST_BATCH_SIZE, -1).detach().cpu())
		smiles_list.extend(smiles_iso)
		id_list.extend(mol_id)
		mbs.extend(mb.tolist())

	y_pred = torch.cat(y_pred, dim=0)
	return id_list, smiles_list, mbs, y_pred

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
	parser.add_argument('--config', type=str, required=True, 
						help='Path to configuration')
	parser.add_argument('--csp_no', type=int, default=0, required=True, 
						help='charility phase number [0, 19]')
	parser.add_argument('--resume_path', type=str, default='', required=True, 
						help='Pretrained model path')
	parser.add_argument('--result_path', type=str, default='', required=True,
						help='Results path')

	parser.add_argument('--device', type=int, default=0,
						help='which gpu to use if any (default: 0)')
	parser.add_argument('--no_cuda', type=bool, default=False,
						help='enables CUDA training')

	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	set_seed(42)

	results_dir = "/".join(args.result_path.split('/')[:-1])
	os.makedirs(results_dir, exist_ok = True)
	print('Create the results directory, {}'.format(results_dir))

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
	supp = Chem.SDMolSupplier(config['paths']['test_data'])
	test_set = ChiralityDataset_infer([item for item in batch_filter(supp)], 
										num_points=config['model_para']['num_atoms'], 
										csp_no=args.csp_no, 
										flipping=False)
	supp_ena = Chem.SDMolSupplier(config['paths']['test_data'])
	test_set_ena = ChiralityDataset_infer([item for item in batch_filter(supp_ena)], 
										num_points=config['model_para']['num_atoms'], 
										csp_no=args.csp_no, 
										flipping=True)
	test_set = ConcatDataset([test_set, test_set_ena]) # concat two configurations' datasets
	test_loader = DataLoader(test_set,
								batch_size=TEST_BATCH_SIZE, 
								num_workers=config['train_para']['num_workers'],
								drop_last=True,)
	print('Load {} test data from {}.'.format(len(test_set), config['paths']['test_data']))

	print("Load the model...")
	model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])
	model.to(device) 

	print('Evaluating...')
	id_list, smiles_list, mbs, y_pred = inference(model, device, test_loader, 
													config['model_para']['num_atoms'], 
													config['model_para']['out_channels'],
													config['model_para']['csp_num'])
	y_pred_out = []
	for y in y_pred:
		y_pred_out.append(','.join([str(i) for i in y.tolist()]))

	res_df = pd.DataFrame({'ID': id_list, 'SMILES': smiles_list, 'MB': mbs, 'Pred': y_pred_out})
	print('Average the results of enantiomers...')
	res_df = average_results_on_enantiomers(res_df)
	res_df.to_csv(args.result_path, sep='\t')
	print('Save the test results to {}'.format(args.result_path))