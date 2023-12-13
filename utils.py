'''
Date: 2022-07-20 15:36:44
LastEditors: yuhhong
LastEditTime: 2022-11-11 23:29:18
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def avg_res(preds): 
	avg_preds = []
	for pred in preds.Pred.values:
		avg_preds.append(pred.split(','))

	avg_preds = np.array(avg_preds, dtype=np.float32)
	if len(avg_preds.shape) == 1: # only one configuration
		return ','.join(avg_preds.astype('str'))
	else: # more than one configuration, so average the prediction
		avg_preds = np.average(avg_preds, axis=0)
		avg_preds = F.softmax(torch.from_numpy(avg_preds), dim=0).numpy()
		return ','.join(avg_preds.astype('str'))

def average_results_on_enantiomers(df): 
	g = df.groupby(['SMILES', 'MB'])
	avg_df = g.apply(avg_res).to_frame('Pred')
	avg_df = avg_df.merge(df, on=['SMILES', 'MB']).rename(columns={'Pred_x': 'Pred_avg', 'Pred_y': 'Pred'})
	avg_df = avg_df.reset_index()
	return avg_df

def set_seed(seed): 
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)

def CE_loss(y_hat, y): 
	CE = nn.CrossEntropyLoss()
	return CE(y_hat, y)

def BCE_loss(y_hat, y): 
	BCE = torch.nn.BCEWithLogitsLoss()
	return BCE(y_hat, y)

def MSE(y_hat, y): 
	MSE = torch.mean(torch.square(y - y_hat))
	return MSE

def triplet_loss(z_anchor, z_positive, z_negative, margin=1.0, reduction='mean', distance_metric='euclidean'): 
	if distance_metric == 'euclidean' or distance_metric == 'euclidean_normalized':
		criterion = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(p=2.0), 
			margin=margin, 
			swap=False, 
			reduction=reduction)
	elif distance_metric == 'manhattan':
		criterion = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(p=1.0), 
			margin=margin, 
			swap=False, 
			reduction=reduction)
	elif distance_metric == 'cosine':
		criterion = nn.TripletMarginWithDistanceLoss(distance_function= lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y),  
			margin=margin, 
			swap=False, 
			reduction=reduction)
	else:
		raise Exception(f'distance metric {distance_metric} is not implemented')

	if distance_metric == 'euclidean_normalized':
		z_anchor = z_anchor / torch.linalg.norm(z_anchor + 1e-10, dim=1, keepdim=True)
		z_positive = z_positive / torch.linalg.norm(z_positive + 1e-10, dim=1, keepdim=True)
		z_negative = z_negative / torch.linalg.norm(z_negative + 1e-10, dim=1, keepdim=True)

	loss = criterion(z_anchor, z_positive, z_negative)
	return loss