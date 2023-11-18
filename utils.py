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

def avg_res(preds): 
	preds = preds.Pred.values
	avg_preds = []
	for pred in preds:
		avg_preds.append(pred.split(','))

	avg_preds = np.array(avg_preds, dtype=np.float32)
	avg_preds = np.average(avg_preds, axis=0)
	return ','.join(avg_preds.astype('str'))

def average_results_on_enantiomers(df): 
	g = df.groupby(['ID', 'SMILES', 'MB'])
	avg_df = g.apply(avg_res).to_frame('Pred').reset_index()
	return avg_df

def set_seed(seed): 
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)

def cls_criterion(outputs, targets): 
	targets = torch.squeeze(targets)
	if len(outputs.size()) == 1:
		loss = nn.BCELoss()(outputs, targets.float())
	else: 
		loss = nn.CrossEntropyLoss()(outputs, targets.to(torch.int64))
	return loss