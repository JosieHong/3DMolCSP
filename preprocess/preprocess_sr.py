import os
import argparse
import pandas as pd
import pprint
from tqdm import tqdm
tqdm.pandas()
import pickle
import random 
import numpy as np

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdDepictor

from utils import molnet_filter, gen_conf_from_df, create_X



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--input_dir', type=str, default='./data/SRCls/', 
						help='path to input data (original data)')
	parser.add_argument('--conf_type', type=str, default='etkdgv3', 
						choices=['2d', 'etkdg', 'etkdgv3', 'omega'], 
						help='conformation type')
	parser.add_argument('--output_dir', type=str, required=True, 
						help='path to output data')
	args = parser.parse_args()

	# -------------------------------------
	# load pkl from ChlRo
	# -------------------------------------
	seed = 42
	valid_path = os.path.join(args.input_dir, "test_RS_classification_enantiomers_MOL_69719_11680_5840.pkl")
	valid_df = pd.read_pickle(valid_path)
	print('Load {} data from {}'.format(len(valid_df), valid_path))
	print(valid_df.head())
	train_path = os.path.join(args.input_dir, "train_RS_classification_enantiomers_MOL_326865_55084_27542.pkl")
	train_df = pd.read_pickle(train_path)
	print('Load {} data from {}'.format(len(train_df), train_path))
	print(train_df.head())
	exit()
	# sample one conformation for one configuration
	# train_df = train_df.groupby('ID').sample(1, random_state=seed).sort_values('SMILES_nostereo').reset_index(drop=True)
	# valid_df = valid_df.groupby('ID').sample(1, random_state=seed).sort_values('SMILES_nostereo').reset_index(drop=True)

	# filter
	train_df['pass_filter'] = train_df['ID'].apply(lambda x: molnet_filter(x))
	valid_df['pass_filter'] = valid_df['ID'].apply(lambda x: molnet_filter(x))
	print('# train succ: {}, # train fail: {}'.format(len(train_df[train_df['pass_filter'] == True]),
														len(train_df[train_df['pass_filter'] == False])))
	print('# valid succ: {}, # valid fail: {}'.format(len(valid_df[valid_df['pass_filter'] == True]),
														len(valid_df[valid_df['pass_filter'] == False])))
	train_df = train_df[train_df['pass_filter'] == True]
	valid_df = valid_df[valid_df['pass_filter'] == True]

	# asign positive and negative
	
	# -------------------------------------
	# final
	# -------------------------------------
	# test
	# train_df = train_df.iloc[:10]
	# valid_df = valid_df.iloc[:10]

	# convert dataframe into mol_list and generate conformations
	train_out = gen_conf_from_df(train_df, args.conf_type)
	valid_out = gen_conf_from_df(valid_df, args.conf_type)

	# save into pkl
	train_data_pkl = []
	for mol in train_out:
		x = create_X(mol, num_points=200)
		train_data_pkl.append({'mol': x,
								'id': mol.GetProp('id'), 
								'smiles': mol.GetProp('smiles'), 
								'chiral_tag': int(mol.GetProp('chiral_tag'))})
	out_path = os.path.join(args.output_dir, 'sr_train.pkl')
	with open(out_path, 'wb') as f: 
		pickle.dump(train_data_pkl, f)
		print('Save {}'.format(out_path))

	valid_data_pkl = []
	for mol in valid_out:
		x = create_X(mol, num_points=200)
		valid_data_pkl.append({'mol': x,
								'id': mol.GetProp('id'), 
								'smiles': mol.GetProp('smiles'), 
								'chiral_tag': int(mol.GetProp('chiral_tag'))})
	out_path = os.path.join(args.output_dir, 'sr_valid.pkl')
	with open(out_path, 'wb') as f: 
		pickle.dump(valid_data_pkl, f)
		print('Save {}'.format(out_path))
	
	print('Done!')
	
