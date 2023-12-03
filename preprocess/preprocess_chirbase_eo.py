'''
Date: 2022-11-21 15:53:39
LastEditors: yuhhong
LastEditTime: 2022-11-22 23:35:52
'''
import os
import argparse
import pandas as pd
pd.set_option('display.max_columns', None)
import pprint
from tqdm import tqdm
tqdm.pandas()
import numpy as np
import pickle
import random 

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, rdMolAlign, PandasTools, rdMolTransforms
from rdkit.Chem.Draw import rdDepictor

from utils import ATOM_LIST, create_X

def find_pos(index, max_index): 
	remainder = index % 2
	candidate_indexes = [idx for i, idx in enumerate(range(max_index)) if i % 2 == remainder]
	return random.choice(candidate_indexes)

def gen_conf(smiles, conf_type):
	if conf_type == 'etkdg': 
		mol = Chem.MolFromSmiles(smiles)
		if mol == None: return None
		mol_from_smiles = Chem.AddHs(mol)
		AllChem.EmbedMolecule(mol_from_smiles)

	elif conf_type == 'etkdgv3': 
		mol = Chem.MolFromSmiles(smiles)
		if mol == None: return None
		mol_from_smiles = Chem.AddHs(mol)
		AllChem.EmbedMolecule(mol_from_smiles, AllChem.ETKDGv3()) 

	elif conf_type == '2d':
		mol = Chem.MolFromSmiles(smiles)
		if mol == None: return None
		mol_from_smiles = Chem.AddHs(mol)
		rdDepictor.Compute2DCoords(mol_from_smiles)

	elif conf_type == 'omega':
		# print("Is GPU ready? (True/False)", oeomega.OEOmegaIsGPUReady())
		mol_from_smiles = oechem.OEMol()
		oechem.OEParseSmiles(mol_from_smiles, smiles)
		oechem.OESuppressHydrogens(mol_from_smiles)
		
		# First we set up Omega
		omega = oeomega.OEOmega() 
		omega.SetMaxConfs(1) # Only generate one conformer for our molecule
		omega.SetStrictStereo(True) # Set to False to pick random stereoisomer if stereochemistry is not specified (not relevant here)
		omega.SetStrictAtomTypes(False) # Be a little loose about atom typing to ensure parameters are available to omega for all molecules
		omega(mol_from_smiles)
		
	else: 
		raise ValueError("Undifined conformer type: {}".format(conf_type))
	return mol_from_smiles

def align_conf(df): 
	assert len(df) == 2, '2 molecules are needed to be aligned, but {} are given.'.format(len(df))
	mol1 = df.iloc[0]['Mol']
	mol2 = df.iloc[1]['Mol']
	# conf11 = mol1.GetConformer().GetPositions()
	# conf12 = mol2.GetConformer().GetPositions()
	rmsd, trans = rdMolAlign.GetAlignmentTransform(mol1, mol2)
	rdMolTransforms.TransformConformer(mol1.GetConformer(0), trans) # tranform the mol1 as alignment
	# conf21 = mol1.GetConformer().GetPositions()
	# conf22 = mol2.GetConformer().GetPositions()
	# print(np.array_equal(conf11, conf21), np.array_equal(conf12, conf22))

	# change the value in original df
	df.at[0, 'Mol'] = mol1
	df.at[1, 'Mol'] = mol2
	df['Align_RMSD'] = rmsd
	return df

'''
preprocess: 
	1. remove the invalid 3d conformer
	2. remove the molecules with unlabeled atoms 
		(we only label the atoms: ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I'])
	3. remove the molecules with more than 300 atom 
		(it can be changed, but how about using 300 temporarily)
	4. generate conformations
'''

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--input_eo', type=str, required=True, 
						help='path to input data (elution order)')
	parser.add_argument('--input', type=str, required=True, 
						help='path to input data (original data)')
	parser.add_argument('--conf_type', type=str, default='etkdgv3', 
						choices=['2d', 'etkdg', 'etkdgv3', 'omega'], 
						help='conformation type')
	parser.add_argument('--csp_setting', type=str, required=True, 
						help='path to csp settings')
	parser.add_argument('--output', type=str, required=True, 
						help='path to output data')
	parser.add_argument('--test_ratio', type=float, default = 0.1,
						help='test ratio')
	args = parser.parse_args()

	# -------------------------------------
	# data with enantiomers (elution order)
	# -------------------------------------
	supp = Chem.SDMolSupplier(args.input_eo)
	print('Get {} data from {}'.format(len(supp), args.input_eo))

	# filter the molecules
	df_dict = {'SMILES': [], 'SMILES_iso': [], 'CSP_NO': [], 'Elution_Order': []}
	for mol in supp: 
		if mol is None:
			continue

		mol_block = Chem.MolToMolBlock(mol).split("\n")
		mol_block_length = sum([1 for d in mol_block if len(d)==69 and len(d.split())==16])
		if mol_block_length < mol.GetNumAtoms(): 
			print(mol_block_length, '<', mol.GetNumAtoms())
			continue
		if mol.GetNumAtoms() >= 100: # --num_atoms 100
			print('Too many atoms')
			continue

		flag_remove = False
		for atom in mol.GetAtoms(): 
			if atom.GetSymbol() not in ATOM_LIST:
				flag_remove = True
				print('Unlabeled atom: {}'.format(atom.GetSymbol()))
				break

		if flag_remove: 
			continue

		smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
		smiles_iso = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))

		df_dict['SMILES'].append(smiles)
		df_dict['SMILES_iso'].append(smiles_iso)
		df_dict['CSP_NO'].append(str(mol.GetProp('csp_no')))
		df_dict['Elution_Order'].append(str(int(mol.GetProp('class'))))

	df = pd.DataFrame.from_dict(df_dict)
	df = df.drop_duplicates(subset=['SMILES_iso', 'CSP_NO']) # same smiles_iso, same csp_no but diff elution order
	
	# -------------------------------------
	# data without enantiomers (k2/k1)
	# -------------------------------------
	supp = Chem.SDMolSupplier(args.input)
	print('Get {} data from {}'.format(len(supp), args.input))

	df_org_dict = {'SMILES': [], 'CSP_NO': [], 'K2/K1': []}
	for mol in supp: 
		if mol is None:
			continue
		if not mol.HasProp('csp_no'): 
			print('Unknow mobile phase')
			continue

		smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
		df_org_dict['SMILES'].append(smiles)
		df_org_dict['CSP_NO'].append(str(mol.GetProp('csp_no')))
		df_org_dict['K2/K1'].append(round(float(mol.GetProp('k2/k1')), 4))

	df_org = pd.DataFrame.from_dict(df_org_dict)
	df_org = df_org.sort_values(['SMILES', 'CSP_NO', 'K2/K1'], ascending=False).drop_duplicates(['SMILES', 'CSP_NO'], keep='first').sort_index()
	print(df_org['K2/K1'].min(), df_org['K2/K1'].max())

	# -------------------------------------
	# csp settings (csp category)
	# -------------------------------------
	assert os.path.exists(args.csp_setting)
	df_csp = pd.read_csv(args.csp_setting)
	df_csp['CSP_ID'] = df_csp['CSP_ID'].astype(str)
	CSP_DICT = {i: [e, c] for i, e, c in zip(df_csp['CSP_ID'].tolist(), 
									df_csp['CSP_Encode'].tolist(), 
									df_csp['CSP_Category'].tolist())}
	print('Load the CSP settings: ')
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(CSP_DICT)

	# -------------------------------------
	# final
	# -------------------------------------
	# test
	# df = df[:100]

	# merge k2/k1, csp_category, and elution order together
	df = df.merge(df_org, left_on=['SMILES', 'CSP_NO'], right_on=['SMILES', 'CSP_NO'])
	df['CSP_Category'] = df['CSP_NO'].apply(lambda x: int(CSP_DICT[x][1]))

	# convert dataframe into mol_list and generate conformations 
	print('Generating conformations...')
	df = df.groupby('SMILES').filter(lambda x: len(x) == 2)
	# generate conformations
	df['Mol'] = df['SMILES_iso'].progress_apply(lambda x: gen_conf(x, args.conf_type))
	df['Conf_Number'] = df['Mol'].apply(lambda x: int(x.GetNumConformers()))
	df = df[df['Conf_Number'] >= 1]
	# align enantiomers
	print('Align enantiomers...')
	df = df.groupby(['SMILES'], as_index=False).progress_apply(align_conf)
	# remove nan
	df = df.dropna()
	df.reset_index(inplace=True, drop=True)
	df.reset_index(inplace=True)
	df['ena_index'] = df['index'].progress_apply(lambda x: x//2*2 + (x+1)%2)
	df['pos_index'] = df['index'].progress_apply(lambda x: find_pos(x, len(df)))
	print(df.head())
	
	# save to csv
	df[['index', 'ena_index', 'pos_index', 'SMILES', 'SMILES_iso', 'Elution_Order']].to_csv(args.output.replace('.sdf', '.csv'))
	print('Save results to {}'.format(args.output.replace('.sdf', '.csv')))
	
	# save to sdf
	# print('Writing {} data to {}'.format(len(df), args.output))
	# PandasTools.WriteSDF(df, args.output, molColName='Mol', properties=list(df.columns))
	# print('Done!')

	# save to pkl
	data_pkl = []
	for mol in df['Mol'].tolist(): 
		x = create_X(mol, num_points=100)
		data_pkl.append({'anchor': x})
	for idx, row in df.iterrows():
		data_pkl[idx]['smiles_iso'] = row['SMILES_iso']
		data_pkl[idx]['smiles'] = row['SMILES']
		data_pkl[idx]['neg'] = data_pkl[row['ena_index']]['anchor']
		data_pkl[idx]['pos'] = data_pkl[row['pos_index']]['anchor']
		data_pkl[idx]['k2/k1'] = float(row['K2/K1'])
		data_pkl[idx]['csp_category'] = int(row['CSP_Category'])
		data_pkl[idx]['elution_order'] = int(row['Elution_Order'])

	# split the data by smiles
	smiles_list = list(set(df['SMILES'].tolist()))
	Ltest = np.random.choice(smiles_list, int(len(smiles_list)*args.test_ratio), replace=False)
	print("Get {} training compounds, {} test compounds".format(len(smiles_list)-len(Ltest), len(Ltest)))

	train_data_pkl = []
	test_data_pkl = []
	for d in data_pkl: 
		if d['smiles'] in Ltest: 
			test_data_pkl.append(d)
		else:
			train_data_pkl.append(d)

	out_path = args.output.replace('.sdf', '_train.pkl')
	with open(out_path, 'wb') as f: 
		pickle.dump(train_data_pkl, f)
		print('Save {}'.format(out_path))
	out_path = args.output.replace('.sdf', '_test.pkl')
	with open(out_path, 'wb') as f: 
		pickle.dump(test_data_pkl, f)
		print('Save {}'.format(out_path))
	print('Done!')
	