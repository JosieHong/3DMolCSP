'''
Date: 2022-11-21 15:53:39
LastEditors: yuhhong
LastEditTime: 2022-11-22 23:35:52
'''
import os
import argparse
import pprint
import pandas as pd

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils import ATOM_LIST, convert2cls

'''
preprocess: 
  1. remove the invalid 3d conformer
  2. remove the molecules with unlabeled atoms 
	  (we only label the atoms: ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I'])
  3. remove the molecules with more than 300 atom 
	  (it can be changed, but how about using 300 temporarily)
  4. gather all the mobile_phase & encode the mobile_phase
  5. remove the double- and triple- labeled molecules
'''

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--input', type=str, required=True, 
						help='path to input data')
	parser.add_argument('--csp_setting', type=str, required=True, 
						help='path to csp settings')
	parser.add_argument('--output', type=str, required=True, 
						help='path to output data')
	args = parser.parse_args()

	# load the csp settings
	assert os.path.exists(args.csp_setting)
	df_csp = pd.read_csv(args.csp_setting)
	df_csp['CSP_ID'] = df_csp['CSP_ID'].astype(str)
	CSP_DICT = {i: [e, c] for i, e, c in zip(df_csp['CSP_ID'].tolist(), 
									df_csp['CSP_Encode'].tolist(), 
									df_csp['CSP_Category'].tolist())}
	print('Load the CSP settings: ')
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(CSP_DICT)

	supp = Chem.SDMolSupplier(args.input)
	out_mols = []
	mobile_phase = []
	print('Get {} data from {}'.format(len(supp), args.input))

	# filter out the noise molecules (double- or triple- labeled) &
	# filter out by conditions
	df_dict = {'SMILES': [], 'SMILES_iso': [], 'MB': [], 'K2/K1': [], 'Chiral_Tag': []}
	for mol in supp: 
		if mol is None:
			continue

		# remove invalid molecular blocks
		# e.g. 
		# 22      RDKit          2D
		# 0
		# 39  25 28  0  0  0  0  0  0  0  0999 V2000
		# 69 10001.078110001.2576    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
		# 69  9998.940610000.0077    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
		# 69  9996.7988 9997.9394    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
		# ......
		# 12   4  5  1  0
		# ......
		mol_block = Chem.MolToMolBlock(mol).split("\n")
		mol_block_length = sum([1 for d in mol_block if len(d)==69 and len(d.split())==16])
		if mol_block_length < mol.GetNumAtoms(): 
			print(mol_block_length, '<', mol.GetNumAtoms())
			continue
		if mol.GetNumAtoms() > 100: # --num_atoms 100
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
		
		# if mol.HasProp('mobile_phase'): 
		# 	mb = mol.GetProp('mobile_phase')
		if mol.HasProp('csp_no'): 
			mb = mol.GetProp('csp_no')
		else:
			print('Unknow mobile phase')
			continue
		if mb not in CSP_DICT.keys(): 
			print('Undefined mobile phase: {}'.format(mb))
			continue

		smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
		smiles_iso = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))
		chir = round(float(mol.GetProp('k2/k1')), 4)
		# y = convert2cls(chir, str(CSP_DICT[mb][1]))
		chir_tag = ';'.join([str(atom.GetChiralTag()) for atom in Chem.MolFromSmiles(smiles_iso).GetAtoms()])

		df_dict['SMILES'].append(smiles)
		df_dict['SMILES_iso'].append(smiles_iso)
		df_dict['K2/K1'].append(chir)
		df_dict['MB'].append(mb)
		# df_dict['Y'].append(y)
		df_dict['Chiral_Tag'].append(chir_tag)

	df = pd.DataFrame.from_dict(df_dict)
	# df_uniq = df[df.duplicated(subset=['SMILES', 'MB', 'Y'])==False] 
	df_uniq = df.sort_values(['SMILES', 'Chiral_Tag', 'MB', 'K2/K1'], ascending=False).drop_duplicates(['SMILES', 'Chiral_Tag', 'MB'], keep='first').sort_index()

	# convert dataframe into mol_list
	out_mols = []
	for idx, row in df_uniq.iterrows(): 
		mol = Chem.MolFromSmiles(row['SMILES_iso'])
		mol.SetProp('k2/k1', str(row['K2/K1']))
		mb = row['MB']
		mol.SetProp('encode_mobile_phase', str(CSP_DICT[mb][0]))
		mol.SetProp('mobile_phase_category', str(CSP_DICT[mb][1]))
		out_mols.append(mol)

	print('Writing {} data to {}'.format(len(out_mols), args.output))
	w = Chem.SDWriter(args.output)
	for m in out_mols:
		w.write(m)
	print('Done!')