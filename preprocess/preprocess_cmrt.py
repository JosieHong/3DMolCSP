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
	parser.add_argument('--input', type=str, default = '',
						help='path to input data')
	parser.add_argument('--csp_setting', type=str, required=True, 
						help='path to csp settings')
	parser.add_argument('--output', type=str, default = '',
						help='path to output data')
	args = parser.parse_args()

	# load the csp settings
	assert os.path.exists(args.csp_setting)
	df_csp = pd.read_csv(args.csp_setting)
	CSP_DICT = {n: [e, c] for n, e, c in zip(df_csp['Short_Name'].tolist(), 
									df_csp['CSP_Encode'].tolist(), 
									df_csp['CSP_Category'].tolist()) if not pd.isnull(n)}
	print('Load the CSP settings: ')
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(CSP_DICT)

	# calculate k2/k1
	# k2/k1 ~~ (t2-2.9)/(t1-2.9)
	df_rt = pd.read_csv(args.input, index_col=0)

	df_rt['isomer_SMILES'] = df_rt['SMILES']
	df_rt['SMILES'] = df_rt['SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=False))
	df_rt = df_rt[df_rt['RT'] != 0]
	df_rt = df_rt.groupby(['index']).filter(lambda x: len(x) == 2)

	df_rt_alpha = df_rt.groupby('index').apply(lambda x: (x['RT'].max() - 2.9) / (x['RT'].min() -2.9)).to_frame(name='K2/K1')
	df_rt_alpha = df_rt_alpha.merge(df_rt[['index', 'SMILES', 'Column']], on='index', how='left')
	df_rt_alpha = df_rt_alpha[df_rt_alpha.duplicated()]


	# convert dataframe into mol_list
	out_mols = []
	for idx, row in df_rt_alpha.iterrows(): 
		mol = Chem.MolFromSmiles(row['SMILES'])

		# filter the molecules 
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

		mb_short = row['Column']
		if mb_short not in CSP_DICT.keys():
			continue

		mol.SetProp('k2/k1', str(row['K2/K1']))
		mol.SetProp('encode_mobile_phase', str(CSP_DICT[mb_short][0]))
		mol.SetProp('mobile_phase_category', str(CSP_DICT[mb_short][1]))
		out_mols.append(mol)

	print('Writing {} data to {}'.format(len(out_mols), args.output))
	w = Chem.SDWriter(args.output)
	for m in out_mols:
		w.write(m)
	print('Done!')