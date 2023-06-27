'''
Date: 2022-11-21 15:53:39
LastEditors: yuhhong
LastEditTime: 2022-11-22 23:35:52
'''
import pandas as pd
from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import argparse

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

ATOM_LIST = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I']
CSP_DICT = {'91423': [0, 2], 
			'91119': [1, 2],
			'90879': [2, 2], 
			'90704': [3, 2], 
			'90589': [4, 2], 
			'90357': [5, 2], 
			'90246': [6, 2], 
			'90211': [7, 2], 
			'45173': [8, 2], 
			'45167': [9, 2], 
			'44869': [10, 2], 
			'23735': [11, 2], 
			'15723': [12, 2], 
			'394': [13, 2], 
			'91518': [14, 1],
			'2': [15, 1], 
			'45172': [16, 1], 
			'3575': [17, 2], 
			} # the largest 20 phases & reported in the paper

def convert2cls(chir, csp_category): 
    if csp_category == '1': 
        # For polysaccharide CSPs:
        if chir < 1.15:
            y = 0
        elif chir < 1.2:
            y = 1
        elif chir < 2.1:
            y = 2
        else:
            y = 3
    elif csp_category == '2': 
        # For Pirkle CSPs:
        if chir < 1.05: 
            y = 0
        elif chir < 1.15:
            y = 1
        elif chir < 2: 
            y = 2
        else:
            y = 3
    else:
        raise Exception("The category for CSP should be 1 or 2, rather than {}.".format(csp_category))
    return y
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--input', type=str, default = '',
						help='path to input data')
	parser.add_argument('--output', type=str, default = '',
						help='path to output data')
	args = parser.parse_args()

	supp = Chem.SDMolSupplier(args.input)
	out_mols = []
	mobile_phase = []
	print('Get {} data from {}'.format(len(supp), args.input))

	# filter out the noise molecules (double- or triple- labeled) &
	# filter out by conditions
	df_dict = {'SMILES': [], 'MB': [], 'K2/K1': [], 'Y': []}
	for mol in supp: 
		if mol is None:
			continue

		# remove invalid molecular blocks
		# e.x.
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
			mb = 'unknown'
		if mb not in CSP_DICT.keys():
			continue

		if mol.HasProp('csp_no'):
			mb = mol.GetProp('csp_no')
		else:
			mb = 'unknown'
		smiles = Chem.MolToSmiles(mol)
		chir = round(float(mol.GetProp('k2/k1')), 4)
		y = convert2cls(chir, str(CSP_DICT[mb][1]))
		
		df_dict['SMILES'].append(smiles)
		df_dict['K2/K1'].append(chir)
		df_dict['MB'].append(mb)
		df_dict['Y'].append(y)
	df = pd.DataFrame.from_dict(df_dict)
	# df_uniq = df[df.duplicated(subset=['SMILES', 'MB', 'Y'])==False]
	df_uniq = df.sort_values(['SMILES', 'MB', 'K2/K1'], ascending=False).drop_duplicates(['SMILES', 'MB'], keep='first').sort_index()

	# convert dataframe into mol_list
	out_mols = []
	for idx, row in df_uniq.iterrows(): 
		mol = Chem.MolFromSmiles(row['SMILES'])
		mol.SetProp('k2/k1', str(row['K2/K1']))
		mb = row['MB']
		mol.SetProp('encode_mobile_phase', str(CSP_DICT[mb][0]))
		mol.SetProp('mobile_phase_category', str(CSP_DICT[mb][1]))
		out_mols.append(mol)

	# old version: did not remove the noise molecules
	# for idx, mol in enumerate(supp):
	# 	if mol is None:
	# 		continue

	# 	# remove invalid molecular blocks
	# 	# e.x.
	# 	# 22      RDKit          2D
	# 	# 0
	# 	# 39  25 28  0  0  0  0  0  0  0  0999 V2000
	# 	# 69 10001.078110001.2576    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
	# 	# 69  9998.940610000.0077    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
	# 	# 69  9996.7988 9997.9394    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
	# 	# ......
	# 	# 12   4  5  1  0
	# 	# ......
	# 	mol_block = Chem.MolToMolBlock(mol).split("\n")
	# 	mol_block_length = sum([1 for d in mol_block if len(d)==69 and len(d.split())==16])
	# 	if mol_block_length < mol.GetNumAtoms(): 
	# 		print(mol_block_length, '<', mol.GetNumAtoms())
	# 		continue
	# 	if mol.GetNumAtoms() > 100: # --num_atoms 100
	# 		print('Too many atoms')
	# 		continue

	# 	flag_remove = False
	# 	for atom in mol.GetAtoms():
	# 		if atom.GetSymbol() not in ATOM_LIST:
	# 			flag_remove = True
	# 			print('Unlabeled atom: {}'.format(atom.GetSymbol()))
	# 			break
	# 	if flag_remove: 
	# 		continue
		
	# 	# if mol.HasProp('mobile_phase'): 
	# 	# 	mb = mol.GetProp('mobile_phase')
	# 	if mol.HasProp('csp_no'):
	# 		mb = mol.GetProp('csp_no')
	# 	else:
	# 		mb = 'unknown'
	# 	if mb not in CSP_DICT.keys():
	# 		continue
		
	# 	mol.SetProp('encode_mobile_phase', str(CSP_DICT[mb][0]))
	# 	mol.SetProp('mobile_phase_category', str(CSP_DICT[mb][1]))
	# 	out_mols.append(mol)

	print('Writing {} data to {}'.format(len(out_mols), args.output))
	w = Chem.SDWriter(args.output)
	for m in out_mols:
		w.write(m)
	print('Done!')