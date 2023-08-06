import os
import argparse
import pprint
import pandas as pd

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def mol_list2data_frame(supp):
	data = {'SMILES': [], 'SMILES_iso': [], 'mol': [], 'k2/k1': [], 'encode_mobile_phase': [], 'mobile_phase_category': []}
	for mol in supp: 
		smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
		smiles_iso = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))

		data['SMILES'].append(smiles)
		data['SMILES_iso'].append(smiles_iso)
		data['mol'].append(mol)
		data['k2/k1'].append(float(mol.GetProp('k2/k1')))
		data['encode_mobile_phase'].append(int(mol.GetProp('encode_mobile_phase')))
		data['mobile_phase_category'].append(mol.GetProp('mobile_phase_category'))

	return pd.DataFrame.from_dict(data)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--minuend', type=str, required=True, 
						help='sdf file1')
	parser.add_argument('--subtrahend', type=str, required=True, 
						help='sdf file2')
	parser.add_argument('--output', type=str, required=True, 
						help='path to output data')
	args = parser.parse_args()

	minuend_supp = Chem.SDMolSupplier(args.minuend)
	minuend_df = mol_list2data_frame(minuend_supp)
	print('Load {} data from {}'.format(len(minuend_df), args.minuend))
	
	subtrahend_supp = Chem.SDMolSupplier(args.subtrahend)
	subtrahend_df = mol_list2data_frame(subtrahend_supp)
	print('Load {} data from {}'.format(len(subtrahend_df), args.subtrahend))

	# refer: https://stackoverflow.com/questions/37313691/how-to-remove-a-pandas-dataframe-from-another-dataframe
	res_df = pd.concat([minuend_df, subtrahend_df, subtrahend_df]).drop_duplicates(keep=False)
	res_df = pd.concat([minuend_df[['SMILES', 'encode_mobile_phase']], 
						subtrahend_df[['SMILES', 'encode_mobile_phase']], 
						subtrahend_df[['SMILES', 'encode_mobile_phase']]]).drop_duplicates(keep=False)
	res_df = res_df.merge(minuend_df, on=['SMILES', 'encode_mobile_phase'], how='left')

	out_mols = res_df['mol'].tolist()
	print('Writing {} data to {}'.format(len(out_mols), args.output))
	w = Chem.SDWriter(args.output)
	for m in out_mols:
		w.write(m)
	print('Done!')
