'''
Date: 2021-10-20 14:30:36
LastEditors: yuhhong
LastEditTime: 2022-11-21 15:18:06
'''
import numpy as np

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import argparse



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--input', type=str, default = '',
						help='path to input data')
	parser.add_argument('--use_isomeric_smiles', action='store_true', 
						help='Whether to use isomeric smiles')
	parser.add_argument('--output_train', type=str, default = '',
						help='path to output data')
	parser.add_argument('--output_test', type=str, default = '',
						help='path to output data')
	args = parser.parse_args()


	suppl = Chem.SDMolSupplier(args.input)
	mols = [x for x in suppl if x != None]

	
	# output the smiles list
	SMILES_OUT_PATH = '/'.join(args.input.split('/')[:-1]) + '/SMILES_list.txt'
	if args.use_isomeric_smiles:
		smiles = list(set([Chem.MolToSmiles(m, isomericSmiles=True) for m in mols]))
	else:
		smiles = list(set([Chem.MolToSmiles(m, isomericSmiles=False) for m in mols]))


	smiles_out = "\n".join(smiles)
	with open(SMILES_OUT_PATH, 'w') as f:
		f.write(smiles_out)
	print("Load {} / {} data from {}".format(len(mols), len(smiles), args.input))


	# split the data by smiles
	Ltest = np.random.choice(smiles, int(len(smiles)*0.1), replace=False)
	Ltrain = [x for x in smiles if x not in Ltest]
	print("Get {} training data, {} test data".format(len(Ltrain), len(Ltest)))


	# wite the training and test data
	w_train = Chem.SDWriter(args.output_train)
	w_test = Chem.SDWriter(args.output_test)
	for m in mols:
		if args.use_isomeric_smiles:
			s = Chem.MolToSmiles(m, isomericSmiles=True)
		else: 
			s = Chem.MolToSmiles(m, isomericSmiles=False)

		if s in Ltest:
			w_test.write(m)
		else:
			w_train.write(m)
	print("Save training and test data to {} and {}".format(args.output_train, args.output_test))
