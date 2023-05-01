'''
Date: 2022-11-21 15:53:39
LastEditors: yuhhong
LastEditTime: 2022-11-22 23:35:52
'''
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
'''

ATOM_LIST = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I']
CSP_NO = ['91027',
			'4297',
			'91119',
			'90704',
			'91423',
			'2',
			'90357',
			'91518',
			'3575',
			'15723',
			'90211',
			'394',
			'44869',
			'45172',
			'90589',
			'90879',
			'45167',
			'90246',
			'23735',
			'45173'] # the largest 20 phases

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
	for idx, mol in enumerate(supp):
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
		if mol.GetNumAtoms() > 300: # --num_atoms 300
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
		if mb not in CSP_NO:
			continue
		mol.SetProp('tmp', mb)
		mobile_phase.append(mb)
		out_mols.append(mol)

	# encode all the mobile_phase into integers
	mobile_phase = list(set(mobile_phase))
	mobile_phase_dict = {}
	for idx, mp in enumerate(mobile_phase):
		mobile_phase_dict[mp] = idx
	print('Generate mobile phase dictionary: \n', mobile_phase_dict)
	
	encoded_out_mols = []
	for mol in out_mols: 
		mol.SetProp('encode_mobile_phase', str(mobile_phase_dict[mol.GetProp('tmp')]))
		mol.ClearProp('tmp')
		encoded_out_mols.append(mol)

	print('Writing {} data to {}'.format(len(out_mols), args.output))
	w = Chem.SDWriter(args.output)
	for m in out_mols:
		w.write(m)
	print('Done!')