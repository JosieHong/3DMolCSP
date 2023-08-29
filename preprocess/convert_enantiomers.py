import os
import argparse
import pprint
import pandas as pd

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

'''
rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW -> R
rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW -> S
'''

def enantiomer(cc1, cc2): 
	for p1, p2 in zip(cc1, cc2):
		if p1[0] != p2[0] or p1[1] == p2[1]:
			return False
	return True
		
def find_the_other_enantimer(smiles1): 
	mol1 = Chem.MolFromSmiles(smiles1)
	cc1 = Chem.FindMolChiralCenters(mol1)
	# chir_tag1 = ';'.join(['0' if str(atom.GetChiralTag()) == 'CHI_UNSPECIFIED' else '1' for atom in mol1.GetAtoms()])
	
	smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles1), isomericSmiles=False)
	mol = Chem.MolFromSmiles(smiles)
	isomers = tuple(EnumerateStereoisomers(mol))
	# print('{} isomers are found!\n'.format(len(isomers)))

	flag = False
	for mol2 in isomers: 
		smiles2 = Chem.MolToSmiles(mol2, isomericSmiles=True)
		mol2 = Chem.MolFromSmiles(smiles2)
		# chir_tag2 = ';'.join(['0' if str(atom.GetChiralTag()) == 'CHI_UNSPECIFIED' else '1' for atom in mol2.GetAtoms()])
		cc2 = Chem.FindMolChiralCenters(mol2)
		if enantiomer(cc1, cc2): 
		# if chir_tag2 == chir_tag1 and smiles2 != smiles1:
			flag = True
			break
	# assert flag == True, 'Can not find the enantiomers for {}'.format(smiles1)
	if flag: 
		return smiles2, mol2
	else:
		return smiles1, mol1



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--input', type=str, required=True, 
						help='path to input data')
	parser.add_argument('--output', type=str, required=True, 
						help='path to output data')
	args = parser.parse_args()

	supp = Chem.SDMolSupplier(args.input)
	out_mols = []
	for mol in supp:
		smiles1 = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))
		smiles2, mol2 = find_the_other_enantimer(smiles1)

		mol2.SetProp('k2/k1', mol.GetProp('k2/k1'))
		mol2.SetProp('encode_mobile_phase', mol.GetProp('encode_mobile_phase'))
		mol2.SetProp('mobile_phase_category', mol.GetProp('mobile_phase_category'))
		out_mols.append(mol2)

	print('Writing {} data to {}'.format(len(out_mols), args.output))
	w = Chem.SDWriter(args.output)
	for m in out_mols:
		w.write(m)
	print('Done!')
