'''
Date: 2022-11-21 15:53:39
LastEditors: yuhhong
LastEditTime: 2022-11-22 23:35:52
'''
import os
import argparse
import pandas as pd
import pprint
from tqdm import tqdm

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem

from utils import ATOM_LIST, convert2cls

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
	parser.add_argument('--conf_type', type=str, default='etkdg', 
                        choices=['2d', 'etkdg', 'etkdgv3', 'omega'], 
                        help='conformation type')
	parser.add_argument('--csp_setting', type=str, required=True, 
						help='path to csp settings')
	parser.add_argument('--output', type=str, required=True, 
						help='path to output data')
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

		smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
		smiles_iso = Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))

		df_dict['SMILES'].append(smiles)
		df_dict['SMILES_iso'].append(smiles_iso)
		df_dict['CSP_NO'].append(str(mol.GetProp('csp_no')))
		df_dict['Elution_Order'].append(int(mol.GetProp('class')))

	df = pd.DataFrame.from_dict(df_dict)
	df = df.drop_duplicates()
	
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
	df_org = df_org.drop_duplicates()

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
	# merge k2/k1, csp_category, and elution order together
	df = df.merge(df_org, left_on=['SMILES', 'CSP_NO'], right_on=['SMILES', 'CSP_NO'])
	df['CSP_Category'] = df['CSP_NO'].apply(lambda x: CSP_DICT[x][1])

	# convert dataframe into mol_list and generate conformations
	out_mols = []
	print('Generating conformations...')
	for idx, row in tqdm(df.iterrows(), total=df.shape[0]): 
		smiles = row['SMILES_iso']

		if args.conf_type == 'etkdg': 
			mol = Chem.MolFromSmiles(smiles)
			if mol == None: continue
			mol_from_smiles = Chem.AddHs(mol)
			AllChem.EmbedMolecule(mol_from_smiles)

		elif args.conf_type == 'etkdgv3': 
			mol = Chem.MolFromSmiles(smiles)
			if mol == None: continue
			mol_from_smiles = Chem.AddHs(mol)
			AllChem.EmbedMolecule(mol_from_smiles, AllChem.ETKDGv3()) 

		elif args.conf_type == '2d':
			mol = Chem.MolFromSmiles(smiles)
			if mol == None: continue
			mol_from_smiles = Chem.AddHs(mol)
			rdDepictor.Compute2DCoords(mol_from_smiles)

		elif args.conf_type == 'omega':
			# print("Is GPU ready? (True/False)", oeomega.OEOmegaIsGPUReady())
			mol_from_smiles = oechem.OEMol()
			oechem.OEParseSmiles(mol_from_smiles, smiles)
			oechem.OESuppressHydrogens(mol_from_smiles)
			
			# First we set up Omega
			omega = oeomega.OEOmega() 
			omega.SetMaxConfs(1) # Only generate one conformer for our molecule
			omega.SetStrictStereo(False) # Set to False to pick random stereoisomer if stereochemistry is not specified (not relevant here)
			omega.SetStrictAtomTypes(False) # Be a little loose about atom typing to ensure parameters are available to omega for all molecules
			omega(mol_from_smiles)
			
		else: 
			raise ValueError("Undifined conformer type: {}".format(args.conf_type))

		# append id, conformer, property, adduct
		# rdkit
		if args.conf_type == 'etkdg' or args.conf_type == 'etkdgv3' or args.conf_type == '2d': 
			mol_from_smiles.SetProp('id', str(idx))
			mol_from_smiles.SetProp('smiles', str(row['SMILES_iso']))
			mol_from_smiles.SetProp('k2/k1', str(row['K2/K1']))
			mol_from_smiles.SetProp('mobile_phase_category', str(row['CSP_Category']))
			mol_from_smiles.SetProp('elution_order', str(row['Elution_Order']))
			out_mols.append(mol_from_smiles)
		# oechem
		else: 
			oechem.OESetSDData(mol_from_smiles, 'id', str(idx))
			oechem.OESetSDData(mol_from_smiles, 'smiles', str(row['SMILES_iso']))
			oechem.OESetSDData(mol_from_smiles, 'k2/k1', str(row['K2/K1']))
			oechem.OESetSDData(mol_from_smiles, 'mobile_phase_category', str(row['CSP_Category']))
			oechem.OESetSDData(mol_from_smiles, 'elution_order', str(row['Elution_Order']))
			out_mols.append(mol_from_smiles)

	print('Writing {} data to {}'.format(len(out_mols), args.output))
	w = Chem.SDWriter(args.output)
	for m in out_mols:
		w.write(m)
	print('Done!')

	