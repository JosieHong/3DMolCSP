from tqdm import tqdm
import pandas as pd
import argparse

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdDepictor

'''
preprocess: 
  1. remove the invalid 3d conformer
  2. remove the molecules with unlabeled atoms 
	  (we only label the atoms: ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I'])
  3. remove the molecules with more than 300 atom 
	  (it can be changed, but how about using 300 temporarily)
  4. generate conformations
'''

ATOM_LIST = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I']

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--input', type=str, default = '', required=True, 
						help='path to input data')
	parser.add_argument('--output', type=str, default = '', required=True,  
						help='path to output data')
	parser.add_argument('--conf_type', type=str, default = 'etkdg', required=True, 
						choices=['2d', 'etkdg', 'etkdgv3', 'omega'], 
						help='Dataset name')
	parser.add_argument('--license', type=str, default="./license/oe_license.txt", 
						help='Path to openeye license')
	args = parser.parse_args()

	df = pd.read_csv(args.input, header=None, names=['SMILES'])
	smiles_list = df['SMILES'].tolist()



	# 0. filter out the molecules 
	filtered_smiles_list = []
	for s in smiles_list:
		mol = Chem.MolFromSmiles(s) 
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

		filtered_smiles_list.append(s)



	# 1. load openeye license: Generating OMEGA conformers needs a license. 
	# If you don't have the license, you could choose not to use that conformers type. 
	if args.conf_type == 'omega': 
		import openeye
		from openeye import oechem
		from openeye import oeomega

		# load openeye license
		if os.path.isfile(args.license): 
			license_file = open(args.license, 'r')
			openeye.OEAddLicenseData(license_file.read())
			license_file.close()
		else:
			print("Error: Your OpenEye license is not readable; please check your filename and that you have mounted your Google Drive")
		licensed = oechem.OEChemIsLicensed()
		print("Was your OpenEye license correctly installed (True/False)? " + str(licensed))
		if not licensed: 
			print("Error: Your OpenEye license is not correctly installed.")
			raise Exception("Error: Your OpenEye license is not correctly installed.")

	

	# 2. generate conformers of 2D, ETKDG, ETKDGv3 and OMEGA
	out_supp = []
	for idx, smiles in enumerate(tqdm(smiles_list)): 
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

		# append smiles
		# rdkit
		if args.conf_type == 'etkdg' or args.conf_type == 'etkdgv3' or args.conf_type == '2d': 
			mol_from_smiles.SetProp('smiles', str(smiles_list[idx]))
			out_supp.append(mol_from_smiles)
		# oechem
		else: 
			oechem.OESetSDData(mol_from_smiles, 'smiles', str(smiles_list[idx]))
			out_supp.append(mol_from_smiles)



	# 3. save the resutls
	output_path = '.'.join(args.output.split('.')[:-1])+'_'+args.conf_type+'.sdf'
	if args.conf_type == 'omega': 
		ofs = oechem.oemolostream()
		ofs.SetFormat(oechem.OEFormat_SDF)
		ofs.open(output_path)
		for mol in out_supp: 
			oechem.OEWriteConstMolecule(ofs, mol)
	
	else:
		ofs = Chem.SDWriter(output_path)
		for mol in out_supp: 
			ofs.write(mol)
	print("Save the generate conformers in {}".format(output_path))