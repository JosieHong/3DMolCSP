import pandas as pd
import pprint
from tqdm import tqdm
tqdm.pandas()
import pickle
import random 
import numpy as np

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdDepictor

ATOM_LIST = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I']

ENCODE_ATOM = {'C': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
				'H': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
				'O': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
				'N': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
				'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
				'S': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
				'Cl': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
				'P': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
				'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
				'Br': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
				'I': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

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

def molnet_filter(smiles):
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return False
	
	mol_block = Chem.MolToMolBlock(mol).split("\n")
	mol_block_length = sum([1 for d in mol_block if len(d)==69 and len(d.split())==16])
	if mol_block_length < mol.GetNumAtoms(): 
		print(mol_block_length, '<', mol.GetNumAtoms())
		return False
	if mol.GetNumAtoms() >= 200: # num_atoms: 200
		print('Too many atoms')
		return False

	flag_remove = False
	for atom in mol.GetAtoms(): 
		if atom.GetSymbol() not in ATOM_LIST:
			flag_remove = True
			print('Unlabeled atom: {}'.format(atom.GetSymbol()))
			break

	if flag_remove: 
		return False
	return True

def gen_conf_from_df(df, conf_type): 
	out_mols = []
	print('Generating conformations...') 
	for idx, row in tqdm(df.iterrows(), total=df.shape[0]): 
		smiles = row['ID']

		if conf_type == 'etkdg': 
			mol = Chem.MolFromSmiles(smiles)
			if mol == None: continue
			mol_from_smiles = Chem.AddHs(mol)
			AllChem.EmbedMolecule(mol_from_smiles)

		elif conf_type == 'etkdgv3': 
			mol = Chem.MolFromSmiles(smiles)
			if mol == None: continue
			mol_from_smiles = Chem.AddHs(mol)
			AllChem.EmbedMolecule(mol_from_smiles, AllChem.ETKDGv3()) 

		elif conf_type == '2d':
			mol = Chem.MolFromSmiles(smiles)
			if mol == None: continue
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
			omega.SetStrictStereo(False) # Set to False to pick random stereoisomer if stereochemistry is not specified (not relevant here)
			omega.SetStrictAtomTypes(False) # Be a little loose about atom typing to ensure parameters are available to omega for all molecules
			omega(mol_from_smiles)
			
		else: 
			raise ValueError("Undifined conformer type: {}".format(conf_type))

		# append id, conformer, property, adduct
		# rdkit
		if conf_type == 'etkdg' or conf_type == 'etkdgv3' or conf_type == '2d': 
			mol_from_smiles.SetProp('id', str(row['ID']))
			mol_from_smiles.SetProp('smiles', str(row['SMILES_nostereo']))
			mol_from_smiles.SetProp('chiral_tag', str(row['RS_label_binary']))
			out_mols.append(mol_from_smiles)
		# oechem
		else: 
			oechem.OESetSDData(mol_from_smiles, 'id', str(row['ID']))
			oechem.OESetSDData(mol_from_smiles, 'smiles', str(row['SMILES_nostereo']))
			oechem.OESetSDData(mol_from_smiles, 'chiral_tag', str(row['RS_label_binary']))
			out_mols.append(mol_from_smiles)
	return out_mols

def create_X(mol, num_points): 
	try: # more accurat method
		conformer = mol.GetConformer()
		point_set = conformer.GetPositions().tolist() # 0. x,y,z-coordinates; 

	except: # parse the MolBlock by ourself
		mol_block = Chem.MolToMolBlock(mol).split("\n")
		point_set = parse_mol_block(mol_block) # 0. x,y,z-coordinates; 
		point_set = point_set.tolist()

	for idx, atom in enumerate(mol.GetAtoms()): 
		point_set[idx] = point_set[idx] + ENCODE_ATOM[atom.GetSymbol()] # atom type (one-hot);
		point_set[idx].append(atom.GetDegree()) # 1. number of immediate neighbors who are “heavy” (nonhydrogen) atoms;
		point_set[idx].append(atom.GetExplicitValence()) # 2. valence minus the number of hydrogens;
		point_set[idx].append(atom.GetMass()) # 3. atomic mass; 
		point_set[idx].append(atom.GetFormalCharge()) # 4. atomic charge;
		point_set[idx].append(atom.GetNumImplicitHs()) # 5. number of implicit hydrogens;
		point_set[idx].append(int(atom.GetIsAromatic())) # Is aromatic
		point_set[idx].append(int(atom.IsInRing())) # Is in a ring

		point_set[idx].append(100 if int(atom.GetChiralTag()) > 0 else 0) 
		# point_set[idx].append(int(atom.GetChiralTag())*100) 

	point_set = np.array(point_set).astype(np.float32)
	# center the points
	points_xyz = point_set[:, :3]
	centroid = np.mean(points_xyz, axis=0)
	points_xyz -= centroid
	point_set = np.concatenate((points_xyz, point_set[:, 3:]), axis=1)

	# pad zeros
	point_set = np.pad(point_set, ((0, num_points-point_set.shape[0]), (0, 0)), constant_values=0)
	return point_set
	
def parse_mol_block(mol_block): 
	'''
	Input:  mol_block   [list denotes the lines of mol block]
	Return: points      [list denotes the atom points, (npoints, 4)]
	'''
	points = []
	for d in mol_block: 
		# print(len(d), d)
		if len(d) == 69: # the format of molecular block is fixed
			atom = [i for i in d.split(" ") if i!= ""]
			# atom: [x, y, z, atom_type, charge, stereo_care_box, valence]
			# sdf format (atom block): 
			# https://docs.chemaxon.com/display/docs/mdl-molfiles-rgfiles-sdfiles-rxnfiles-rdfiles-formats.md
			
			if len(atom) == 16 and atom[3] in ENCODE_ATOM.keys(): 
				# only x-y-z coordinates
				point = [float(atom[0]), float(atom[1]), float(atom[2])]
			elif len(atom) == 16: # check the atom type
				print("Error: {} is not in {}, please check the dataset.".format(atom[3], ENCODE_ATOM.keys()))
				exit()
			else: 
				continue
			points.append(point)
	
	points = np.array(points)
	# center the points
	points_xyz = points[:, :3]
	centroid = np.mean(points_xyz, axis=0)
	points_xyz -= centroid

	points = np.concatenate((points_xyz, points[:, 3:]), axis=1)
	return points