'''
Date: 2021-11-30 13:55:07
LastEditors: yuhhong
LastEditTime: 2022-11-30 13:33:16
'''
import torch
from torch.utils.data import Dataset

import numpy as np
import math

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.rdchem import HybridizationType



class BaseDataset(Dataset):
	def __init__(self): 
		self.ENCODE_ATOM = {'C': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
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
		
	def __len__(self):
		return 0

	def __getitem__(self, idx):
		pass

	def create_X(self, mol, num_points): 
		try: # more accurat method
			conformer = mol.GetConformer()
			point_set = conformer.GetPositions().tolist() # 0. x,y,z-coordinates; 

		except: # parse the MolBlock by ourself
			mol_block = Chem.MolToMolBlock(mol).split("\n")
			point_set = self.parse_mol_block(mol_block) # 0. x,y,z-coordinates; 

		for idx, atom in enumerate(mol.GetAtoms()): 
			point_set[idx] = point_set[idx] + self.ENCODE_ATOM[atom.GetSymbol()] # atom type (one-hot);
			point_set[idx].append(atom.GetDegree()) # 1. number of immediate neighbors who are “heavy” (nonhydrogen) atoms;
			point_set[idx].append(atom.GetExplicitValence()) # 2. valence minus the number of hydrogens;
			point_set[idx].append(atom.GetMass()/100) # 3. atomic mass; 
			point_set[idx].append(atom.GetFormalCharge()) # 4. atomic charge;
			point_set[idx].append(atom.GetNumImplicitHs()) # 5. number of implicit hydrogens;
			point_set[idx].append(int(atom.GetIsAromatic())) # Is aromatic
			point_set[idx].append(int(atom.IsInRing())) # Is in a ring

			# hybridization = atom.GetHybridization()
			# point_set[idx].append(1 if hybridization == HybridizationType.SP else 0)
			# point_set[idx].append(1 if hybridization == HybridizationType.SP2 else 0)
			# point_set[idx].append(1 if hybridization == HybridizationType.SP3 else 0)
			
		point_set = np.array(point_set).astype(np.float32)

		# generate mask
		point_mask = np.ones_like(point_set[0])
		point_mask = torch.cat((torch.Tensor(point_mask), torch.zeros((num_points-point_mask.shape[0]))), dim=0)
		
		# pad zeros
		point_set = torch.cat((torch.Tensor(point_set), torch.zeros((num_points-point_set.shape[0], point_set.shape[1]))), dim=0)
		return point_set, point_mask # torch.Size([num_points, 14]), torch.Size([num_points])
	
	def parse_mol_block(self, mol_block): 
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
				
				if len(atom) == 16 and atom[3] in self.ENCODE_ATOM.keys(): 
					# only x-y-z coordinates
					point = [float(atom[0]), float(atom[1]), float(atom[2])]
				elif len(atom) == 16: # check the atom type
					print("Error: {} is not in {}, please check the dataset.".format(atom[3], self.ENCODE_ATOM.keys()))
					exit()
				else: 
					continue
				points.append(point)
		
		# center the points
		points_xyz = points[:, :3]
		centroid = np.mean(points_xyz, axis=0)
		points_xyz -= centroid

		points = np.concatenate((points_xyz, points[:, 3:]), axis=1)
		return points

	def ensemble_mol(self, mol, conformer):
		mol = Chem.AddHs(mol) 
		if conformer == '2d':
			rdDepictor.Compute2DCoords(mol)
		elif conformer == 'etkdg': 
			try: # 3D comformers
				# Original ETKDG:
				# https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00654
				AllChem.EmbedMolecule(mol, AllChem.ETKDG(), randomSeed=42) 
			except: # using 2D comformers
				rdDepictor.Compute2DCoords(mol)
		elif conformer == 'etkdgv3':
			try: # 3D comformers
				# An update describing ETKDGv3 and extensions to better 
				# handle small rings and macrocycles: 
				# https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00025
				AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
			except: # using 2D comformers
				rdDepictor.Compute2DCoords(mol)
		return mol

# Chirality
class ChiralityDataset(BaseDataset): 
	def __init__(self, supp, num_points=200, multi_csp=False, csp_no=0, data_augmentation=False): 
		super(ChiralityDataset, self).__init__()
		self.num_points = num_points
		self.multi_csp = multi_csp
		self.data_augmentation = data_augmentation

		if multi_csp: # multiple csp_no 
			assert csp_no == 0, "Charility phase number can not be chosen, if multi_scp==True. "
			self.supp = supp

		else: # single csp_no 
			assert csp_no >= 0 and csp_no < 20
			self.supp = [] # without balance
			for mol in supp: 
				mb = int(mol.GetProp('adduct'))
				if mb != csp_no: 
					continue
				self.supp.append(mol)
	
	def balance_indices(self, indices): 
		print('Balance the dataset...')
		train_supp = [mol for i, mol in enumerate(self.supp) if i in indices]

		# seperate by csp
		csp_dict = {}
		for i, mol in enumerate(train_supp): 
			mb = int(mol.GetProp('adduct'))
			chir = float(mol.GetProp('k2/k1'))
			y = self.convert2cls(chir)

			if mb in csp_dict.keys(): 
				if y in csp_dict[mb].keys():
					csp_dict[mb][y].append(i)
				else:
					csp_dict[mb][y] = [i]
			else:
				csp_dict[mb] = {y: [i]}
		
		# stat = {}
		# for i, mol in enumerate(train_supp): 
		# 	chir = float(mol.GetProp('k2/k1'))
		# 	y = self.convert2cls(chir)
		# 	if y in stat.keys():
		# 		stat[y].append(i)
		# 	else:
		# 		stat[y] = [i]
		# print('Before balance: {}'.format({k: len(v) for k, v in stat.items()}))

		output_indices = []
		for csp, stat in csp_dict.items(): 
			print('Before balance ({}): {}'.format(csp, {k: len(v) for k, v in stat.items()}))

			lengths = [len(v) for v in stat.values()]
			gcd = self.least_common_multiple(lengths)
			if gcd // max(lengths) > 3: 
				gcd = max(lengths) * 3
			coef = {k: gcd//len(v) for k, v in stat.items()}
			print(coef)
			balance_indices = []
			balance_stat = {}
			for i, mol in enumerate(train_supp): 
				chir = float(mol.GetProp('k2/k1'))
				y = self.convert2cls(chir)
				balance_indices += [i]*coef[y]

				if y in balance_stat.keys():
					balance_stat[y] += coef[y]
				else:
					balance_stat[y] = coef[y]
			print('After balance ({}): {}'.format(csp, balance_stat))
			output_indices += balance_indices
		return output_indices

	def least_common_multiple(self, num):
		minimum = 1
		for i in num:
			minimum = int(i)*int(minimum) / math.gcd(int(i), int(minimum))
		return int(minimum)

	def __len__(self):
		return len(self.supp)

	def __getitem__(self, idx):
		mol = self.supp[idx]
		smiles = Chem.MolToSmiles(mol)
		X, mask = self.create_X(mol, self.num_points)
		chir = float(mol.GetProp('k2/k1'))
		Y = self.convert2cls(chir)
		if self.multi_csp:
			mb = int(mol.GetProp('adduct'))
			multi_Y = [np.nan] * 20
			multi_Y[mb] = Y
			return smiles, X, mask, torch.Tensor(multi_Y).to(torch.int64)
		else: 
			return smiles, X, mask, Y

	def convert2cls(self, chir): 
		# if chir < 1: # no data fallen in this class
		#     y = 0
		# elif chir < 1.15:
		#     y = 1
		# elif chir < 1.2:
		#     y = 2
		# else:
		#     y = 3
		if chir < 1.15:
			y = 0
		elif chir < 1.2:
			y = 1
		else:
			y = 2
		return y