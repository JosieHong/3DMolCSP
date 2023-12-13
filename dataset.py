'''
Date: 2021-11-30 13:55:07
LastEditors: yuhhong
LastEditTime: 2022-11-30 13:33:16
'''
import torch
from torch.utils.data import Dataset

import numpy as np
import math
import pickle

from rdkit import Chem
from rdkit.Geometry import Point3D
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

		point_set = np.array(point_set).astype(np.float32)

		# center the points
		points_xyz = point_set[:, :3]
		centroid = np.mean(points_xyz, axis=0)
		points_xyz -= centroid
		point_set = np.concatenate((points_xyz, point_set[:, 3:]), axis=1)

		# pad zeros
		point_set = torch.cat((torch.Tensor(point_set), torch.zeros((num_points-point_set.shape[0], point_set.shape[1]))), dim=0)
		return point_set #, point_mask # torch.Size([num_points, 14]), torch.Size([num_points])
	
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
		return points
		


class ChiralityDataset(BaseDataset): 
	def __init__(self, supp, num_points=200, csp_no=0, flipping=False): 
		super(ChiralityDataset, self).__init__()
		self.num_points = num_points
		if flipping:
			self.supp = [] # without balance
			for mol in supp: 
				mb = int(mol.GetProp('adduct'))
				if mb != csp_no: 
					continue

				# flipping the conformation
				conf = mol.GetConformer()
				point_set = conf.GetPositions()
				point_set[:, -1] *= -1
				for i in range(mol.GetNumAtoms()): 
					x, y, z = point_set[i]
					conf.SetAtomPosition(i, Point3D(x,y,z))
				self.supp.append(mol)
		else: 
			self.supp = [] # without balance
			for mol in supp: 
				mb = int(mol.GetProp('adduct'))
				if mb != csp_no: 
					continue
				self.supp.append(mol)
	
	def count_cls(self, out_cls, indices): 
		print('Count the dataset...')
		train_supp = [mol for i, mol in enumerate(self.supp) if i in indices]

		samples_per_cls = [0] * out_cls
		for i, mol in enumerate(train_supp): 
			chir = float(mol.GetProp('k2/k1'))
			y = self.convert2cls(chir, mol.GetProp('csp_category'))
			samples_per_cls[y] += 1
		return samples_per_cls

	def balance_indices(self, indices): 
		print('Balance the dataset...')
		train_supp = [mol for i, mol in enumerate(self.supp) if i in indices]

		# seperate by csp
		csp_dict = {}
		for i, mol in enumerate(train_supp): 
			mb = int(mol.GetProp('adduct'))
			chir = float(mol.GetProp('k2/k1'))
			y = self.convert2cls(chir, mol.GetProp('csp_category'))

			if mb in csp_dict.keys(): 
				if y in csp_dict[mb].keys():
					csp_dict[mb][y].append(i)
				else:
					csp_dict[mb][y] = [i]
			else:
				csp_dict[mb] = {y: [i]}

		output_indices = []
		for csp, stat in csp_dict.items(): 
			print('Before balance ({}): {}'.format(csp, {k: len(v) for k, v in stat.items()}))
			if len(stat) < 3:
				print('Only {} class, drop this csp.'.format(len(stat)))
				continue
				
			lengths = [len(v) for v in stat.values()]
			gcd = self.least_common_multiple(lengths)
			if gcd // max(lengths) > 3: 
				gcd = max(lengths) * 3
			coef = {k: gcd//len(v) for k, v in stat.items()}
			balance_indices = []
			balance_stat = {}
			for i, mol in enumerate(train_supp): 
				mb = int(mol.GetProp('adduct'))
				if mb != csp: 
					continue
				chir = float(mol.GetProp('k2/k1'))
				y = self.convert2cls(chir, mol.GetProp('csp_category'))
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
		# mol_id = mol.GetProp('id')
		mol_id = Chem.MolToSmiles(mol, isomericSmiles=True)
		smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
		X = self.create_X(mol, self.num_points)
		chir = float(mol.GetProp('k2/k1'))
		Y = self.convert2cls(chir, mol.GetProp('csp_category'))
		mb = int(mol.GetProp('adduct'))
		return mol_id, smiles, mb, X, Y

	def convert2cls(self, chir, csp_category): 
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



# inference dataset
class ChiralityDataset_infer(BaseDataset): 
	def __init__(self, supp, num_points=200, csp_no=0, flipping=False): 
		super(ChiralityDataset_infer, self).__init__()
		self.num_points = num_points
		self.csp_no = csp_no
		if flipping:
			self.supp = [] # without balance
			for mol in supp: 
				# flipping the conformation
				conf = mol.GetConformer()
				point_set = conf.GetPositions()
				point_set[:, -1] *= -1
				for i in range(mol.GetNumAtoms()): 
					x, y, z = point_set[i]
					conf.SetAtomPosition(i, Point3D(x,y,z))
				self.supp.append(mol)
		else:
			self.supp = supp

	def __len__(self):
		return len(self.supp)

	def __getitem__(self, idx): 
		mol = self.supp[idx]
		# mol_id = mol.GetProp('id')
		mol_id = Chem.MolToSmiles(mol, isomericSmiles=True)
		smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
		X = self.create_X(mol, self.num_points)
		mb = int(self.csp_no)
		return mol_id, smiles, mb, X



# elution order prediction
class ChiralityDataset_EO(BaseDataset): 
	def __init__(self, root_path, num_points=200): 
		super(ChiralityDataset_EO, self).__init__()
		with open(root_path, 'rb') as file: 
			data = pickle.load(file)

		# filter out the inseparable enantiomers
		self.filtered_data = []
		for d in data:
			sep_cls = self.convert2cls(d['k2/k1'], d['csp_category'])
			if sep_cls > 1: 
				self.filtered_data.append(d)
				
	def __len__(self):
		return len(self.filtered_data)

	def __getitem__(self, idx): 
		return self.filtered_data[idx]['smiles_iso'], self.filtered_data[idx]['smiles'], \
				self.filtered_data[idx]['pos'], self.filtered_data[idx]['neg'], self.filtered_data[idx]['anchor'], \
				int(self.filtered_data[idx]['elution_order'])

	def convert2cls(self, chir, csp_category): 
		if csp_category == 1: 
			# For polysaccharide CSPs:
			if chir < 1.15:
				y = 0
			elif chir < 1.2:
				y = 1
			elif chir < 2.1:
				y = 2
			else:
				y = 3
		elif csp_category == 2: 
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
			raise Exception("The category for CSP should be 1 or 2, rather than {}".format(csp_category))
		return y
