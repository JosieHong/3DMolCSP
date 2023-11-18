'''
Date: 2022-08-06 13:47:37
LastEditors: yuhhong
LastEditTime: 2022-11-21 16:05:50
'''
import os
import argparse
from tqdm import tqdm

import pandas as pd

from rdkit import Chem
# suppress rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdDepictor

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Molecular Conformers Generator')
    parser.add_argument('--path', type=str, default="", 
                        help='path to input data')
    parser.add_argument('--conf_type', type=str, default='etkdg', 
                        choices=['2d', 'etkdg', 'etkdgv3', 'omega'], 
                        help='conformation type')
    parser.add_argument('--license', type=str, default="./license/oe_license.txt", 
                        help='path to openeye license')
    args = parser.parse_args()



    # 0. load openeye license: Generating OMEGA conformers needs a license. 
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



    # 1. extract SMILES list from datasets
    smiles_list = []
    prop_dict = {}
    id_list = []
    adduct_list = []
    csp_category = []
    
    supp = Chem.SDMolSupplier(args.path)
    prop_dict['k2/k1'] = []
    for idx, mol in enumerate(supp):
        smiles_list.append(Chem.MolToSmiles(mol))
        prop_dict['k2/k1'].append(mol.GetProp('k2/k1'))
        id_list.append('chira_'+str(idx).zfill(6))
        adduct_list.append(mol.GetProp('encode_mobile_phase')) 
        csp_category.append(mol.GetProp('mobile_phase_category'))

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

        # append id, conformer, property, adduct
        # rdkit
        if args.conf_type == 'etkdg' or args.conf_type == 'etkdgv3' or args.conf_type == '2d': 
            mol_from_smiles.SetProp('id', str(id_list[idx]))
            mol_from_smiles.SetProp('smiles', str(smiles_list[idx]))
            mol_from_smiles.SetProp('adduct', str(adduct_list[idx]))
            mol_from_smiles.SetProp('csp_category', str(csp_category[idx]))
            for task in prop_dict.keys():
                mol_from_smiles.SetProp(task, str(prop_dict[task][idx]))
            out_supp.append(mol_from_smiles)
        # oechem
        else: 
            oechem.OESetSDData(mol_from_smiles, 'id', str(id_list[idx]))
            oechem.OESetSDData(mol_from_smiles, 'smiles', str(smiles_list[idx]))
            oechem.OESetSDData(mol_from_smiles, 'adduct', str(adduct_list[idx]))
            oechem.OESetSDData(mol_from_smiles, 'csp_category', str(csp_category[idx]))
            for task in prop_dict.keys(): 
                oechem.OESetSDData(mol_from_smiles, task, str(prop_dict[task][idx]))
            out_supp.append(mol_from_smiles)



    # 3. save the resutls
    output_path = '.'.join(args.path.split('.')[:-1])+'_'+args.conf_type+'.sdf'
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