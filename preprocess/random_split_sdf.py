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
    parser.add_argument('--output_train', type=str, default = '',
                        help='path to output data')
    parser.add_argument('--output_test', type=str, default = '',
                        help='path to output data')
    args = parser.parse_args()

    DATA_PATH = args.input
    TRAIN_PATH = args.output_train
    TEST_PATH = args.output_test

    suppl = Chem.SDMolSupplier(DATA_PATH)
    mols = [x for x in suppl if x is not None]


    # output the smiles list
    SMILES_OUT_PATH = '/'.join(args.input.split('/')[:-1]) + '/SMILES_list.txt'
    smiles = list(set([Chem.MolToSmiles(m) for m in mols]))
    smiles_out = "\n".join(smiles)
    with open(SMILES_OUT_PATH, 'w') as f:
        f.write(smiles_out)
    print("Load {}/{} data from {}".format(len(mols), len(smiles), DATA_PATH))


    # split the data by smiles
    Ltest = np.random.choice(smiles, int(len(smiles)*0.1))
    Ltrain = [x for x in smiles if x not in Ltest]
    print("Get {} training data, {} test data".format(len(Ltrain), len(Ltest)))


    # wite the training and test data
    w_train = Chem.SDWriter(TRAIN_PATH)
    w_test = Chem.SDWriter(TEST_PATH)
    for m in mols:
        s = Chem.MolToSmiles(m)
        if s in Ltest:
            w_test.write(m)
        else:
            w_train.write(m)
    print("Save training and test data to {} and {}".format(TRAIN_PATH, TEST_PATH))
