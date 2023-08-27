# 3DMolChir

## Set up

```bash
# RDKit 2022.03.5
# https://www.rdkit.org/docs/GettingStartedInPython.html
conda create -c rdkit -n <env-name> rdkit
conda activate <env-name>
# please use rdkit >= 2021.03.4
# https://github.com/rdkit/rdkit/pull/4272

# Pytorch 1.13.0
# Please choose the proper cuda version from their official website:
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge tensorboard

pip install lxml tqdm pandas pyteomics PyYAML scikit-learn
```

## Experiments

### Five-fold cross-validation on ChirBase

1. Preprocess ChirBase

```bash
# preprocessing
python ./preprocess/preprocess_chirbase.py \
--input ./data/ChirBase/chirbase.sdf \
--output ./data/ChirBase/chirbase_clean.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv

python ./preprocess/preprocess_chirbase.py \
--input ./data/ChirBase/chirbase.sdf \
--output ./data/ChirBase/chirbase_clean2.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv

python ./preprocess/preprocess_chirbase.py \
--input ./data/ChirBase/chirbase.sdf \
--output ./data/ChirBase/chirbase_clean3.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv

python ./preprocess/preprocess_chirbase.py \
--input ./data/ChirBase/chirbase.sdf \
--output ./data/ChirBase/chirbase_clean4.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv

# preprocess results: 
# ~~1. no duplicated (76795)~~
# ~~2. duplicated the isomer SMILES (43967)~~
# ~~3. duplicated the non-isomer SMILES (43700)~~
# 4. duplicated the isomer SMILES with the same chiral atom (43785)

# generate enantiomers
python ./preprocess/convert_enantiomers.py --input ./data/ChirBase/chirbase_clean4.sdf --output ./data/ChirBase/chirbase_clean4_enatiomers.sdf

# generate 3D conformations
python ./preprocess/gen_conformers.py --path ./data/ChirBase/chirbase_clean4.sdf --conf_type etkdg
python ./preprocess/gen_conformers.py --path ./data/ChirBase/chirbase_clean4_enatiomers.sdf --conf_type etkdg

# (option) OMEGA conformations are available
python ./preprocess/gen_conformers.py --path ./data/ChirBase/chirbase_clean.sdf --conf_type omega

# (option) randomly split training and validation set for section 3
python ./preprocess/random_split_sdf.py --input ./data/ChirBase/chirbase_clean4_etkdg.sdf --output_train ./data/ChirBase/chirbase_clean4_etkdg_train.sdf --output_test ./data/ChirBase/chirbase_clean4_etkdg_test.sdf
python ./preprocess/random_split_sdf.py --input ./data/ChirBase/chirbase_clean_omega.sdf --output_train ./data/ChirBase/chirbase_clean_omega_train.sdf --output_test ./data/ChirBase/chirbase_clean_omega_test.sdf
```

2. Five-fold cross-validation

```bash
# training from scratch
nohup bash ./experiments/train_chir_etkdg_5fold.sh > molnet_chir_etkdg_5fold.out 
nohup bash ./experiments/train_chir_etkdg_5fold_p1.sh > molnet_chir_etkdg_5fold_p1.out 
nohup bash ./experiments/train_chir_etkdg_5fold_p2.sh > molnet_chir_etkdg_5fold_p2.out 

# traning from pre-trained model
nohup bash ./experiments/train_chir_etkdg_5fold_tl.sh > molnet_chir_etkdg_5fold_tl.out 
nohup bash ./experiments/train_chir_etkdg_5fold_tl_p1.sh > molnet_chir_etkdg_5fold_tl_p1.out 
nohup bash ./experiments/train_chir_etkdg_5fold_tl_p2.sh > molnet_chir_etkdg_5fold_tl_p2.out 
```

### Training on ChirBase and testing on CMRT

1. Training (using all data)

```bash
# traning from pre-trained model
nohup bash ./experiments/train_chir_etkdg_tl.sh > molnet_chir_etkdg_tl.out 
nohup bash ./experiments/train_chir_etkdg_tl_p1.sh > molnet_chir_etkdg_tl_p1_0804.out 
nohup bash ./experiments/train_chir_etkdg_tl_p2.sh > molnet_chir_etkdg_tl_p2_0804.out 
```

2. Preprocess CMRT

```bash
# preprocessing
python ./preprocess/preprocess_cmrt.py \
--input ./data/CMRT/cmrt_all_column.csv \
--output ./data/CMRT/cmrt_clean.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv

# generate enantiomers
python ./preprocess/convert_enantiomers.py --input ./data/CMRT/cmrt_clean.sdf --output ./data/CMRT/cmrt_clean_enatiomers.sdf

# generate 3D conformations
python ./preprocess/gen_conformers.py --path ./data/CMRT/cmrt_clean.sdf --conf_type etkdg
python ./preprocess/gen_conformers.py --path ./data/CMRT/cmrt_clean_enatiomers.sdf --conf_type etkdg
```

3. infer on CMRT

```bash
# inference on one enantiomer
nohup bash ./experiments/infer_cmrt_etkdg_tl.sh > molnet_cmrt_etkdg_tl_infer.out 

# inference on the other enantiomer
nohup bash ./experiments/infer_cmrt_etkdg_tl.sh > molnet_cmrt_etkdg_tl_infer.out
```

### ~~5. Training on ChirBase excluding CMRT~~

```bash
# preprocessing
# ChirBase - CMRT
python ./preprocess/minus_sdf.py --minuend ./data/ChirBase/chirbase_clean4.sdf --subtrahend ./data/CMRT/cmrt_clean.sdf --output ./data/ChirBase/chirbase_minus_cmrt_clean.sdf
nohup python ./preprocess/gen_conformers.py --path ./data/ChirBase/chirbase_minus_cmrt_clean.sdf --conf_type etkdg
```

### ~~3. Train & Eval (multi-task learning)~~

```bash
nohup python main_chir_kfold.py --config ./configs/molnet_chirality_cls_etkdg_multi.yaml --k_fold 5 \
                            --log_dir ./logs/molnet_chirality/ \
                            --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp_multi.pt \
                            --result_path ./results/molnet_chirality_cls_etkdg_csp_multi.csv \
                            --device 2 > molnet_chir_etkdg_5fold_multi.out 

nohup python main_chir_kfold.py --config ./configs/molnet_chirality_cls_etkdg_multi_L.yaml --k_fold 5 \
                            --log_dir ./logs/molnet_chirality/ \
                            --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp_multi_L.pt \
                            --result_path ./results/molnet_chirality_cls_etkdg_csp_multi_L.csv \
                            --device 0 > molnet_chir_etkdg_5fold_multi_L.out
```

## Jupyter Notebook

```bash
conda activate molnet
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=molnet
conda install -c conda-forge notebook

jupyter notebook --no-browser --port=8889
# on my own PC
ssh -N -f -L localhost:8888:localhost:8889 yuhhong@boltzmann.luddy.indiana.edu
ssh -N -f -L localhost:8888:localhost:8889 yuhhong@ampere.luddy.indiana.edu
# visit: 
# http://localhost:8888/
```

