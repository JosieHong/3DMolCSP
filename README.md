# 3DMolCSP

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use) 

Enhanced Structure-Based Prediction of Chiral Stationary Phases for Chromatographic Enantioseparation from 3D Molecular Conformations

*paper on Analytical Chemistry is coming...*

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

## Data preprocessing

### Demo set (Chirobiotic V)

```bash
# The demo dataset is already cleaned. 
# generate 3D conformations
python ./preprocess/gen_conformers.py --path ./data/demo/chirobiotic_v.sdf --conf_type etkdg

# (optional)
python ./preprocess/random_split_sdf.py --input ./data/demo/chirobiotic_v_etkdg.sdf --output_train ./data/demo/chirobiotic_v_etkdg_train.sdf --output_test ./data/demo/chirobiotic_v_etkdg_test.sdf
```

### ChirBase

```bash
# preprocessing
python ./preprocess/preprocess_chirbase.py \
--input ./data/ChirBase/chirbase.sdf \
--output ./data/ChirBase/chirbase_clean.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv

# generate 3D conformations
python ./preprocess/gen_conformers.py --path ./data/ChirBase/chirbase_clean.sdf --conf_type etkdg

# (optional) OMEGA conformations are available
# OMEGA algorithm requires a license. Please get the license here: https://www.eyesopen.com/omega
python ./preprocess/gen_conformers.py --path ./data/ChirBase/chirbase_clean.sdf --conf_type omega --license <path to OMEGA license>

# (optional) randomly split training and validation set for Exp3
python ./preprocess/random_split_sdf.py --input ./data/ChirBase/chirbase_clean_etkdg.sdf --output_train ./data/ChirBase/chirbase_clean_etkdg_train.sdf --output_test ./data/ChirBase/chirbase_clean_etkdg_test.sdf
python ./preprocess/random_split_sdf.py --input ./data/ChirBase/chirbase_clean_omega.sdf --output_train ./data/ChirBase/chirbase_clean_omega_train.sdf --output_test ./data/ChirBase/chirbase_clean_omega_test.sdf
```

### CMRT

```bash
# preprocessing
python ./preprocess/preprocess_cmrt.py \
--input ./data/CMRT/cmrt_all_column.csv \
--output ./data/CMRT/cmrt_clean.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv

# generate 3D conformations
python ./preprocess/gen_conformers.py --path ./data/CMRT/cmrt_clean.sdf --conf_type etkdg
```

## Experiments

### Exp1: Demo

1. Preprocess demo dataset

2. Five-fold cross-validation

```bash
# training from scratch
python main_chir_kfold.py --config ./configs/molnet_train_demo.yaml --k_fold 5 --csp_no 3 \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --checkpoint ./check_point/demo_sc.pt \
                                    --result_path ./demo/demo_sc.csv \
                                    --device 1

# training from pretrained model 
python main_chir_kfold.py --config ./configs/molnet_train_demo.yaml --k_fold 5 --csp_no 3 \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --resume_path ./check_point/molnetv2_qtof_etkdgv3.pt \
                                    --transfer \
                                    --checkpoint ./check_point/demo_tl.pt \
                                    --result_path ./demo/demo_tl.csv \
                                    --device 1
```

Results on demo dataset:

```bash
# --------------- Final Results of 3DMolCSP-SC --------------- #
fold_0: acc: 0.81, auc: 0.95
fold_1: acc: 0.90, auc: 0.99
fold_2: acc: 0.73, auc: 0.87
fold_3: acc: 0.81, auc: 0.96
fold_4: acc: 0.84, auc: 0.98
mean acc: 0.82, mean auc: 0.95

# --------------- Final Results of 3DMolCSP-TL --------------- #
fold_0: acc: 0.80, auc: 0.94
fold_1: acc: 0.91, auc: 0.99
fold_2: acc: 0.81, auc: 0.97
fold_3: acc: 0.79, auc: 0.96
fold_4: acc: 0.80, auc: 0.97
mean acc: 0.82, mean auc: 0.97
```

### Exp2: Five-fold cross-validation on ChirBase

1. Preprocess ChirBase

2. Five-fold cross-validation

```bash
# training from scratch
nohup bash ./experiments/train_chir_etkdg_5fold.sh > molnet_chir_etkdg_5fold.out 

# traning from pre-trained model
nohup bash ./experiments/train_chir_etkdg_5fold_tl.sh > molnet_chir_etkdg_5fold_tl.out 
```

### Exp3: Training on ChirBase and testing on CMRT

1. Training (using all data)

```bash
# traning from pre-trained model
nohup bash ./experiments/train_chir_etkdg_tl.sh > molnet_chir_etkdg_tl.out 
```

2. Preprocess CMRT

3. infer on CMRT

```bash
nohup bash ./experiments/infer_cmrt_etkdg_tl.sh > molnet_cmrt_etkdg_tl_infer.out 
```

### Exp4: Elution order prediction

```bash
nohup bash ./experiments/train_chir_etkdg_eo.sh > train_chir_etkdg_eo.out 
```

<!-- ## Citation

If you feel this work useful, please cite: 

```
``` -->

---

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
