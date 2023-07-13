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

### 1. Preprocess

```bash
python ./preprocess/preprocess_chirality.py \
--input ./data/Chirality/chirbase.sdf \
--output ./data/Chirality/chirbase_clean.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv

python ./preprocess/preprocess_chirality.py \
--input ./data/Chirality/chirbase.sdf \
--output ./data/Chirality/chirbase_clean2.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv

python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean.sdf --conf_type etkdg 
python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean2.sdf --conf_type etkdg

# (option) OMEGA conformations are available
python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean.sdf --conf_type omega

# (option) randomly split training and validation set for section 3
python ./preprocess/random_split_sdf.py --input ./data/Chirality/chirbase_clean2_etkdg.sdf --output_train ./data/Chirality/chirbase_clean2_etkdg_train.sdf --output_test ./data/Chirality/chirbase_clean2_etkdg_test.sdf
python ./preprocess/random_split_sdf.py --input ./data/Chirality/chirbase_clean_omega.sdf --output_train ./data/Chirality/chirbase_clean_omega_train.sdf --output_test ./data/Chirality/chirbase_clean_omega_test.sdf
```

### 2. Five-fold cross-validation

```bash
# training from scratch
nohup bash ./experiments/molnet_chir_etkdg_5fold.sh > molnet_chir_etkdg_5fold.out 

# traning from pre-trained model
nohup bash ./experiments/molnet_chir_etkdg_5fold_tl.sh > molnet_chir_etkdg_5fold_tl.out 

# input enantiomers
nohup bash ./experiments/molnet_chir2_etkdg_5fold.sh > molnet_chir2_etkdg_5fold.out 
```

## 3. Training (using all data) & test on all CSPs

```bash
# traning from pre-trained model
nohup bash ./experiments/molnet_chir_etkdg_tl.sh > molnet_chir_etkdg_tl.out 
nohup bash ./experiments/molnet_chir_etkdg_tl_p1.sh > molnet_chir_etkdg_tl_p1.out 
nohup bash ./experiments/molnet_chir_etkdg_tl_p2.sh > molnet_chir_etkdg_tl_p2.out 

# infer on all CSPs
nohup bash ./experiments/molnet_chir_etkdg_tl_infer.sh > molnet_chir_etkdg_tl_infer.out
```

## 4. Test on CMRT

```bash
# preprocessing
python ./preprocess/preprocess_cmrt.py \
--input ./data/CMRT/cmrt_all_column.csv \
--output ./data/CMRT/cmrt_clean.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv

python ./preprocess/gen_conformers.py --path ./data/CMRT/cmrt_clean.sdf --conf_type etkdg

# infer on CMRT
nohup bash ./experiments/molnet_cmrt_etkdg_tl_infer.sh > molnet_cmrt_etkdg_tl_infer.out 
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
