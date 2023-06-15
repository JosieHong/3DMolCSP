# 3DMolChar



## Single charility phase

### 1. Preprocess

```bash
python ./preprocess/preprocess_chirality.py --input ./data/Chirality/chirbase.sdf --output ./data/Chirality/chirbase_clean.sdf

python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean.sdf --conf_type etkdg # slow
# python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean.sdf --conf_type omega # slow

python ./preprocess/random_split_sdf.py --input ./data/Chirality/chirbase_clean_etkdg.sdf --output_train ./data/Chirality/chirbase_clean_etkdg_train.sdf --output_test ./data/Chirality/chirbase_clean_etkdg_test.sdf
# Load 78598/35898 data from ./data/Chirality/chirbase_clean_etkdg.sdf
# Get 32488 training data, 3589 test data
# python ./preprocess/random_split_sdf.py --input ./data/Chirality/chirbase_clean_omega.sdf --output_train ./data/Chirality/chirbase_clean_omega_train.sdf --output_test ./data/Chirality/chirbase_clean_omega_test.sdf
```

### 2. Train & Eval

```bash
nohup bash ./experiments/molnet_char_etkdg_10fold.sh > molnet_char_etkdg_10fold.out 

nohup bash ./experiments/molnet_char_etkdg_5fold.sh > molnet_char_etkdg_5fold.out 

# multiple charility phases
nohup python main_char_kfold.py --config ./configs/molnet_chirality_multi_cls_etkdg.yaml --multi_csp True --k_fold 5 \
                            --log_dir ./logs/molnet_chirality/ \
                            --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp_multi.pt \
                            --device 1 > molnet_chirality_cls_etkdg_csp_multi.out 
```

```bash
# match csp no and name
grep -m 10 '^15723' chirbase.sdf -B 20 
```

## Multi-charility phase (multi-task learning)

### 1. Preprocess

```bash

```

### 2. Train & Eval

```bash

```

