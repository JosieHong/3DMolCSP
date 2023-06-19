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
# test
python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_eval.yaml --k_fold 5 --csp_no 0 \
                                --log_dir ./logs/molnet_chirality/ \
                                --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp0.pt \
                                --result_path ./results/molnet_chirality_cls_etkdg_csp0.csv

nohup bash ./experiments/molnet_char_etkdg_10fold.sh > molnet_char_etkdg_10fold.out 

nohup bash ./experiments/molnet_char_etkdg_5fold.sh > molnet_char_etkdg_5fold.out 
nohup bash ./experiments/molnet_char_etkdg_5fold.sh > molnet_char_etkdg_5fold_ovr.out 
```

```bash
# match csp no and name
grep -m 10 '^15723' chirbase.sdf -B 20 
```

### 3. Train & Eval (multi-task learning)

```bash
nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_multi.yaml --k_fold 5 \
                            --log_dir ./logs/molnet_chirality/ \
                            --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp_multi.pt \
                            --result_path ./results/molnet_chirality_cls_etkdg_csp_multi.csv \
                            --device 2 > molnet_char_etkdg_5fold_multi.out 
```

