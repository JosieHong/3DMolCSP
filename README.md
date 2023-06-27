# 3DMolChar



### 1. Preprocess

```bash
python ./preprocess/preprocess_chirality.py --input ./data/Chirality/chirbase.sdf --output ./data/Chirality/chirbase_clean.sdf
python ./preprocess/preprocess_chirality.py --input ./data/Chirality/chirbase.sdf --output ./data/Chirality/chirbase_clean2.sdf

python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean.sdf --conf_type etkdg # slow
# python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean.sdf --conf_type omega # slow
python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean2.sdf --conf_type etkdg

# we do not need spliting dataset since we apply k-fold validation
# python ./preprocess/random_split_sdf.py --input ./data/Chirality/chirbase_clean_etkdg.sdf --output_train ./data/Chirality/chirbase_clean_etkdg_train.sdf --output_test ./data/Chirality/chirbase_clean_etkdg_test.sdf
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
nohup bash ./experiments/molnet_char_etkdg_5fold_p1.sh > molnet_char_etkdg_5fold_p1.out 
nohup bash ./experiments/molnet_char_etkdg_5fold_p2.sh > molnet_char_etkdg_5fold_p2.out 
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

nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_multi_L.yaml --k_fold 5 \
                            --log_dir ./logs/molnet_chirality/ \
                            --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp_multi_L.pt \
                            --result_path ./results/molnet_chirality_cls_etkdg_csp_multi_L.csv \
                            --device 0 > molnet_char_etkdg_5fold_multi_L.out
```

### Jupyter Notebook

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
