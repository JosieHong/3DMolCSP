###
 # @Date: 2022-11-29 13:40:45
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:41:20
### 
python ./preprocess/preprocess_chirality.py --input ./data/Chirality/chirbase.sdf --output ./data/Chirality/chirbase_clean.sdf

python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean.sdf --dataset chira --conf_type etkdg # slow
# python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean.sdf --dataset chira --conf_type omega # slow

python ./preprocess/random_split_sdf.py --input ./data/Chirality/chirbase_clean.sdf --output_train ./data/Chirality/chirbase_clean_train.sdf --output_test ./data/Chirality/chirbase_clean_test.sdf
# Load 78598/35816 data from ./data/Chirality/chirbase_clean.sdf
# Get 32408 training data, 3581 test data
python ./preprocess/random_split_sdf.py --input ./data/Chirality/chirbase_clean_etkdg.sdf --output_train ./data/Chirality/chirbase_clean_etkdg_train.sdf --output_test ./data/Chirality/chirbase_clean_etkdg_test.sdf
# Load 78598/35898 data from ./data/Chirality/chirbase_clean_etkdg.sdf
# Get 32488 training data, 3589 test data
# python ./preprocess/random_split_sdf.py --input ./data/Chirality/chirbase_clean_omega.sdf --output_train ./data/Chirality/chirbase_clean_omega_train.sdf --output_test ./data/Chirality/chirbase_clean_omega_test.sdf
