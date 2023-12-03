#!/bin/bash

# ad
# python ./preprocess/preprocess_chirbase_eo.py \
# --input_eo ./data/ChirBase_eo/w_ena/ad_sr.sdf \
# --input ./data/ChirBase/chirbase.sdf \
# --csp_setting ./preprocess/chirality_stationary_phase_list.csv \
# --output ./data/ChirBase_eo/exp/ad_sr_clean.sdf 

python main_chir_eo.py --config ./configs/molnet_train_eo_ad.yaml \
--checkpoint ./check_point_eo/molnet_eo_ad.pt \
--result_path ./results_eo/molnet_eo_ad.csv 

# ia
python ./preprocess/preprocess_chirbase_eo.py \
--input_eo ./data/ChirBase_eo/w_ena/ia_sr.sdf \
--input ./data/ChirBase/chirbase.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv \
--output ./data/ChirBase_eo/exp/ia_sr_clean.sdf

python main_chir_eo.py --config ./configs/molnet_train_eo_ia.yaml \
--checkpoint ./check_point_eo/molnet_eo_ia.pt \
--result_path ./results_eo/molnet_eo_ia.csv 

# ic
python ./preprocess/preprocess_chirbase_eo.py \
--input_eo ./data/ChirBase_eo/w_ena/ic_sr.sdf \
--input ./data/ChirBase/chirbase.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv \
--output ./data/ChirBase_eo/exp/ic_sr_clean.sdf

python main_chir_eo.py --config ./configs/molnet_train_eo_ic.yaml \
--checkpoint ./check_point_eo/molnet_eo_ic.pt \
--result_path ./results_eo/molnet_eo_ic.csv  

# od 
python ./preprocess/preprocess_chirbase_eo.py \
--input_eo ./data/ChirBase_eo/w_ena/od_sr.sdf \
--input ./data/ChirBase/chirbase.sdf \
--csp_setting ./preprocess/chirality_stationary_phase_list.csv \
--output ./data/ChirBase_eo/exp/od_sr_clean.sdf

python main_chir_eo.py --config ./configs/molnet_train_eo_od.yaml \
--checkpoint ./check_point_eo/molnet_eo_od.pt \
--result_path ./results_eo/molnet_eo_od.csv 

