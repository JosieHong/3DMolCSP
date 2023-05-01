###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 
# Classification
python main_char.py --config ./configs/molnet_chirality_cls.yaml --csp_no 0 \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_csp0.pt \
--resume_path ./check_point/molnet_chirality_cls_csp0.pt \
--device 0

python main_char.py --config ./configs/molnet_chirality_cls_etkdg.yaml --csp_no 0 \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_etkdg_csp0.pt \
--resume_path ./check_point/molnet_chirality_cls_etkdg_csp0.pt \
--device 0

python main_char.py --config ./configs/molnet_chirality_cls_omega.yaml --csp_no 0 \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_omega_csp0.pt \
--resume_path ./check_point/molnet_chirality_cls_omega_csp0.pt \
--device 0

# Classification (k-fold validation)
python main_char_kfold.py --config ./configs/molnet_chirality_cls.yaml --csp_no 0 \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_csp0.pt \
--resume_path ./check_point/molnet_chirality_cls_csp0.pt \
--device 0

python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg.yaml --csp_no 0 \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_etkdg_csp0.pt \
--resume_path ./check_point/molnet_chirality_cls_etkdg_csp0.pt \
--device 0

python main_char_kfold.py --config ./configs/molnet_chirality_cls_omega.yaml --csp_no 0 \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_omega_csp0.pt \
--resume_path ./check_point/molnet_chirality_cls_omega_csp0.pt \
--device 0
