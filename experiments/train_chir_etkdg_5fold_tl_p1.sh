###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {0..8}
do
    if [[ "$VARIABLE" =~ ^(2|14|0|15|13|5|12|3|1|17)$ ]]; then
        echo "python main_chir_kfold.py --config ./configs/molnet_train_s.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--resume_path ./check_point/molnet_agilent.pt \
--transfer \
--checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
--result_path ./results0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
--device 0"

        python main_chir_kfold.py --config ./configs/molnet_train_s.yaml --k_fold 5 --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --resume_path ./check_point/molnet_agilent.pt \
                                    --transfer \
                                    --checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
                                    --result_path ./results0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
                                    --device 0

        echo "Done!"
    elif [[ "$VARIABLE" =~ ^(8|11|6|9)$ ]]; then
        echo "python main_chir_kfold.py --config ./configs/molnet_train_xl.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--resume_path ./check_point/molnet_agilent.pt \
--transfer \
--checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
--result_path ./results0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
--device 0"

        python main_chir_kfold.py --config ./configs/molnet_train_xl.yaml --k_fold 5 --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --resume_path ./check_point/molnet_agilent.pt \
                                    --transfer \
                                    --checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
                                    --result_path ./results0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
                                    --device 0

        echo "Done!"
    else
        echo "python main_chir_kfold.py --config ./configs/molnet_train_l.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--resume_path ./check_point/molnet_agilent.pt \
--transfer \
--checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
--result_path ./results0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
--device 0"

        python main_chir_kfold.py --config ./configs/molnet_train_l.yaml --k_fold 5 --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --resume_path ./check_point/molnet_agilent.pt \
                                    --transfer \
                                    --checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
                                    --result_path ./results0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
                                    --device 0

        echo "Done!"
    fi
done

# nohup python main_chir_kfold.py --config ./configs/molnet_train_xl.yaml --k_fold 5 --csp_no 6 \
# --log_dir ./logs/molnet_chirality/ \
# --resume_path ./check_point/molnet_agilent.pt \
# --transfer \
# --checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp6-5fold_tl.pt \
# --result_path ./results0804/molnet_chirality_cls_etkdg_csp6-5fold_tl.csv \
# --device 0 > 3dmolCSP_0810_csp6.out

# nohup python main_chir_kfold.py --config ./configs/molnet_train_xl.yaml --k_fold 5 --csp_no 8 \
# --log_dir ./logs/molnet_chirality/ \
# --resume_path ./check_point/molnet_agilent.pt \
# --transfer \
# --checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp8-5fold_tl.pt \
# --result_path ./results0804/molnet_chirality_cls_etkdg_csp8-5fold_tl.csv \
# --device 0 > 3dmolCSP_0810_csp8.out

# nohup python main_chir_kfold.py --config ./configs/molnet_train_xl.yaml --k_fold 5 --csp_no 9 \
# --log_dir ./logs/molnet_chirality/ \
# --resume_path ./check_point/molnet_agilent.pt \
# --transfer \
# --checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp9-5fold_tl.pt \
# --result_path ./results0804/molnet_chirality_cls_etkdg_csp9-5fold_tl.csv \
# --device 1 > 3dmolCSP_0810_csp9.out

# nohup python main_chir_kfold.py --config ./configs/molnet_train_xl.yaml --k_fold 5 --csp_no 11 \
# --log_dir ./logs/molnet_chirality/ \
# --resume_path ./check_point/molnet_agilent.pt \
# --transfer \
# --checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp11-5fold_tl.pt \
# --result_path ./results0804/molnet_chirality_cls_etkdg_csp11-5fold_tl.csv \
# --device 0 > 3dmolCSP_0810_csp11.out

# nohup python main_chir_kfold.py --config ./configs/molnet_train_xl.yaml --k_fold 5 --csp_no 15 \
# --log_dir ./logs/molnet_chirality/ \
# --resume_path ./check_point/molnet_agilent.pt \
# --transfer \
# --checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp15-5fold_tl.pt \
# --result_path ./results0804/molnet_chirality_cls_etkdg_csp15-5fold_tl.csv \
# --device 1 > 3dmolCSP_0810_csp15.out