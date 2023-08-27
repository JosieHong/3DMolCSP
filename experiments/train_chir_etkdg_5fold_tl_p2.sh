###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {9..17}
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

