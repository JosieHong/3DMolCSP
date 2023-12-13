###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {0..17}
do
    if [[ "$VARIABLE" =~ ^(4|16|7|10)$ ]]; then
        echo "python main_chir_kfold.py --config ./configs/molnet_train_l.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE.pt --device 0"

        python main_chir_kfold.py --config ./configs/molnet_train_l.yaml --k_fold 5 --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.pt \
                                    --result_path ./results1203/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.csv \
                                    --device 1

        echo "Done!"
    elif [[ "$VARIABLE" =~ ^(8|11|6|9)$ ]]; then
        echo "python main_chir_kfold.py --config ./configs/molnet_train_xl.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE.pt --device 0"

        python main_chir_kfold.py --config ./configs/molnet_train_xl.yaml --k_fold 5 --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.pt \
                                    --result_path ./results1203/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.csv \
                                    --device 1

        echo "Done!"
    else
        echo "python main_chir_kfold.py --config ./configs/molnet_train_s.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE.pt --device 0"

        python main_chir_kfold.py --config ./configs/molnet_train_s.yaml --k_fold 5 --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.pt \
                                    --result_path ./results1203/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.csv \
                                    --device 1

        echo "Done!"
    fi
done


