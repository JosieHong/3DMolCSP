###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {9..17}
do
    if [[ "$VARIABLE" =~ ^(0|1|3|5|14|15)$ ]]; then
        echo "python main_chir_kfold.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE.pt --device 2"

        python main_chir_kfold.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --k_fold 5 --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.pt \
                                    --result_path ./results/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.csv \
                                    --device 2

        echo "Done!"
    else
        echo "python main_chir_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE.pt --device 2"

    python main_chir_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.pt \
                                --result_path ./results/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.csv \
                                --device 2

    echo "Done!"
    fi
done