###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {0..17}
do
    
    echo "python main_chir_kfold.py --config ./configs/molnet_non_train.yaml --k_fold 5 --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --resume_path ./check_point/molnet_agilent.pt \
                                --transfer \
                                --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
                                --result_path ./results_ena/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
                                --device 0"

    python main_chir_kfold.py --config ./configs/molnet_non_train.yaml --k_fold 5 --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --resume_path ./check_point/molnet_agilent.pt \
                                --transfer \
                                --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
                                --result_path ./results_ena/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
                                --device 0

    echo "Done!"
done
