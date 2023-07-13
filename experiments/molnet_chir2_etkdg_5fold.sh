###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {0..17}
do
    echo "python main_chir2_kfold.py --config ./configs/molnet2_chirality_cls_etkdg.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE.pt --device 0"

    python main_chir2_kfold.py --config ./configs/molnet2_chirality_cls_etkdg.yaml --k_fold 5 --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --checkpoint ./check_point2/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.pt \
                                --result_path ./results2/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.csv \
                                --device 0

    echo "Done!"
done
