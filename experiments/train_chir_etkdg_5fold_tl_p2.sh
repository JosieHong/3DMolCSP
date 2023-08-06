###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {0..17}
do
    if [[ "$VARIABLE" =~ ^(0|1|3|5|7|12|14|15|17)$ ]]; then
        echo "python main_chir_kfold.py --config ./configs/molnet_train_s.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--resume_path ./check_point/molnet_agilent.pt \
--transfer \
--checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
--result_path ./results0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
--device 1"

        python main_chir_kfold.py --config ./configs/molnet_train_s.yaml --k_fold 5 --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --resume_path ./check_point/molnet_agilent.pt \
                                    --transfer \
                                    --checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
                                    --result_path ./results0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
                                    --device 1

        echo "Done!"
    else
        echo "python main_chir_kfold.py --config ./configs/molnet_train_s.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--resume_path ./check_point/molnet_agilent.pt \
--transfer \
--checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
--result_path ./results0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
--device 1"

    python main_chir_kfold.py --config ./configs/molnet_train_l.yaml --k_fold 5 --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --resume_path ./check_point/molnet_agilent.pt \
                                --transfer \
                                --checkpoint ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.pt \
                                --result_path ./results0804/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold_tl.csv \
                                --device 1

    echo "Done!"
    fi
done