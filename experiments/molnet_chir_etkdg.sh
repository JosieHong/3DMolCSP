###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {0..17}
do
    if [[ "$VARIABLE" =~ ^(0|1|3|5|14|15)$ ]]; then
        echo "python main_chir.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point_all/molnet_chirality_cls_etkdg_csp$VARIABLE.pt --device 0"

        python main_chir.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --checkpoint ./check_point_all/molnet_chirality_cls_etkdg_csp$VARIABLE.pt \
                                    --result_path ./results/molnet_chirality_cls_etkdg_csp$VARIABLE.csv \
                                    --device 0

        echo "Done!"
    else
        echo "python main_chir.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point_all/molnet_chirality_cls_etkdg_csp$VARIABLE.pt --device 0"

    python main_chir.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --checkpoint ./check_point_all/molnet_chirality_cls_etkdg_csp$VARIABLE.pt \
                                --result_path ./results/molnet_chirality_cls_etkdg_csp$VARIABLE.csv \
                                --device 0

    echo "Done!"
    fi
done
