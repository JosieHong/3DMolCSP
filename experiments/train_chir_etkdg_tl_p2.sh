#!/bin/bash
###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

start=$(date +%s)

for VARIABLE in {9..17}
do
    if [[ "$VARIABLE" =~ ^(4|16|7|10)$ ]]; then
        echo "python main_chir.py --config ./configs/molnet_train_l.yaml --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --resume_path ./check_point/molnetv2_qtof_etkdgv3.pt \
                                    --transfer \
                                    --checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                                    --device 0" 

        python main_chir.py --config ./configs/molnet_train_l.yaml --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --resume_path ./check_point/molnetv2_qtof_etkdgv3.pt \
                                    --transfer \
                                    --checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                                    --device 0
    elif [[ "$VARIABLE" =~ ^(8|11|6|9)$ ]]; then
        echo "python main_chir.py --config ./configs/molnet_train_xl.yaml --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --resume_path ./check_point/molnetv2_qtof_etkdgv3.pt \
                                    --transfer \
                                    --checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                                    --device 0"

        python main_chir.py --config ./configs/molnet_train_xl.yaml --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --resume_path ./check_point/molnetv2_qtof_etkdgv3.pt \
                                    --transfer \
                                    --checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                                    --device 0
    else
        echo "python main_chir.py --config ./configs/molnet_train_s.yaml --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --resume_path ./check_point/molnetv2_qtof_etkdgv3.pt \
                                    --transfer \
                                    --checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                                    --device 0"

        python main_chir.py --config ./configs/molnet_train_s.yaml --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --resume_path ./check_point/molnetv2_qtof_etkdgv3.pt \
                                    --transfer \
                                    --checkpoint ./check_point1203/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                                    --device 0
    fi
    echo "Done!"
done

end=$(date +%s)

echo "Elapsed Time: $(($end-$start)) seconds"
