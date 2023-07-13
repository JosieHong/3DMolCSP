#!/bin/bash
###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

start=$(date +%s)

for VARIABLE in {9..17}
do
    
    echo "python main_chir.py --config ./configs/molnet_chirality_cls_etkdg_tl.yaml --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --resume_path ./check_point/molnet_agilent.pt \
                                --transfer \
                                --checkpoint ./check_point_all/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                                --device 1"

    python main_chir.py --config ./configs/molnet_chirality_cls_etkdg_tl.yaml --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --resume_path ./check_point/molnet_agilent.pt \
                                --transfer \
                                --checkpoint ./check_point_all/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                                --device 1

    echo "Done!"
done

end=$(date +%s)

echo "Elapsed Time: $(($end-$start)) seconds"