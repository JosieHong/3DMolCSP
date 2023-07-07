###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {0..17}
do
    
    echo "python main_char.py --config ./configs/molnet_chirality_cls_etkdg_tl.yaml --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --resume_path ./check_point/molnet_agilent.pt \
                                --transfer \
                                --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                                --device 2"

    python main_char.py --config ./configs/molnet_chirality_cls_etkdg_tl.yaml --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --resume_path ./check_point/molnet_agilent.pt \
                                --transfer \
                                --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                                --device 2

    echo "Done!"
done
