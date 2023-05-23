###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {0..19}
do
    echo "python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg.yaml --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE.pt "

    python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg.yaml --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE.pt

    echo "Done!"
done

# python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg.yaml --csp_no 1 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/tmp.pt