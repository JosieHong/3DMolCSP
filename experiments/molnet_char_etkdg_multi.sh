###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 
nohup python main_char_kfold.py --config ./configs/molnet_chirality_multi_cls_etkdg.yaml --multi_csp True \
                            --log_dir ./logs/molnet_chirality/ \
                            --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp_multi.pt \
                            --device 1 > molnet_chirality_cls_etkdg_csp_multi.out 


