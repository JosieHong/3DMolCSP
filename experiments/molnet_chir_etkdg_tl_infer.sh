###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {0..17}
do
    
    echo "python infer_chir.py --config ./configs/molnet_chirality_cls_etkdg_tl.yaml --csp_no $VARIABLE \
                    --resume_path ./check_point_all/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                    --result_path ./results_all/molnet_chirality_cls_etkdg_csp$VARIABLE.csv \
                    --device 0"

    python infer_chir.py --config ./configs/molnet_chirality_cls_etkdg_tl.yaml --csp_no $VARIABLE \
                    --resume_path ./check_point_all/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
                    --result_path ./results_all/molnet_chirality_cls_etkdg_csp$VARIABLE.csv \
                    --device 0

    echo "Done!"
done
