###
 # @Date: 2022-11-29 13:42:46
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-11-29 13:43:51
### 

for VARIABLE in {0..17}
do
    if [[ "$VARIABLE" =~ ^(0|1|3|5|14|15)$ ]]; then
        echo "python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE.pt --device 0"

        python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --k_fold 5 --csp_no $VARIABLE \
                                    --log_dir ./logs/molnet_chirality/ \
                                    --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.pt \
                                    --result_path ./results/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.csv \
                                    --device 0

        echo "Done!"
    else
        echo "python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no $VARIABLE \
--log_dir ./logs/molnet_chirality/ \
--checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE.pt --device 0"

    python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no $VARIABLE \
                                --log_dir ./logs/molnet_chirality/ \
                                --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.pt \
                                --result_path ./results/molnet_chirality_cls_etkdg_csp$VARIABLE-5fold.csv \
                                --device 0

    echo "Done!"
    fi
done

# try diff conf
# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no 16 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp16-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp16-5fold.csv \
#                                 --device 2 > nohup_csp16.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --k_fold 5 --csp_no 1 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp1-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp1-5fold.csv \
#                                 --device 2 > nohup_csp1.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no 11 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp11-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp11-5fold.csv \
#                                 --device 2 > nohup_csp11.out

nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_tl.yaml --k_fold 5 --csp_no 2 \
                                --log_dir ./logs/molnet_chirality/ \
                                --resume_path ./check_point/molnet_agilent.pt \
                                --transfer \
                                --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp2-5fold_tl.pt \
                                --result_path ./results/molnet_chirality_cls_etkdg_csp2-5fold_tl.csv \
                                --device 2 > nohup_csp2_tl.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no 7 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp7-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp7-5fold.csv \
#                                 --device 2 > nohup_csp7.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no 4 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp4-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp4-5fold.csv \
#                                 --device 2 > nohup_csp4.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no 12 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp12-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp12-5fold.csv \
#                                 --device 0 > nohup_csp12.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --k_fold 5 --csp_no 5 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp5-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp5-5fold.csv \
#                                 --device 2 > nohup_csp5.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --k_fold 5 --csp_no 15 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp15-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp15-5fold.csv \
#                                 --device 2 > nohup_csp15.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no 13 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp13-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp13-5fold.csv \
#                                 --device 0 > nohup_csp13.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no 9 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp9-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp9-5fold.csv \
#                                 --device 2 > nohup_csp9.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no 10 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp10-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp10-5fold.csv \
#                                 --device 0 > nohup_csp10.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no 8 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp8-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp8-5fold.csv \
#                                 --device 0 > nohup_csp8.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no 6 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp6-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp6-5fold.csv \
#                                 --device 2 > nohup_csp6.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --k_fold 5 --csp_no 3 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp3-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp3-5fold.csv \
#                                 --device 0 > nohup_csp3.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --k_fold 5 --csp_no 14 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp14-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp14-5fold.csv \
#                                 --device 0 > nohup_csp14.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_S.yaml --k_fold 5 --csp_no 0 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp0-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp0-5fold.csv \
#                                 --device 1 > nohup_csp0.out

# nohup python main_char_kfold.py --config ./configs/molnet_chirality_cls_etkdg_L.yaml --k_fold 5 --csp_no 17 \
#                                 --log_dir ./logs/molnet_chirality/ \
#                                 --checkpoint ./check_point/molnet_chirality_cls_etkdg_csp17-5fold.pt \
#                                 --result_path ./results/molnet_chirality_cls_etkdg_csp17-5fold.csv \
#                                 --device 1 > nohup_csp17.out

