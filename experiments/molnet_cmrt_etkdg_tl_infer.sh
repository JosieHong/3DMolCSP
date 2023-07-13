#! /bin/bash

for VARIABLE in {0..17}
do
	echo "python infer_chir.py --config ./configs/molnet_cmrt_cls_etkdg_tl.yaml --csp_no $VARIABLE \
					--resume_path ./check_point_all/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
					--result_path ./results_all/molnet_cmrt_cls_etkdg_csp$VARIABLE.csv \
					--device 1"

	python infer_chir.py --config ./configs/molnet_cmrt_cls_etkdg_tl.yaml --csp_no $VARIABLE \
					--resume_path ./check_point_all/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
					--result_path ./results_all/molnet_cmrt_cls_etkdg_csp$VARIABLE.csv \
					--device 1

	echo "Done!"
done

