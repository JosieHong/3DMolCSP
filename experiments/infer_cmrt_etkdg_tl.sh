#! /bin/bash

for VARIABLE in {0..17}
do
	echo "python infer_chir.py --config ./configs/molnet_non_train.yaml --csp_no $VARIABLE \
					--resume_path ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
					--result_path ./results0828/molnet_cmrt_cls_etkdg_csp$VARIABLE-ena.csv \
					--device 0"

	python infer_chir.py --config ./configs/molnet_non_train.yaml --csp_no $VARIABLE \
					--resume_path ./check_point0804/molnet_chirality_cls_etkdg_csp$VARIABLE-tl.pt \
					--result_path ./results0828/molnet_cmrt_cls_etkdg_csp$VARIABLE-ena.csv \
					--device 0

	echo "Done!"
done
