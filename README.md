# 3DMolChar



## Single charility phase

### 1. Preprocess

```bash
python ./preprocess/preprocess_chirality.py --input ./data/Chirality/chirbase.sdf --output ./data/Chirality/chirbase_clean.sdf

python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean.sdf --dataset chira --conf_type etkdg # slow
# python ./preprocess/gen_conformers.py --path ./data/Chirality/chirbase_clean.sdf --dataset chira --conf_type omega # slow

python ./preprocess/random_split_sdf.py --input ./data/Chirality/chirbase_clean.sdf --output_train ./data/Chirality/chirbase_clean_train.sdf --output_test ./data/Chirality/chirbase_clean_test.sdf
# Load 78598/35816 data from ./data/Chirality/chirbase_clean.sdf
# Get 32408 training data, 3581 test data
python ./preprocess/random_split_sdf.py --input ./data/Chirality/chirbase_clean_etkdg.sdf --output_train ./data/Chirality/chirbase_clean_etkdg_train.sdf --output_test ./data/Chirality/chirbase_clean_etkdg_test.sdf
# Load 78598/35898 data from ./data/Chirality/chirbase_clean_etkdg.sdf
# Get 32488 training data, 3589 test data
# python ./preprocess/random_split_sdf.py --input ./data/Chirality/chirbase_clean_omega.sdf --output_train ./data/Chirality/chirbase_clean_omega_train.sdf --output_test ./data/Chirality/chirbase_clean_omega_test.sdf
```

### 2. Train & Eval

```bash
nohup bash ./experiments/molnet_char_etkdg.sh > molnet_char_etkdg.out 

# --------------- Final Results 0 --------------- #
fold_0: acc: 0.765625, auc: 0.8207879486105293
fold_1: acc: 0.828125, auc: 0.9003992354999779
fold_2: acc: 0.90625, auc: 0.9796717171717172
fold_3: acc: 0.84375, auc: 0.8162789843595782
fold_4: acc: 0.8125, auc: 0.7899604548759583
fold_5: acc: 0.875, auc: 0.9680898850253689
fold_6: acc: 0.875, auc: 0.845906793570728
fold_7: acc: 0.921875, auc: 0.9478509527555127
fold_8: acc: 0.90625, auc: 0.9836932073774177
fold_9: acc: 0.9375, auc: 0.9986947271577252
mean acc: 0.8671875, mean auc: 0.9051333906404512
--
# --------------- Final Results 1 --------------- #
fold_0: acc: 0.8671875, auc: 0.9517275013270368
fold_1: acc: 0.8515625, auc: 0.9279135971518317
fold_2: acc: 0.828125, auc: 0.9443457534094294
fold_3: acc: 0.859375, auc: 0.9151928481493385
fold_4: acc: 0.86328125, auc: 0.9489531918321853
fold_5: acc: 0.90625, auc: 0.9626238075511937
fold_6: acc: 0.859375, auc: 0.9618031124752341
fold_7: acc: 0.89453125, auc: 0.9521250897194649
fold_8: acc: 0.8515625, auc: 0.931195457793983
fold_9: acc: 0.87109375, auc: 0.9230688548016476
mean acc: 0.865234375, mean auc: 0.9418949214211345
--
# --------------- Final Results 2 --------------- #
fold_0: acc: 0.5692567567567568, auc: 0.6000764372752267
fold_1: acc: 0.5536317567567568, auc: 0.7003772764814057
fold_2: acc: 0.5945945945945946, auc: 0.7121036888282867
fold_3: acc: 0.5924831081081081, auc: 0.6704872148433756
fold_4: acc: 0.6558277027027027, auc: 0.7982852443244169
fold_5: acc: 0.6228885135135135, auc: 0.7509712634393854
fold_6: acc: 0.59375, auc: 0.7089953466711107
fold_7: acc: 0.6043074324324325, auc: 0.7423494941880538
fold_8: acc: 0.613597972972973, auc: 0.7125233646098023
fold_9: acc: 0.6372466216216216, auc: 0.7847011239455184
mean acc: 0.6037584459459461, mean auc: 0.7180870454606583
--
# --------------- Final Results 3 --------------- #
fold_0: acc: 0.65625, auc: 0.8389529907474378
fold_1: acc: 0.8515625, auc: 0.9384180107465966
fold_2: acc: 0.7734375, auc: 0.9004006282255407
fold_3: acc: 0.7734375, auc: 0.9062574561609226
fold_4: acc: 0.8515625, auc: 0.9427917984401631
fold_5: acc: 0.8515625, auc: 0.9279534399056866
fold_6: acc: 0.828125, auc: 0.8987223222788764
fold_7: acc: 0.8359375, auc: 0.9164664923712307
fold_8: acc: 0.8828125, auc: 0.9539455482269802
fold_9: acc: 0.84375, auc: 0.9285812298420236
mean acc: 0.81484375, mean auc: 0.9152489916945459
--
# --------------- Final Results 4 --------------- #
fold_0: acc: 0.9375, auc: 0.9926615280038105
fold_1: acc: 0.921875, auc: 0.9142167698202183
fold_2: acc: 0.953125, auc: 0.9721453365852616
fold_3: acc: 0.921875, auc: 0.8771480017130342
fold_4: acc: 0.9375, auc: 0.9438744141163496
fold_5: acc: 0.96875, auc: 1.0
fold_6: acc: 0.984375, auc: 0.9944533165092696
fold_7: acc: 0.9375, auc: 0.9913723212081935
fold_8: acc: 0.984375, auc: 0.9549435028248587
fold_9: acc: 0.953125, auc: 0.987615218537876
mean acc: 0.95, mean auc: 0.9628430409318872
--
# --------------- Final Results 5 --------------- #
fold_0: acc: 0.921875, auc: 0.9510929491963974
fold_1: acc: 0.90625, auc: 0.9662276402842441
fold_2: acc: 0.875, auc: 0.929821867266019
fold_3: acc: 0.90625, auc: 0.8275266626017564
fold_4: acc: 0.921875, auc: 0.9761113747824711
fold_5: acc: 0.90625, auc: 0.8768073797121826
fold_6: acc: 0.953125, auc: 0.9652894458541944
fold_7: acc: 0.9375, auc: 0.9802710770860417
fold_8: acc: 0.953125, auc: 0.9934964047025997
fold_9: acc: 0.875, auc: 0.9157716280748515
mean acc: 0.915625, mean auc: 0.9382416429560759
--
# --------------- Final Results 6 --------------- #
fold_0: acc: 0.9015625, auc: 0.9617621314061943
fold_1: acc: 0.9171875, auc: 0.9655150835151661
fold_2: acc: 0.921875, auc: 0.9666434697224084
fold_3: acc: 0.8203125, auc: 0.9031802404553444
fold_4: acc: 0.8359375, auc: 0.916186862594929
fold_5: acc: 0.91875, auc: 0.9597626032767413
fold_6: acc: 0.8234375, auc: 0.9128079063383857
fold_7: acc: 0.903125, auc: 0.9628810697577476
fold_8: acc: 0.9046875, auc: 0.9552017932203194
fold_9: acc: 0.8984375, auc: 0.9499780063942328
mean acc: 0.8845312499999999, mean auc: 0.9453919166681469
--
# --------------- Final Results 7 --------------- #
fold_0: acc: 0.84375, auc: 0.9357218013468014
fold_1: acc: 0.859375, auc: 0.9566432323342351
fold_2: acc: 0.84375, auc: 0.9411925296936104
fold_3: acc: 0.953125, auc: 0.9937134502923977
fold_4: acc: 0.921875, auc: 0.9818025812648328
fold_5: acc: 0.921875, auc: 0.9769014550264551
fold_6: acc: 0.953125, auc: 0.9897972470238096
fold_7: acc: 0.890625, auc: 0.9301188292245035
fold_8: acc: 0.921875, auc: 0.9947076994812613
fold_9: acc: 0.921875, auc: 0.9760235377289835
mean acc: 0.903125, mean auc: 0.9676622363416889
--
# --------------- Final Results 8 --------------- #
fold_0: acc: 0.796875, auc: 0.8553918592981092
fold_1: acc: 0.9375, auc: 0.9735352515204053
fold_2: acc: 0.9375, auc: 0.9616948195588163
fold_3: acc: 0.921875, auc: 0.9534298822810817
fold_4: acc: 0.953125, auc: 0.9846564922836109
fold_5: acc: 0.9375, auc: 0.9650560272607175
fold_6: acc: 0.90625, auc: 0.9661535632123868
fold_7: acc: 0.953125, auc: 0.963682745825603
fold_8: acc: 0.90625, auc: 0.9645267013688067
fold_9: acc: 0.90625, auc: 0.9266034339909579
mean acc: 0.915625, mean auc: 0.9514730776600494
--
# --------------- Final Results 9 --------------- #
fold_0: acc: 0.593125, auc: 0.6755358387322179
fold_1: acc: 0.59625, auc: 0.7401422753145456
fold_2: acc: 0.601875, auc: 0.7643959688354701
fold_3: acc: 0.665, auc: 0.8272185876085407
fold_4: acc: 0.646875, auc: 0.7500843554153627
fold_5: acc: 0.616875, auc: 0.7686264592260597
fold_6: acc: 0.645625, auc: 0.767425067881932
fold_7: acc: 0.705625, auc: 0.8645564145577463
fold_8: acc: 0.731875, auc: 0.8434999216404707
fold_9: acc: 0.674375, auc: 0.8085333738252872
mean acc: 0.64775, mean auc: 0.7810018263037634
--
# --------------- Final Results 10 --------------- #
fold_0: acc: 0.890625, auc: 0.9236351950642399
fold_1: acc: 0.96875, auc: 0.9529004351519564
fold_2: acc: 0.96875, auc: 0.9878641323953824
fold_3: acc: 0.9375, auc: 0.9683574014401083
fold_4: acc: 0.90625, auc: 0.941220104524878
fold_5: acc: 0.984375, auc: 0.995619658119658
fold_6: acc: 0.9375, auc: 0.9004677841360563
fold_7: acc: 0.953125, auc: 0.8719932223175147
fold_8: acc: 0.953125, auc: nan
fold_9: acc: 0.96875, auc: 0.954067708954927
mean acc: 0.946875, mean auc: nan
--
# --------------- Final Results 11 --------------- #
fold_0: acc: 0.796875, auc: 0.7933878077976527
fold_1: acc: 0.859375, auc: 0.9707224357300569
fold_2: acc: 0.890625, auc: 0.9720558449074074
fold_3: acc: 0.84375, auc: 0.9022527180421918
fold_4: acc: 0.90625, auc: 0.9586584650729387
fold_5: acc: 0.9375, auc: 0.9940242128309279
fold_6: acc: 0.90625, auc: 0.9195784144171241
fold_7: acc: 0.90625, auc: 0.8503050104297584
fold_8: acc: 0.921875, auc: 0.9844012883668056
fold_9: acc: 0.875, auc: 0.9690109799034614
mean acc: 0.884375, mean auc: 0.9314397177498325
--
# --------------- Final Results 12 --------------- #
fold_0: acc: 0.9583333333333334, auc: 0.9848207912602329
fold_1: acc: 0.9427083333333334, auc: 0.9910327193402252
fold_2: acc: 0.8854166666666666, auc: 0.966916433896657
fold_3: acc: 0.9270833333333334, auc: 0.955862954281573
fold_4: acc: 0.9375, auc: 0.9645841527181588
fold_5: acc: 0.953125, auc: 0.9695421002879515
fold_6: acc: 0.9375, auc: 0.9511510501261409
fold_7: acc: 0.9479166666666666, auc: 0.9692634127817095
fold_8: acc: 0.9427083333333334, auc: 0.9598477937810298
fold_9: acc: 0.953125, auc: 0.9710588224487501
mean acc: 0.9385416666666668, mean auc: 0.9684080230922429
--
# --------------- Final Results 13 --------------- #
fold_0: acc: 0.6875, auc: 0.8521332547319918
fold_1: acc: 0.84375, auc: 0.9241384346647505
fold_2: acc: 0.890625, auc: 0.9696067929019874
fold_3: acc: 0.875, auc: 0.9272651677824092
fold_4: acc: 0.9375, auc: 0.9905658384043273
fold_5: acc: 0.875, auc: 0.957332412467748
fold_6: acc: 0.890625, auc: 0.9094643102540765
fold_7: acc: 0.875, auc: 0.9358344447478574
fold_8: acc: 0.921875, auc: 0.9638062169312169
fold_9: acc: 0.875, auc: 0.9685216523713978
mean acc: 0.8671875, mean auc: 0.9398668525257762
--
# --------------- Final Results 14 --------------- #
fold_0: acc: 0.8046875, auc: 0.9046369130006632
fold_1: acc: 0.82421875, auc: 0.8878386301111677
fold_2: acc: 0.86328125, auc: 0.9069041951958567
fold_3: acc: 0.9140625, auc: 0.9573698686725506
fold_4: acc: 0.89453125, auc: 0.9364248669584256
fold_5: acc: 0.95703125, auc: 0.9773627278554029
fold_6: acc: 0.90625, auc: 0.9743589219163052
fold_7: acc: 0.94140625, auc: 0.9465895783078876
fold_8: acc: 0.921875, auc: 0.9805312725963647
fold_9: acc: 0.92578125, auc: 0.9765071813409844
mean acc: 0.8953125, mean auc: 0.9448524155955609
--
# --------------- Final Results 15 --------------- #
fold_0: acc: 0.90625, auc: 0.951529558683173
fold_1: acc: 0.8515625, auc: 0.8507666482842186
fold_2: acc: 0.9140625, auc: 0.9753191874179669
fold_3: acc: 0.9375, auc: 0.9856907452851975
fold_4: acc: 0.9140625, auc: 0.9626763231464279
fold_5: acc: 0.8671875, auc: 0.9655609550207146
fold_6: acc: 0.921875, auc: 0.9819390444390445
fold_7: acc: 0.9296875, auc: 0.9827087227170836
fold_8: acc: 0.9453125, auc: 0.9898649250962462
fold_9: acc: 0.921875, auc: 0.9918998121336888
mean acc: 0.9109375, mean auc: 0.963795592222376
--
# --------------- Final Results 16 --------------- #
fold_0: acc: 0.8125, auc: 0.9070705539770056
fold_1: acc: 0.8645833333333334, auc: 0.9333156540922795
fold_2: acc: 0.890625, auc: 0.963954447259553
fold_3: acc: 0.8541666666666666, auc: 0.9245320749113057
fold_4: acc: 0.8541666666666666, auc: 0.9568401959537983
fold_5: acc: 0.90625, auc: 0.9664522062505322
fold_6: acc: 0.8489583333333334, auc: 0.9131676368155478
fold_7: acc: 0.9010416666666666, auc: 0.9768136480209808
fold_8: acc: 0.9114583333333334, auc: 0.9729905739082376
fold_9: acc: 0.921875, auc: 0.9789626505075955
mean acc: 0.8765625, mean auc: 0.9494099641696836
--
# --------------- Final Results 17 --------------- #
fold_0: acc: 0.916015625, auc: 0.9658998610711613
fold_1: acc: 0.83203125, auc: 0.8977421564055308
fold_2: acc: 0.853515625, auc: 0.9231864097575512
fold_3: acc: 0.87109375, auc: 0.9448807613975033
fold_4: acc: 0.8203125, auc: 0.9036320524045217
fold_5: acc: 0.859375, auc: 0.916860993877426
fold_6: acc: 0.87890625, auc: 0.9431543306631182
fold_7: acc: 0.87890625, auc: 0.9489533101360447
fold_8: acc: 0.876953125, auc: 0.9400763942103234
fold_9: acc: 0.837890625, auc: 0.9215710654552166
mean acc: 0.8625, mean auc: 0.9305957335378396
--
# --------------- Final Results 18 --------------- #
fold_0: acc: 0.9427083333333334, auc: 0.9827136702564427
fold_1: acc: 0.9375, auc: 0.9863519815294303
fold_2: acc: 0.9583333333333334, auc: 0.993034173757147
fold_3: acc: 0.9635416666666666, auc: 0.988260690478091
fold_4: acc: 0.9114583333333334, auc: 0.9601565254327903
fold_5: acc: 0.9114583333333334, auc: 0.9372211418543474
fold_6: acc: 0.9427083333333334, auc: 0.9776528197554262
fold_7: acc: 0.921875, auc: 0.9511332730403593
fold_8: acc: 0.8958333333333334, auc: 0.9394315300521882
fold_9: acc: 0.921875, auc: 0.9575633872508873
mean acc: 0.9307291666666666, mean auc: 0.967351919340711
--
# --------------- Final Results 19 --------------- #
fold_0: acc: 0.859375, auc: 0.9305018744596499
fold_1: acc: 0.828125, auc: 0.9234192582975354
fold_2: acc: 0.890625, auc: 0.9610424935453432
fold_3: acc: 0.875, auc: 0.9191664608436961
fold_4: acc: 0.859375, auc: 0.934318026172099
fold_5: acc: 0.9166666666666666, auc: 0.9919440976759346
fold_6: acc: 0.90625, auc: 0.9735737735282356
fold_7: acc: 0.859375, auc: 0.9492067760977715
fold_8: acc: 0.8854166666666666, auc: 0.9585157108248153
fold_9: acc: 0.875, auc: 0.9325623355075171
mean acc: 0.8755208333333334, mean auc: 0.9474250806952599

# multiple charility phases
python main_char_kfold.py --config ./configs/molnet_chirality_multi_cls_etkdg.yaml --multi_csp True \
                            --log_dir ./logs/molnet_chirality/ \
                            --checkpoint ./check_point/molnet_chirality_multi_cls_etkdg.pt

# Traceback (most recent call last):
#   File "main_char_kfold.py", line 272, in <module>
#     y_true, y_pred = train(model, device, train_loader, optimizer, config['train_para']['accum_iter'], config['train_para']['batch_size'], config['model_para']['num_atoms'], config['model_para']['out_channels'])
#   File "main_char_kfold.py", line 54, in train
#     y = F.one_hot(y, num_classes=out_cls).to(torch.float32)
# RuntimeError: CUDA error: device-side assert triggered
```



## Multi-charility phase (multi-task learning)

### 1. Preprocess

```bash

```

### 2. Train & Eval

```bash

```

