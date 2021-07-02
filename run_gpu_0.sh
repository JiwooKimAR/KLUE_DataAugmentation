#!/bin/bash

lr=("1e-5" "2e-5" "3e-5" "5e-5")
wr=(0 0.1 0.2 0.6)
wd=(0 0.01)
batch_size=(16 32)
total_epochs=(3)

for j in ${lr[@]}; do
	for k in ${wr[@]}; do
		for l in ${wd[@]}; do
			for m in ${batch_size[@]}; do
				for n in ${total_epochs[@]}; do
					echo `CUDA_VISIBLE_DEVICES=0 python main_all.py --task 3 --output_dir checkpoint --lr ${j} --wr ${k} --wd ${l} --batch_size ${m} --total_epochs ${n} --aug_bt False --aug_rd False --aug_rs False --result_dir results_all/`
				done
			done
		done
	done
done

