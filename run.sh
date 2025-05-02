#!/bin/bash
echo "run the ASL under different levels of 12.1"
#list=(1 2 3 4 5 6 7 8 9 10)
#for i in 0 1 2 3 4 5 6 7 8 9
#do
#	python train.py --flag ${list[i]} --output '.\output\test\flag'$i
#done

python train.py --gamma_neg 1 --gamma_pos 1 --output '.\output\mfm_aug\voc2007\fixappha\neg1\pos1'
python train.py --gamma_neg 1 --gamma_pos 2 --output '.\output\mfm_aug\voc2007\fixappha\neg1\pos2'
python train.py --gamma_neg 1 --gamma_pos 3 --output '.\output\mfm_aug\voc2007\fixappha\neg1\pos3'
python train.py --gamma_neg 1 --gamma_pos 4 --output '.\output\mfm_aug\voc2007\fixappha\neg1\pos4'
python train.py --gamma_neg 1 --gamma_pos 5 --output '.\output\mfm_aug\voc2007\fixappha\neg1\pos5'
python train.py --gamma_neg 2 --gamma_pos 1 --output '.\output\mfm_aug\voc2007\fixappha\neg2\pos1'
python train.py --gamma_neg 2 --gamma_pos 2 --output '.\output\mfm_aug\voc2007\fixappha\neg2\pos2'
python train.py --gamma_neg 2 --gamma_pos 3 --output '.\output\mfm_aug\voc2007\fixappha\neg2\pos3'
python train.py --gamma_neg 2 --gamma_pos 4 --output '.\output\mfm_aug\voc2007\fixappha\neg2\pos4'
python train.py --gamma_neg 2 --gamma_pos 5 --output '.\output\mfm_aug\voc2007\fixappha\neg2\pos5'
python train.py --gamma_neg 3 --gamma_pos 1 --output '.\output\mfm_aug\voc2007\fixappha\neg3\pos1'
python train.py --gamma_neg 3 --gamma_pos 2 --output '.\output\mfm_aug\voc2007\fixappha\neg3\pos2'
python train.py --gamma_neg 3 --gamma_pos 3 --output '.\output\mfm_aug\voc2007\fixappha\neg3\pos3'
python train.py --gamma_neg 3 --gamma_pos 4 --output '.\output\mfm_aug\voc2007\fixappha\neg3\pos4'
python train.py --gamma_neg 3 --gamma_pos 5 --output '.\output\mfm_aug\voc2007\fixappha\neg3\pos5'
python train.py --gamma_neg 4 --gamma_pos 1 --output '.\output\mfm_aug\voc2007\fixappha\neg4\pos1'
python train.py --gamma_neg 4 --gamma_pos 2 --output '.\output\mfm_aug\voc2007\fixappha\neg4\pos2'
python train.py --gamma_neg 4 --gamma_pos 3 --output '.\output\mfm_aug\voc2007\fixappha\neg4\pos3'
python train.py --gamma_neg 4 --gamma_pos 4 --output '.\output\mfm_aug\voc2007\fixappha\neg4\pos4'
python train.py --gamma_neg 4 --gamma_pos 5 --output '.\output\mfm_aug\voc2007\fixappha\neg4\pos5'
python train.py --gamma_neg 5 --gamma_pos 1 --output '.\output\mfm_aug\voc2007\fixappha\neg5\pos1'
python train.py --gamma_neg 5 --gamma_pos 2 --output '.\output\mfm_aug\voc2007\fixappha\neg5\pos2'
python train.py --gamma_neg 5 --gamma_pos 3 --output '.\output\mfm_aug\voc2007\fixappha\neg5\pos3'
python train.py --gamma_neg 5 --gamma_pos 4 --output '.\output\mfm_aug\voc2007\fixappha\neg5\pos4'
python train.py --gamma_neg 5 --gamma_pos 5 --output '.\output\mfm_aug\voc2007\fixappha\neg5\pos5'