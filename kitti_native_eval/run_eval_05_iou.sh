#!/bin/bash


set -e

cd $3


$1 $2 $4 $6 $5 $8 | tee -a ./$7_results_05_iou.txt

cp ./$7_results_05_iou.txt $4/results_05_iou_$5.txt
