#!/bin/bash
#sudo su

set -e

cd $3

$1 $2 $4 $6 $5 $8 | tee -a ./$7_results.txt

cp ./$7_results.txt $4/results_$5.txt
