#!/bin/bash
#sudo su

set -e
#first arguments refers to the scripts' path
#echo "parameter 1: $1"
#echo "parameter 2: $2"
#echo "parameter 3: $3"
#echo "parameter 4: $4"
#echo "parameter 5: $5"
cd $3
#second refers to the threshold used by the model
#third is the checkpoint name


#!/bin/bash

#echo "$6 $5" | tee -a ./results.txt
# move $2 to $4
# move $3 to $5
#echo "$1 $2 $4 $6 $5  | tee -a ./results.txt"
$1 $2 $4 $6 $5 $8 | tee -a ./$7_results.txt

cp ./$7_results.txt $4/results_$5.txt

#echo "results: results_$5.txt"
#echo "$4/"
