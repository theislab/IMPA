#!/bin/bash

cd /storage/groups/ml01/workspace/alessandro.palma/cellpainting
for dataset in dataset06.tar dataset07.tar
do
echo Downloading dataset $dataset 
wget https://ml.jku.at/software/cellpainting/dataset/$dataset
done 
