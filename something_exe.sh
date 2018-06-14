#!/bin/bash

# === Something config
DATA_DIR="/home/xinqizhu/Something_frames"
TRAIN_LIST="/home/xinqizhu/repo/TRN-pytorch/video_datasets/something/train_videofolder.txt"
GMM_OUT="/home/xinqizhu/repo/CS221_Project/Something_Fishers/gmm_list"
DATASET="Something"

trainlist01="/home/xinqizhu/repo/TRN-pytorch/video_datasets/something/train_videofolder.txt"
testlist01="/home/xinqizhu/repo/TRN-pytorch/video_datasets/something/val_videofolder.txt"

training_output="/home/xinqizhu/repo/CS221_Project/Something_Fishers/train"
testing_output="/home/xinqizhu/repo/CS221_Project/Something_Fishers/test"

CLASS_INDEX="/home/xinqizhu/repo/TRN-pytorch/video_datasets/something/category.txt"
CLASS_INDEX_OUT="./class_index"

#python computeFVs.py $DATA_DIR $trainlist01 $training_output $GMM_OUT $DATASET
#python computeFVs.py $DATA_DIR $testlist01 $testing_output $GMM_OUT $DATASET

python compute_class_index.py $CLASS_INDEX $CLASS_INDEX_OUT $DATASET
python classify_experiment.py $training_output $testing_output $trainlist01 $testlist01 10000 2000 "./Something_Fishers" $DATASET

