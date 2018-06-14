#!/bin/bash

# === UCF101 config
DATA_DIR_S="/home/xinqizhu/UCF101_frames_shuffled"
DATA_DIR_N="/home/xinqizhu/UCF101_frames"
TRAIN_LIST="/home/xinqizhu/ucfTrainTestlist/trainlist01.txt"
#GMM_OUT="/home/xinqizhu/UCF101_Fishers_shuffled/gmm_list"
GMM_OUT="/home/xinqizhu/UCF101_Fishers/gmm_list"
DATASET="UCF101"

trainlist01="/home/xinqizhu/ucfTrainTestlist/trainlist01.txt"
testlist01="/home/xinqizhu/ucfTrainTestlist/testlist01_cl.txt"

training_s_output="/home/xinqizhu/UCF101_Fishers_shuffled/train"
training_n_output="/home/xinqizhu/UCF101_Fishers/train"
#testing_output="/home/xinqizhu/UCF101_Fishers_shuffled/test_s_s"
#testing_output="/home/xinqizhu/UCF101_Fishers_shuffled/test_s_n"
testing_output="/home/xinqizhu/UCF101_Fishers/test_n_n"
#testing_output="/home/xinqizhu/UCF101_Fishers/test_n_s"

CLASS_INDEX="/home/xinqizhu/ucfTrainTestlist/classInd.txt"
CLASS_INDEX_OUT="./class_index"


#python computeFVs.py $DATA_DIR $trainlist01 $training_output $GMM_OUT $DATASET
#python computeFVs.py $DATA_DIR $testlist01 $testing_output $GMM_OUT $DATASET

python compute_class_index.py $CLASS_INDEX $CLASS_INDEX_OUT $DATASET
python score_inf.py $training_n_output $training_s_output\
    $testing_output \
    $trainlist01 $testlist01 10000 6000 "/home/xinqizhu/UCF101_Fishers" $DATASET \
    --load_n_pca /home/xinqizhu/UCF101_Fishers/n_pca \
    --load_n_classifier /home/xinqizhu/UCF101_Fishers/n_classifier \
    --save_s_pca /home/xinqizhu/UCF101_Fishers/s_pca \
    --save_s_classifier /home/xinqizhu/UCF101_Fishers/s_classifier
    #--load_pca /home/xinqizhu/UCF101_Fishers/pca \
    #--load_classifier /home/xinqizhu/UCF101_Fishers/classifier
    #--save_n_pca /home/xinqizhu/UCF101_Fishers/n_pca_test \
    #--save_n_classifier /home/xinqizhu/UCF101_Fishers/n_classifier_test \
