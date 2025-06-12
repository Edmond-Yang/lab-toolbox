#!/bin/bash

VERSION=1.0
PUBLISH_DATE="2025-06-09"
EPOCH=200

echo "Candy's Lab rPPG ToolBox @2025"
echo "Version: $VERSION"
echo "Publish Date: $PUBLISH_DATE"


declare -A test_dict
test_dict=(
#        ["COHFACE_all"]="PURE_all"
#        ["VIPL_fold1"]="UBFC_all"
#        ["Tokyo_train"]="Tokyo_test"
#        ["UBFC_train"]="UBFC_test"
#        ["PURE_train"]="PURE_test"
#        ["PURE_all"]="PURE_test"
#        ["UBFC_all"]="UBFC_test"
#        ["VIPL_fold1"]="VIPL_fold2"
#        ["VIPL_fold2"]="VIPL_fold1"
#        ["VIPL_fold3"]="VIPL_fold4"
#        ["VIPL_fold4"]="VIPL_fold3"
#        ["VIPL_fold5"]="VIPL_fold6"
#        ["VIPL_fold6"]="VIPL_fold5"
#        ["COHFACE_all"]="COHFACE_test"
#        ["COHFACE_test"]="COHFACE_all"
#        ["IT_all"]="PURE_all"
#        ["PURE_all"]="VIPL_fold1"
#        ['UBFC_all']="VIPL_fold1"
)

model_name="foundation-model"


for model_name in "sinc"
do
    for train in "Car_GarSti_5fold_1,Car_GarSti_5fold_2,Car_GarSti_5fold_3,Car_GarSti_5fold_4" "Car_GarSti_5fold_5,Car_GarSti_5fold_2,Car_GarSti_5fold_3,Car_GarSti_5fold_4" "Car_GarSti_5fold_1,Car_GarSti_5fold_5,Car_GarSti_5fold_3,Car_GarSti_5fold_4" "Car_GarSti_5fold_1,Car_GarSti_5fold_2,Car_GarSti_5fold_5,Car_GarSti_5fold_4" "Car_GarSti_5fold_1,Car_GarSti_5fold_2,Car_GarSti_5fold_3,Car_GarSti_5fold_5"
    do
        if [[ $test == *"Car_all"* ]]; then
            echo python3 train.py --model $model_name --train "$train" --epoch $EPOCH
            python3 -u train.py --model $model_name --train "$train" --epoch $EPOCH
        else
            echo python3 train.py --model $model_name --train "$train" --epoch $EPOCH  --preload
            python3 -u train.py --model $model_name --train "$train" --epoch $EPOCH  --preload
        fi

#        for test in ${test_dict[$train]}
#        do
#            if [[ $test == *"Car_all"* ]]; then
#                echo python3 test.py --model $model_name --train "$train" --test "$test" --epoch $EPOCH
#                python3 -u test.py --model $model_name --train "$train" --test "$test" --epoch $EPOCH
#            else
#                echo python3 test.py --model $model_name --train "$train" --test "$test" --epoch $EPOCH  --preload
#                python3 -u test.py --model $model_name --train "$train" --test "$test" --epoch $EPOCH  --preload
#            fi
#        done

    done
done