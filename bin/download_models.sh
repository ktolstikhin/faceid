#!/bin/bash

cd $(dirname $(dirname $(realpath $0)))
model_path=${1:-models}

if [ ! -e $model_path ]; then
    mkdir $model_path
fi

cd $model_path

models=(
    shape_predictor_68_face_landmarks.dat
    shape_predictor_5_face_landmarks.dat
    mmod_human_face_detector.dat
    dlib_face_recognition_resnet_model_v1.dat
)

for model in "${models[@]}"; do

    if [ ! -e $model ]; then
        wget http://dlib.net/files/${model}.bz2
        bunzip2 -d ${model}.bz2
        rm -f ${model}.bz2
    fi

done

upper_body_detector=haarcascade_upperbody.xml

if [ ! -e $upper_body_detector ]; then
    wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/${upper_body_detector}
fi

