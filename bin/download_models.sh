#!/bin/bash

cd $(dirname $(dirname $(realpath $0)))
model_path=${1:-models}

if [ ! -e $model_path ]; then
    mkdir $model_path
fi

cd $model_path

dlib_models=(
    shape_predictor_68_face_landmarks.dat
    shape_predictor_5_face_landmarks.dat
    mmod_human_face_detector.dat
    dlib_face_recognition_resnet_model_v1.dat
)

for model in "${dlib_models[@]}"; do

    if [ ! -e $model ]; then
        wget http://dlib.net/files/${model}.bz2
        bunzip2 -d ${model}.bz2
        rm -f ${model}.bz2
    fi

done

person_detector=ssd_mobilenet_v1_coco_2017_11_17

if [ ! -e $person_detector ]; then
    mkdir $person_detector && cd $person_detector

    wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/${person_detector}.pbtxt
    wget http://download.tensorflow.org/person_detectors/object_detection/${person_detector}.tar.gz
    tar -zxvf ${person_detector}.tar.gz && rm -f ${person_detector}.tar.gz
    mv ${person_detector}/frozen_inference_graph.pb ${person_detector}.pb

    rm -rf $person_detector
fi

