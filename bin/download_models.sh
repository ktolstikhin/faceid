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

object_detector=ssd_mobilenet_v1_coco_2017_11_17

if [ ! -e $object_detector ]; then
    mkdir $object_detector && cd $object_detector

    wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/${object_detector}.pbtxt
    wget http://download.tensorflow.org/object_detectors/object_detection/${object_detector}.tar.gz
    tar -zxvf ${object_detector}.tar.gz && rm -f ${object_detector}.tar.gz
    mv ${object_detector}/frozen_inference_graph.pb ${object_detector}.pb

    rm -rf $object_detector
fi

