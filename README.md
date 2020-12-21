# FaceID

This is a proof-of-concept of a computer vision system used for face detection and recognition in a video stream using a face database of known persons. The computer vision functionality is based on [Dlib](http://dlib.net/), [OpenCV](https://opencv.org/), and [TensorFlow](https://www.tensorflow.org/). The classification algorithms are built using [scikit-learn](https://scikit-learn.org/stable/), [NumPy](https://numpy.org/), and [SciPy](https://www.scipy.org/).

## Description

The project root directory contains the main CLI script `faceid.py`. The project settings are stored in the `cfg` directory which includes the following:

* File `models.json` contains paths to pre-trained models of face detector, encoder, and classifier.
* File `settings.py` stores module settings.
* The `video` folder contains `json` configuration files used to initialize video devices.

The `faceid.py` project's CLI is used to:

* Start video streaming and displaying recognized faces.
* Create and initialize a face database. The database is just a folder with subfolders named after class labels (names) of known persons. For example:

```bash
.
├── Alexander
│   ├── 02d1cb978a0f4273950cdc73e60ea27c_aligned.jpg
│   ├── 02d1cb978a0f4273950cdc73e60ea27c.jpg
│   ├── 02d1cb978a0f4273950cdc73e60ea27c.npy
│   ...
├── Konstantin
│   ├── 014c2acc91a642ffbf4021ef5bbb6a85_aligned.jpg
│   ├── 014c2acc91a642ffbf4021ef5bbb6a85.jpg
│   ├── 014c2acc91a642ffbf4021ef5bbb6a85.npy
│   ...
└── Pavel
    ├── 0251ece9d7d94d3cb7832f7c4a247898_aligned.jpg
    ├── 0251ece9d7d94d3cb7832f7c4a247898.jpg
    ├── 0251ece9d7d94d3cb7832f7c4a247898.npy
    ...
```
Here, files with `.npy` extension are vector embeddings of faces used by a classifier to predict faces found by a face detector in a video stream. Files with `_aligned.jpg` suffix store cropped and aligned faces found on the corresponding images. The aligned faces are used to build face embeddings.

* Train and test classifier model built on top of [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
* Initialize, configure, and read video devices in real time.

The process of detection and recognition of faces in a video stream could be briefly explained as following:

1. Frames from a video device are fetched in a separate stream. A separate thread is created for each device.
2. Frames from all streams are sent to a queue for recognition.
3. Recognition occurs in separate threads, which constantly take frames from the queue in batches.
4. The recognition process includes sequential execution of the following steps for each received frame:
    * The face detector model detects positions of faces on the input image.
    * Next, the found faces are cut out and sent as inputs to the face landmark model whose predictions are used for face alignment.
    * After that, the aligned faces are fed to the face encoder model, which produces vector representations of faces (face embeddings).
    * The resulting embeddings are fed to the face classifier model, which assigns a class label to each found face if the prediction probability is not lower than a defined threshold. Otherwise, the face is marked as `Unknown_Face`.
5. The obtained results then become available to the stream reading thread.
6. Further, the found faces are passed to an object tracker, which assigns them unique ID numbers and tracks their positions in the frame and their statuses. It updates states of the recognized objects in the global person register.

## Installation

It is required to download pre-trained models first by invoking the `download_models.sh` script:
```bash
./bin/download_models.sh
```
As a result, in the project's root a folder named `models` will be created with the following content:
```bash
ssd_mobilenet_v1_coco_2017_11_17           # person detector
dlib_face_recognition_resnet_model_v1.dat  # face encoder
mmod_human_face_detector.dat               # face detector
shape_predictor_5_face_landmarks.dat       # 5 points face landmark detector
shape_predictor_68_face_landmarks.dat      # 68 points face landmark detector (used by default)
```
Then, install dependencies listed in the `requirements.txt` file provided in the project's root folder. You can use `pip` to install them:
```bash
pip3 install --user -r requirements.txt
```
Also, it is required to install [NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=en-us) drivers, [CUDA](https://developer.nvidia.com/cuda-zone), and [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) libraries by following the official instructions. Then, install [Dlib](http://dlib.net/) library with the CUDA support by compiling it from source:
```bash
apt-get update && apt-get install -y \
    gcc-6 \
    g++-6 \
    libopenblas-dev \
    liblapack-dev git

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 50

git clone -b 'v19.17' --single-branch https://github.com/davisking/dlib.git
cd dlib; mkdir build

cmake -B ./build -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build ./build

sudo python3 setup.py install
```
Also, in order to be able to display video streams, you will need the following libraries installed:
```bash
apt-get install -y \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev
```

## Usage

First, before actually reading a video stream, detect, and recognize found faces, it is required to initialize a database of known faces, and use it to train the classifier model. In order to do this, you have to split face images between subfolders named after class labels (names) assigned to corresponding persons. Then, initialize the resulting face database:
```bash
./faceid.py db init -f /path/to/face/db
```
This creates face embeddings in subfolders (files with `npy` extension). Then, train a face classifier model using the initialized face database:
```bash
./faceid.py model train -f /path/to/face/db -o ./models/face_clf.pkl --optimize
```
Once the training is over, the classifier model `face_clf.pkl` will be saved to the `models` folder. You can test accuracy of the trained model using another face database as follows:
```bash
./faceid.py model test -f /path/to/other/db -m ./models/face_clf.pkl
```
Now, you can start reading video streams:
```bash
./faceid.py run --show
```
Note, that displaying video streams and recognized faces on them (the `--show` option) dramatically slows down the whole system, so you should use it for debug purposes only. In order to initialize video devices using configuration files `cfg/video/*.json`, run the following command:
```bash
./faceid.py video config
```
In order to reset device settings, run the previous command with the `--reset` option. It is possible to make an *ad hoc* face database by using a video stream from any available video device:
```bash
./faceid.py video live -c cfg/video/camera1.json -o /path/to/face/db/John_Smith
```
Here, by pressing the `s` button an image will be saved in the `John_Smith` folder. The `q` button stops the program.

## Training

In the case you have to train the face encoder model on the custom dataset, you may find instructions provided [here](https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems#question-can-i-re-train-the-face-encoding-model-to-make-it-more-accurate-for-my-images), and [here](http://dlib.net/dnn_metric_learning_on_images_ex.cpp.html) helpful.

