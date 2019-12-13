# FaceID

Система компьютерного зрения, предназначенная для поиска и распознавания лиц в видео потоке на основе имеющейся базы изображений известных лиц. Функционал модулей компьютерного зрения реализован с использованием библиотек [Dlib](http://dlib.net/), [OpenCV](https://opencv.org/), а также [TensorFlow](https://www.tensorflow.org/). Алгоритмы классификации лиц построены с использованием фреймворков [scikit-learn](https://scikit-learn.org/stable/), [NumPy](https://numpy.org/) и [SciPy](https://www.scipy.org/).

## Описание

Управление всеми функциями проекта осуществляется через использование выполняемого скрипта `faceid.py`, находящегося в корневой директории проекта. Настройки проекта хранятся в директории `cfg`:

* Файл `models.json` хранит пути к предобученным моделям детектора, энкодера и классификатора лиц.
* Файл `settings.py` содержит настройки программных модулей проекта.
* В директории `video` находятся `json` файлы конфигураций используемых видео устройств.

Файл `faceid.py` является CLI проекта и реализует такие функции, как:

* Запуск процесса чтения из видео устройств и отображение найденных и распознанных лиц.
* Составление и инициализация базы данных лиц. База данных представляет из себя директорию, разбитую на поддиректории с названиями меток классов известных лиц. Например:

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
Здесь файлы с расширением `.npy` хранят векторные представления лиц, которые используются классификатором для идентификации лиц, найденных детектором лиц. Файлы с окончанием `_aligned.jpg` хранят вырезанные и выровненные изображения лиц, найденных на соответствующих изображениях. Выровненные изображения в свою очередь используются для создания векторных представлений лиц.

* Обучение и тестирование моделей классификатора лиц, построенной на базе алгоритма k ближайших соседей ([k-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)).
* Работа с видео устройствами, их конфигурирование и чтение видео потока в реальном времени.

Процесс поиска и распознавания лиц в видео потоке можно описать упрощённо в виде следующей последовательности:

1. В отдельном потоке считываются кадры из видео устройства. Для каждого устройства создаётся отдельный поток.
2. Далее кадры со всех потоков отправляются в очередь для распознавания.
3. Распознавание происходит в отдельных потоках, которые постоянно забирают пачками кадры из очереди.
4. Процесс распознавания включает в себя последовательное выполнение следующих шагов для каждого полученного кадра:
    * Модель детектора лиц определяет положения лиц на фотографии.
    * Далее найденные лица вырезаются и посылаются на вход модели определителя ключевых точек лиц для выравнивания.
    * После этого выравненные лица подаются на вход модели энкодера лиц для получения векторных представлений лиц (embeddings).
    * Полученные вектора лиц подаются на вход модели классификатора лиц, которая присваивает каждому найденному лицу метку класса, если вероятность полученного предсказания не ниже заданного порога. В противном случае лицо помечается как незнакомое (`Unknown_Face`).
5. Полученные результаты становятся доступными считывающему потоку.
6. Далее найденные лица передаются трекеру объектов, который присваивает им уникальные ID номера и отслеживает их положения в кадре и их статусы. При этом он обновляет состояния распознанных объектов в глобальном реестре лиц, который доступен в основном потоке программы для чтения и модификации.

## Установка

Для работы модуля необходимо скачать предобученные модели. Для этого запустите скрипт `download_models.sh`:
```bash
./bin/download_models.sh
```
В результате выполнения скрипта в корневой директории проекта появится папка `models`, содержащая файлы моделей:
```bash
ssd_mobilenet_v1_coco_2017_11_17           # детектор людей
dlib_face_recognition_resnet_model_v1.dat  # энкодер лиц
mmod_human_face_detector.dat               # детектор лиц
shape_predictor_5_face_landmarks.dat       # определитель 5 ключевых точек лиц
shape_predictor_68_face_landmarks.dat      # определитель 68 ключевых точек лиц (используется по умолчанию)
```
После этого установите зависимости, указанные в файле `requirements.txt`:
```bash
pip3 install --user -r requirements.txt
```
Кроме этого необходимо установить драйвера [NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=en-us), платформу для параллельных вычислений [CUDA](https://developer.nvidia.com/cuda-zone), а также библиотеку [NVIDIA cuDNN](https://developer.nvidia.com/cudnn), следуя официальным инструкциям. Далее необходимо установить библиотеку [Dlib](http://dlib.net/), перед этим скомпилировав её из исходников для обеспечения поддержки платформы CUDA:
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

## Использование

Пережде, чем приступить непосредственно к чтению видео потока, поиску и распознаванию найденных лиц, необходимо создать и инициализировать базу известных лиц, а также обучить на ней модель классификатора лиц. Для этого необходимо разбить изображения с лицами известных людей по папкам, названными согласно меткам классов, присвоенных данным людям. Далее необходимо инициализировать данную базу лиц:
```bash
./faceid.py db init -f /path/to/face/db
```
После этого в папках с изображениями известных лиц появятся векторные представления лиц (файлы с расширением `npy`). Далее нужно обучить модель классификатора, используя инициализированную базу данных:
```bash
./faceid.py model train -f /path/to/face/db -o ./models/face_clf.pkl --optimize
```
В результате обученная модель будет сохранена в файл `face_clf.pkl` в папке моделей `models`. Также можно протестировать полученную модель на какой-либо другой инициализированной базе данных следующим образом:
```bash
./faceid.py model test -f /path/to/other/db -m ./models/face_clf.pkl
```
После того, как была обучена модель классификатора, можно приступить к чтению видео потоков:
```bash
./faceid.py run --show
```
Нужно помнить, что отображение видео потоков и распознанных объектов в них (аргумент `--show`) существенно замедляет работу всей системы и поэтому должно использоваться только в отладочных целях. Для конфигурирования видео устройств, используя настройки `cfg/video/*.json`, выполните следующую команду:
```bash
./faceid.py video config
```
Для сброса настроек видео устройств, выполните предыдущую команду с аргументом `--reset`. Возможно создать *ad hoc* базу данных, используя видео поток с какого-либо устройства:
```bash
./faceid.py video live -c cfg/video/camera1.json -o /path/to/face/db/John_Smith
```
Далее при нажатии клавиши `s` будет происходить сохранение снимка экрана в указанную директорию `John_Smith`. При нажатии `q` чтение прекратится.

## Обучение

При необходимости обучения модели энкодера лиц на собранной базе лиц, например, азиатов, нужно следовать инструкциям, указанным [здесь](https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems#question-can-i-re-train-the-face-encoding-model-to-make-it-more-accurate-for-my-images), а также [данному примеру](http://dlib.net/dnn_metric_learning_on_images_ex.cpp.html).

## Авторы

Copyright (c) 2019 Konstantin Tolstikhin <k.tolstikhin@gmail.com>

