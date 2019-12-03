# FaceID

TODO

## Установка

Для работы модуля необходимо скачать модели определителя ключевых точек лиц, детектора лиц, а также модель энкодера лиц. Для этого запустите скрипт `download_models.sh`:
```bash
./bin/download_models.sh
```
После этого установите зависимости, указанные в файле `requirements.txt`:
```bash
pip3 install --user -r requirements.txt
```
Далее необходимо скомпилировать библиотеку [Dlib](http://dlib.net/) из исходников для обеспечения поддержки платформы параллельных вычислений CUDA:
```bash
apt-get update && apt-get install -y \
    gcc-6 \
    g++-6 \
    libopenblas-dev \
    liblapack-dev git

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 50

git clone -b 'v19.17' --single-branch https://github.com/davisking/dlib.git
cd dlib; make build

cmake -B ./build -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build ./build

sudo python3 setup.py install
```

## Использование

TODO

## Авторы

Copyright (c) 2019 Konstantin Tolstikhin <k.tolstikhin@gmail.com>

