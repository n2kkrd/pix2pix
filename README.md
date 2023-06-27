#### Pix2pix for image colorization
###### Python 3.6 and MXNet 1.1

Это  реализация сети Pix2pix с использованием Python и MXNet
В основном используется для раскрашивания черно-белых изображений, в частности видео.

Прямо сейчас сеть настроена на работу с вводом изображений размером 1x3x256x256.
Изображение будет разделено на каналы яркости и a, b, канал яркости
будет служить входными данными для генератора, который попытается создать тензор каналов a, b
размером 1x256, который мы введем в дискриминатор в последовательности с исходными каналами a, b.

#### Метрики визуализации
tensorboard --logdir=./logs/train

##### Код основан на 
https://github.com/affinelayer/pix2pix-tensorflow
https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter14_generative-adversarial-networks/pixel2pixel.ipynb

##### pix2pix paper
https://arxiv.org/abs/1611.07004

####
Привет колоризации видео:
![Video colorization, captured and colorized in real-time using notebook and usb web-cam](https://s3-us-west-1.amazonaws.com/pix2pix/real_time_video_colorization.gif)