FROM python:3.7
MAINTAINER Natalia Litivinova <natalia.litvinova@canonical.com>

RUN apt-get update \
           && apt-get install -y \
              build-essential \
              cmake \
              git \ 
              wget \
              unzip \
              yasm \
              pkg-config \
              libswscale-dev \
              libtbb2 \
              libtbb-dev \
              libjpeg-dev \
              libpng-dev \
              libtiff-dev \
              libavformat-dev \
              libpq-dev \
              libgtk2.0-dev

RUN pip install numpy \
                torch==1.4.0 \
                torchvision==0.5.0 \
                opencv-contrib-python==3.4.2.17

WORKDIR /home/FasterRetinaFace
COPY images/ images/
COPY weights/ weights/
COPY src/ src/
COPY README.md .

CMD ["/bin/sh" "-c" "python src/main.py"]
