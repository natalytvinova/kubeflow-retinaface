FROM jjanzic/docker-python3-opencv:contrib-opencv-3.4.2
WORKDIR /home/FasterRetinaFace
COPY src/ .
COPY saved/ .
COPY weights/ .
COPY images/ .
COPY README.md .
RUN pip install torch==1.4.0 torchvision==0.5.0
#CMD python FasterRetinaFace/main.py
