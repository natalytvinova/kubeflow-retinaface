import cv2
import torch
import numpy as np


class VideoLoaderOpenCV:
    def __init__(self, video_path, transform):
        self.cap = cv2.VideoCapture(video_path)
        self.transform = transform

    def __call__(self):
        ret, frame = self.cap.read()
        if not ret:
            return {
                'status': False
            }

        return {
            'frame_orig': frame,
            'status': True,
            **self.transform(frame)
        }


class VideoLoaderWebCam:
    def __init__(self, transform, device):
        self.cap = cv2.VideoCapture(0)
        self.transform = transform
        self.device = device

    def __call__(self):
        ret, frame = self.cap.read()
        return {
            'frame_orig': frame,
            'status': True,
            **self.transform(frame)
        }


class VideoLoaderDALI:
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass
