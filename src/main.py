import cv2
import torch
import argparse
from torch.backends import cudnn

from transform import DetectorPreprocessor, DetectorPostprocessor
from loader import VideoLoaderOpenCV, VideoLoaderWebCam
from model import RetinaFace


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video-path', default=None)
    parser.add_argument('--loading-from', default='webcam')
    parser.add_argument('--loader', default='opencv')
    parser.add_argument('--weights-path', default='../weights/mobilenet0.25_Final.pth')
    parser.add_argument('--target-size', default='480,640', help='target size in height,width format')
    parser.add_argument('--device', default='cpu', help='number of GPU for inference or "cpu"')
    parser.add_argument('--conf-th', default=0.5, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    test_image = cv2.resize(cv2.imread('../images/fossa.jpeg'), (150, 150))
    cudnn.benchmark = True
    args = parse_args()
    device = torch.device(args.device) if args.device == 'cpu' else torch.device(f'cuda:{args.device}')
    print('Inference device:', device)

    model = RetinaFace()
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval().to(device)

    target_size = tuple(map(int, args.target_size.split(',')))
    preprocessor = DetectorPreprocessor(
        target_size=target_size,
        mean=(104., 117., 123.),
        std=(1., 1., 1.),
        cvt2rgb=False,
        normalize=False,
        device=device
    )
    postprocessor = DetectorPostprocessor(
        target_size=target_size,
        device=device,
        nms_th=0.5,
        conf_th=args.conf_th
    )

    if args.loading_from == 'webcam' and args.loader == 'opencv':
        loader = VideoLoaderWebCam(transform=preprocessor, device=device)
    elif args.loading_from == 'video_path' and args.loader == 'opencv':
        raise NotImplementedError()
    else:
        raise RuntimeError()

    while True:
        data = loader()
        if not data['status']: break

        frame_orig = data['frame_orig']
        frame = data['image']
        ratio = data['ratio']
        pad = data['pad']

        boxes, scores, landms = postprocessor(model(frame), ratio, pad)
        boxes, scores, landms = tuple(map(lambda x: x.data.cpu().numpy(), (boxes, scores, landms)))
        landms = landms.reshape((landms.shape[0], 5, 2))

        for box, score, kpoints in zip(boxes, scores, landms):
            x1, y1, x2, y2 = list(map(int, box))

            frame_orig = cv2.rectangle(frame_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame_orig = cv2.putText(frame_orig, f'{score:.4f}', (x1, y1), 1, 1.5, (0, 255, 0))
            for idx, point in enumerate(kpoints):
                if idx == 2:
                    x, y = list(map(int, point))
                    pad_y, pad_x = test_image.shape[0] // 2, test_image.shape[1] // 2
                    if x - pad_x > 0 and x + pad_x < frame_orig.shape[1] and y - pad_y > 0 and y + pad_y < frame_orig.shape[0]:
                        frame_orig[y-pad_y: y+pad_y, x-pad_x: x+pad_x] = test_image
                # frame_orig = cv2.circle(frame_orig, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
                # frame_orig = cv2.putText(frame_orig, f'{idx}', (x, y), 1, 1.5, (0, 0, 255))

        cv2.imwrite('../saved/frame_orig.png', frame_orig)
        #cv2.imshow('frame_with_dets', frame_orig)
        #if cv2.waitKey(1) == 27:
        #    break
