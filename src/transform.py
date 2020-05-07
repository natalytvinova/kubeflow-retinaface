import cv2
import torch
import numpy as np
from math import ceil
from torchvision.ops import nms
from itertools import product as product


class ResizeKeepAspectRatio:
    def __init__(self, target_size):
        assert target_size[0] <= target_size[1]
        self.target_size = target_size

        self.new_size = None
        self.ratio = None
        self.pads = None

    def __call__(self, image):
        if self.ratio is None:
            target_size = self.target_size

            old_size = image.shape[:2]  # old_size is in (height, width) format
            ratios = [float(i) / float(j) for i, j in zip(target_size, old_size)]
            min_ratio_index = 0 if ratios[0] < ratios[1] else 1
            min_ratio = ratios[min_ratio_index]

            self.ratio = min_ratio
            self.new_size = tuple([int(x * min_ratio) for x in old_size])

            # new_size should be in (width, height) format
            img = cv2.resize(image, (self.new_size[1], self.new_size[0]))
            # img = cv2.resize(image, None, fx=min_ratio, fy=min_ratio)
            delta_w = target_size[1] - self.new_size[1]
            delta_h = target_size[0] - self.new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            self.pads = [top, bottom, left, right]
        else:
            top, bottom, left, right = self.pads
            img = cv2.resize(image, (self.new_size[1], self.new_size[0]))

        new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return {
            'image': new_im,
            'ratio': self.ratio,
            # 'pad': [top, left]
            'pad': [self.pads[0], self.pads[2]]
        }


class Normalize:
    def __init__(self, mean, std, normalize):
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __call__(self, image):
        if self.normalize:
            image = image / 255.
        image = (image - self.mean) / self.std
        return image


class ToTensor:
    def __call__(self, image):
        return torch.from_numpy(image).float()


class DetectorPreprocessor:
    def __init__(self, target_size, mean, std, device, normalize=True, cvt2rgb=False):
        self.target_size = target_size
        self.normalize = normalize
        self.cvt2rgb = cvt2rgb
        self.device = device
        self.mean = mean
        self.std = std

        self.resize = ResizeKeepAspectRatio(target_size)
        self.normalize = Normalize(mean, std, normalize)
        self.to_tensor = ToTensor()

    def __call__(self, image):
        # TODO try kornia for speedup
        assert isinstance(image, np.ndarray), f'Expected image as np.ndarray, got {type(image)}'

        resized = self.resize(image)
        image_resized = resized['image']
        image_normalized = self.normalize(image_resized)
        image_tensor = self.to_tensor(image_normalized)
        image_tensor = image_tensor.to(self.device).permute((2, 0, 1)).unsqueeze(0)

        return {
            'image': image_tensor,
            'ratio': resized['ratio'],
            'pad': resized['pad']
        }


class PriorBox:
    def __init__(self, image_size):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def __call__(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class DetectorPostprocessor:
    def __init__(self, target_size, device, nms_th=0.5, conf_th=0.05):
        self.target_size = target_size
        # Generate anchors in initialization 'caz don't expect change of image during inference
        self.anchors = PriorBox(target_size)()
        self.anchors = self.anchors.to(device)
        self.nms_th = nms_th
        self.conf_th = conf_th
        self.device = device

    @staticmethod
    def decode(loc, priors, variances):
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def decode_landm(pre, priors, variances):
        landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), dim=1)
        return landms

    def __call__(self, output, ratio, pad):
        loc, conf, landms = output
        # scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        scores = conf.squeeze(0)[:, 1]
        boxes = self.decode(loc.data.squeeze(0), self.anchors.data, [0.1, 0.2])
        landms = self.decode_landm(landms.data.squeeze(0), self.anchors.data, [0.1, 0.2])

        boxes *= torch.Tensor([*self.target_size[::-1], *self.target_size[::-1]]).to(self.device)
        # TODO Convert boxes coords to original shape
        boxes[:, 0] += pad[1]
        boxes[:, 1] += pad[0]
        boxes *= ratio

        # TODO Convert landmarks coords to original shape
        landms *= torch.Tensor([self.target_size[::-1] for _ in range(5)]).to(self.device).view(-1)
        landms[:, 0] += pad[1]
        landms[:, 1] += pad[0]
        landms *= ratio
        # landms = landms.data.cpu().numpy()

        inds = torch.where(scores > self.conf_th)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1]
        # boxes = boxes[order]
        # landms = landms[order]
        # scores = scores[order]

        # do NMS
        keep = nms(boxes, scores, self.nms_th)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        boxes = boxes[keep]
        landms = landms[keep]
        scores = scores[keep]

        return boxes, scores, landms


